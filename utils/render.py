import torch
import torch.nn.functional as F
from typing import Any
from utils.data import load_jsondata,Target_img_tensor
from utils.camera import SceneMeta,PinholeCamera
from loguru import logger
from models.nerf import mse
from tqdm import tqdm, trange
def raw2outputs(raw:torch.Tensor,z_vals:torch.Tensor,rays_d:torch.Tensor):
    rgb=raw[...,:3]
    delta=raw[...,-1]
    dists=z_vals[...,1:]-z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1) 
    # dists=torch.cat([torch.Tensor([1e-10]).expand(dists[...,:1].shape),dists],dim=-1)
    dists=dists*torch.norm(rays_d[...,None,:],dim=-1)
    
    alpha=1-torch.exp(-delta*dists)
    weights=alpha*torch.cumprod(torch.cat([torch.ones_like(alpha[...,:1]),1.-alpha+1e-10],dim=-1),dim=-1)[...,:-1]
    rgb_map=torch.sum(rgb*weights[...,None],dim=-2)
    depth_map=torch.sum(weights*z_vals,dim=-1)
    acc_map=torch.sum(weights,dim=-1)
    disp_map=1./torch.max((depth_map/acc_map),1e-10*torch.ones_like(depth_map))
    
    #backgroundcolor
    rgb_map=rgb_map+(1-acc_map[...,None])*1.0
    
    return rgb_map, disp_map, acc_map, weights, depth_map

def sample_pdf(bins:torch.Tensor,weights:torch.Tensor,Nf_samples:int,isTesting):    
    #weights(N_ray,N_sample-2)
    weights=weights+1e-5
    a=torch.sum(weights,dim=-1,keepdim=True)
    b=torch.sum(weights,dim=-1)
    pdf=weights/torch.sum(weights,dim=-1,keepdim=True)#(N_ray,Nc_sample-2)
    cdf=torch.cat([torch.zeros_like(pdf[:,:1]),torch.cumsum(pdf,dim=-1)],dim=-1)#(N_ray,Nc_sample-1)
    
    #sample
    u=torch.rand([cdf.shape[0],Nf_samples])#(N_ray,Nf_sample)
    
    #find cdfs&bins
    u.contiguous()
    inds=torch.searchsorted(cdf,u,right=True)#(N_ray,Nf_sample)
    lower_inds=torch.max(torch.zeros_like(inds),inds-1)
    upper_inds=torch.min(torch.ones_like(inds)*(cdf.shape[-1]-1),inds)
    bound_inds=torch.stack([lower_inds,upper_inds],dim=-1)
    
    _size=[inds.shape[0],inds.shape[1],cdf.shape[-1]]#[N_ray,Nf_sample,Nc_sample]
    bins_g=torch.gather(bins.unsqueeze(1).expand(_size),-1,bound_inds)#(N_ray,Nf_sample,2)
    cdfs_g=torch.gather(cdf.unsqueeze(1).expand(_size),-1,bound_inds)#(N_ray,Nf_sample,2)
    diff_cdf=cdfs_g[...,1]-cdfs_g[...,0]
    diff_cdf=torch.where(diff_cdf<1e-5,torch.ones_like(diff_cdf),diff_cdf)
    samples=bins_g[...,0]+((u-cdfs_g[...,0])/diff_cdf)*(bins_g[...,1]-bins_g[...,0])
    return samples

def render_batch_rays(ray_batch:torch.Tensor,
                      network_fn:Any,
                      network_query_fn:Any,
                      N_samples:int,
                      perturb:float=0.0,
                      N_importance=0,
                      network_fine:Any=None,
                      pytest:bool=False,
                      **kwargs
                      ):
    #parse data 
    #ray_batch[rays_o, rays_d, near, far, viewdirs]=[1024,3 3 1 1 3]
    rays_o, rays_d, near, far, viewdirs=torch.split(ray_batch,[3,3,1,1,3],-1)
    #sample Nc
    t_vals=torch.linspace(0.,1.,N_samples)
    z_vals=near*(1.-t_vals)+far*t_vals
    if perturb>0.0:
        mids=(z_vals[...,1:]+z_vals[...,:-1])*0.5
        lower=torch.cat([z_vals[...,:1],mids],-1)
        upper=torch.cat([mids,z_vals[...,-1:]],-1)
        dists=upper-lower
        
        t_rands=torch.rand(dists.shape)
        z_vals=lower+dists*t_rands
    
    pts=rays_o[...,None,:]+rays_d[...,None,:]*z_vals[...,None]
    #corse nwk
    raw=network_query_fn(pts,viewdirs,network_fn)
    rgb_map0, disp_map0, acc_map0, weights, depth_map0 = raw2outputs(raw, z_vals, rays_d)
    #fine nwk
    z_vals_mids=(z_vals[...,1:]+z_vals[...,:-1])*0.5
    z_samples=sample_pdf(z_vals_mids,weights[...,1:-1],N_importance,pytest)
    z_samples = z_samples.detach()
    z_vals,_=torch.sort(torch.cat([z_vals,z_samples], -1),-1)
    pts=rays_o[...,None,:]+rays_d[...,None,:]*z_vals[...,None]
    raw=network_query_fn(pts,viewdirs,network_fine)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d)
    ret = {'rgb_map' : rgb_map,
           'disp_map' : disp_map,
           'acc_map' : acc_map,
           'raw':raw,
           'rgb0':rgb_map0,
           'disp0':disp_map0,
           'acc0':acc_map0,
           'z_std':torch.std(z_samples, dim=-1, unbiased=False)
           }
    # for k in ret:
    #     if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
    #         print(f"! [Numerical Error] {k} contains nan or inf.")
    return ret

def batchify_rays(rays_flat:torch.Tensor, chunk:int=1024*32,istesing:bool=False, **kwargs):
    all_ret={}
    if istesing:
        for i in tqdm(range(0,rays_flat.shape[0],chunk),colour='red',desc="\033[0;31mRendering(per img)\033[0m"):
            ret=render_batch_rays(rays_flat[i:i+chunk],**kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k]=[]
                all_ret[k].append(ret[k])
    else:
        for i in range(0,rays_flat.shape[0],chunk):
            ret=render_batch_rays(rays_flat[i:i+chunk],**kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k]=[]
                all_ret[k].append(ret[k])
    all_ret={k:torch.cat(all_ret[k],0)for k in all_ret}
    return all_ret
def render(rays:torch.Tensor,istesing:bool=False,near=1.,far=6.,chunk:int=1024*32,**kwargs):
    #ray_batch[rays_o, rays_d, near, far, viewdirs]=[1024,3 3 1 1 3]
    rays_o, rays_d=rays
    sh=rays_o.shape[:-1]
    viewdirs=rays_d/torch.norm(rays_d,dim=-1,keepdim=True)
    near=near*torch.ones(list(rays_o.shape[:-1])+[1])
    far=far*torch.ones(list(rays_o.shape[:-1])+[1])
    #flatten
    rays_o=torch.reshape(rays_o,[-1,rays_o.shape[-1]])
    rays_d=torch.reshape(rays_d,[-1,rays_d.shape[-1]])
    viewdirs=torch.reshape(viewdirs,[-1,viewdirs.shape[-1]])
    near=torch.reshape(near,[-1,near.shape[-1]])
    far=torch.reshape(far,[-1,far.shape[-1]])
    rays_flat=torch.cat([rays_o, rays_d, near, far, viewdirs],-1)
    #render
    all_ret_flat=batchify_rays(rays_flat,chunk,istesing,**kwargs)
    #reshape
    all_ret={k:torch.reshape(all_ret_flat[k],list(sh)+[-1]) for k in all_ret_flat.keys()}
    k_extract = ['rgb_map', 'disp_map', 'acc_map','rgb0']
    extract_list=[all_ret[k] for k in k_extract]
    return extract_list+[all_ret]

def test_render(targetImg_ten: Target_img_tensor,scene_data: SceneMeta,test_num:int=1,chunk:int=32*32*4,**render_kwargs):
    test_num=max(min(test_num,targetImg_ten.imgs.shape[0]),0)
    imgs=targetImg_ten.imgs[:test_num]
    poses=targetImg_ten.camera_poses[:test_num]
    
    rgbs=[]
    disps=[]
    for i in tqdm(range(imgs.shape[0]),desc='\033[0;37;41mTesting\033[0m',colour='blue'):
        tg=imgs[i]
        pose=poses[i]
        cam=PinholeCamera(scene_meta=scene_data,camera_pose=pose)
        rays_o, rays_d = cam.cast_ray()
        rays=torch.stack([rays_o, rays_d],0)
        rgb, disp,_,_,_=render(rays,istesing=True,chunk=chunk,**render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        tqdm.write("\033[0;31mtest loss for img:{} is {}\033[0m".format(i,mse(rgb,tg)))
    return rgbs,disps