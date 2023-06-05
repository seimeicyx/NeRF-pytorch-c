from loguru import logger
from utils.args import NeRFTrainingArgs
from utils.data import load_jsondata,Target_img_tensor,save_imgs
from models.nerf import create_nerf,mse
from utils.camera import SceneMeta,PinholeCamera
import torch
from typing import Any
import numpy as np
from utils.render import render, test_render
from tqdm import tqdm, trange
from utils.ckpt import save_ckpt
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
def sample_traindata(args,train_targetImg_ten: Target_img_tensor,scene_train: SceneMeta,iter:int):
    imgs=train_targetImg_ten.imgs
    poses=train_targetImg_ten.camera_poses
    img_i=np.random.randint(0, imgs.shape[0])
    
    target_img=imgs[img_i]
    pose=poses[img_i]
    
    H,W,K=scene_train.get_scene()
    cam=PinholeCamera(scene_meta=scene_train,camera_pose=pose)
    rays_o, rays_d = cam.cast_ray()
    if iter<args.precrop_iters:
        dH=int((H*args.precrop_frac)//2)
        dW=int((W*args.precrop_frac)//2)
        coords=torch.stack(torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
    else:
        coords=torch.stack(torch.meshgrid(torch.arange(H),torch.arange(W)),dim=-1)
    coords=torch.reshape(coords,[-1,2])
    select_inds=np.random.choice(coords.shape[0],size=args.N_rand,replace=False)
    select_coords=coords[select_inds].long()
    rays_o=rays_o[select_coords[:,0],select_coords[:,1]]
    rays_d=rays_d[select_coords[:,0],select_coords[:,1]]
    batch_rays=torch.stack([rays_o,rays_d],0)
    target_s=target_img[select_coords[:,0],select_coords[:,1]]
    return batch_rays,target_s

def train(args:NeRFTrainingArgs):
    
    args.exp_dir.mkdir(exist_ok=True)
    train_targetImg_ten,scene_train=load_jsondata(args.train_data_filepath)
    valid_targetImg_ten,_=load_jsondata(args.valid_data_filepath)
    render_kwargs_train,render_kwargs_test,start,grad_vars,optimizer=create_nerf(args)
    bds_dic={
        'far':args.far,
        'near':args.near
    }
    render_kwargs_train.update(bds_dic)
    render_kwargs_test.update(bds_dic)
    
    
    start+=1
    pbar=tqdm(range(start,args.N_iters+1),desc='\033[0;37;41mProcessing\033[0m',colour='pink',\
                postfix=dict)
    for i in pbar:
        #todo:
        batch_rays,target_s=sample_traindata(args,train_targetImg_ten,scene_train,i) 
        if i==50:
            a=52
        rgb, disp, acc,rgb0, rets_dict=render(batch_rays,chunk=args.chunk,**render_kwargs_train)
        loss=mse(rgb,target_s)
        psnr=mse2psnr(loss)
        loss+=mse(rgb0,target_s)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (i / decay_steps))
        pbar.set_postfix(**{'\033[0;31mLOSS\033[0m':'{:.10f}'.format(loss), '\033[0;31mPSNR\033[0m': '{}'.format(float(psnr))})
        pbar.update(1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        if i%args.i_loss==0:
            tqdm.write("\033[0;31mTraining step:{}    loss:{}\033[0m".format(i,loss))
        if i%args.i_valid==0:
            with torch.no_grad():
                rgbs,disps=test_render(valid_targetImg_ten,scene_train,chunk=args.chunk,test_num=args.valid_num,**render_kwargs_test)
                save_imgs(rgbs,args.exp_dir,i)
        if i%args.i_ckpt==0:
            save_ckpt(args.exp_dir,i,optimizer,**render_kwargs_train)
            
            