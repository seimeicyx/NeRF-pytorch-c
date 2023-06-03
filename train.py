from loguru import logger
from utils.args import NeRFTrainingArgs
from utils.data import load_jsondata,Target_img_tensor,Target_img_np
from models.nerf import create_nerf
from utils.camera import SceneMeta,PinholeCamera
import torch
from typing import Any
import numpy as np
def sample_traindata(args:NeRFTrainingArgs,train_targetImg_ten: Target_img_tensor,scene_train: SceneMeta,iter:int):
    imgs=train_targetImg_ten.imgs
    poses=train_targetImg_ten.camera_poses
    img_i=np.random.randint(0, imgs.shape[0])
    
    target_img=imgs[img_i]
    pose=poses[img_i]
    
    H,W,K=scene_train.get_scene()
    cam=PinholeCamera(scene_meta=SceneMeta(H,W,K),camera_pose=pose)
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

def train(args:NeRFTrainingArgs,_device:Any):
    
    args.exp_dir.mkdir(exist_ok=True)
    train_targetImg,scene_train=load_jsondata(args.train_data_filepath)
    render_kwargs_train,render_kwargs_test,start,grad_vars,optimizer=create_nerf(args,_device)
    bds_dic={
        'far':args.far,
        'near':args.near
    }
    render_kwargs_train.update(bds_dic)
    render_kwargs_test.update(bds_dic)
    
    train_targetImg_ten=Target_img_tensor(train_targetImg.imgs,
                                          train_targetImg.camera_pose.to(_device))
    start+=1
    for i in range(start,args.N_iters):
       pass 