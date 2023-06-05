from pathlib import Path
from loguru import logger
import json
from PIL import Image
import numpy as np
from typing import Tuple,List
from dataclasses import dataclass
from utils.camera import SceneMeta
import torch
_device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
@dataclass
class Target_img_np():
    imgs:np.array
    camera_poses:np.array
@dataclass
class Target_img_tensor():
    imgs:torch.tensor
    camera_poses:torch.tensor
def load_imgf32(img_path:str)->np.array:
    img=Image.open(img_path)
    img=np.array(img,dtype=np.float32)
    return img/255.0
def load_jsondata(json_path:Path)->Tuple[Target_img_tensor,SceneMeta]:
    imgs=[]
    camera_poses=[]
    angle_x=0.5
    K=[]
    try :
        with json_path.open(mode="r") as file:
            meta=json.load(file)
            frames=meta['frames']
            basedir=json_path.parent
            for frame in frames:
                imgs.append(load_imgf32(str(basedir.joinpath(frame['file_path']+'.png'))))
                camera_poses.append(np.array(frame['transform_matrix'],dtype=np.float32))
            imgs=np.array(imgs,dtype=np.float32)
            camera_poses=np.array(camera_poses,dtype=np.float32)
            H,W=imgs.shape[1:-1]
            logger.debug("H,W:{},{}".format(H,W)) 
            angle_x= meta['camera_angle_x']
            focal=(W/2.)/np.tan(angle_x/2.)
            K=[[focal,0.0,W/2.],
               [0.0,focal,H/2.],
               [0.0,0.0,1.0]]
            imgs=imgs[...,:3]*imgs[...,-1:]+(1-imgs[...,-1:])*1.
    except BaseException as e:
        logger.error(e)
    finally:
        train_targetImg=Target_img_np(imgs=imgs,camera_poses=camera_poses)
        scene_train=SceneMeta(H=H,W=W,K=K)
        train_targetImg_ten=Target_img_tensor(torch.Tensor(train_targetImg.imgs),
                                          torch.Tensor(train_targetImg.camera_poses).to(_device)[:,:3,:4])
        return train_targetImg_ten,scene_train
imgf2u=lambda img:(np.clip(img,0,1)*255).astype(np.uint8)
def save_imgs(imgs:List,save_path:Path,iters:int):
    if not save_path.exists():
        save_path.mkdir(exist_ok=True)
    save_path=save_path.joinpath(str(iters))
    save_path.mkdir(exist_ok=True)
    for img in imgs:
        img=imgf2u(img)
        img=Image.fromarray(img)
        img.save(str(save_path.joinpath(str(iters)))+".png")
    logger.debug("saved images to :{}".format(save_path))