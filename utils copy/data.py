from pathlib import Path
from loguru import logger
import json
from PIL import Image
import numpy as np
from typing import Tuple,List
from dataclasses import dataclass
@dataclass
class Loaded_data():
    imgs:np.array
    camera_poses:np.array
def load_imgf32(img_path:str)->np.array:
    img=Image.open(img_path)
    img=np.array(img,dtype=np.float32)
    return img/255.0
def load_jsondata(json_path:Path)->Tuple[Loaded_data,List]:
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
    except BaseException as e:
        logger.error(e)
    finally:
        load_data=Loaded_data(imgs=imgs,camera_poses=camera_poses)
        return load_data,K