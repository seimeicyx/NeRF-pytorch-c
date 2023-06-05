import torch
from utils.data import load_jsondata,Target_img_tensor
from utils.camera import SceneMeta,PinholeCamera
from utils.render import render
from loguru import logger
from models.nerf import mse
from utils.args import NeRFTestingArgs
from models.nerf import create_nerf,mse
def test(args:NeRFTestingArgs):
    args.exp_dir.mkdir(exist_ok=True)
    test_targetImg_ten,scene_test=load_jsondata(args.test_data_filepath)
    
    render_kwargs_train,render_kwargs_test,start,grad_vars,optimizer=create_nerf(args,_device)
    bds_dic={
        'far':args.far,
        'near':args.near
    }
    render_kwargs_train.update(bds_dic)
    render_kwargs_test.update(bds_dic)