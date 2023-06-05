import torch
from utils.data import load_jsondata,Target_img_tensor,save_imgs
from utils.camera import SceneMeta,PinholeCamera
from utils.render import render
from loguru import logger
from models.nerf import mse
from utils.args import NeRFTestingArgs
from models.nerf import create_nerf,mse
from utils.ckpt import load_ckpt
from utils.render import test_render
def test(args:NeRFTestingArgs):
    args.exp_dir.mkdir(exist_ok=True)
    test_targetImg_ten,scene_test=load_jsondata(args.test_data_filepath)
    
    _,render_kwargs_test,start,_,optimizer=create_nerf(args)
    bds_dic={
        'far':args.far,
        'near':args.near
    }

    render_kwargs_test.update(bds_dic)
    # _,optimizer,model,model_fine=load_ckpt(args.ctpk,optimizer,render_kwargs_test['network_fn'],render_kwargs_test['network_fine'])
    # render_kwargs_test['network_fn'],render_kwargs_test['network_fine']=model,model_fine
    with torch.no_grad():
        rgbs,disps=test_render(test_targetImg_ten,scene_test,chunk=args.chunk,test_num=args.valid_num,**render_kwargs_test)
        save_imgs(rgbs,args.exp_dir,"test{}".format(start))