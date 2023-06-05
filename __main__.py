import tyro
from typing import  Union
from typing_extensions import Annotated
from utils.args import NeRFTrainingArgs,NeRFTestingArgs
from pathlib import Path
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "0" 
import torch

CmdTrain=Annotated[
        NeRFTrainingArgs,
        tyro.conf.subcommand(name="train",
                             default=NeRFTrainingArgs(train_data_filepath=Path("/home/cad_83/E/chenyingxi/my-nerf/data/nerf_synthetic/lego/transforms_train.json"),
                                                      valid_data_filepath=Path("/home/cad_83/E/chenyingxi/my-nerf/data/nerf_synthetic/lego/transforms_val.json"),
                                                      exp_dir=Path("/home/cad_83/E/chenyingxi/my-nerf/data/train2")))]
CmdTest=Annotated[
        NeRFTestingArgs,
        tyro.conf.subcommand(name="test",
                             default=NeRFTestingArgs(test_data_filepath=Path("/home/cad_83/E/chenyingxi/my-nerf/data/nerf_synthetic/lego/transforms_train.json"),
                                                      ckpt=Path("/home/cad_83/E/chenyingxi/my-nerf/data/train2/1000.tar"),
                                                      exp_dir=Path("/home/cad_83/E/chenyingxi/my-nerf/data/test1")
                                                      ))]
CmdArgs=Union[CmdTrain,CmdTest]

if __name__=="__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    args=tyro.cli(CmdArgs)
    if isinstance(args,NeRFTrainingArgs):
        from train import train
        train(args)
    elif isinstance(args,NeRFTestingArgs):
        from test import test
        test(args)