import tyro
from typing import  Union
from typing_extensions import Annotated
from utils.args import NeRFTrainingArgs,NeRFTestingArgs
from pathlib import Path

CmdTrain=Annotated[
        NeRFTrainingArgs,
        tyro.conf.subcommand(name="train",
                             default=NeRFTrainingArgs(train_data_filepath=Path("/home/cad_83/E/chenyingxi/my-nerf/data/nerf_synthetic/lego/transforms_train.json"),
                                                      exp_dir=Path("/home/cad_83/E/chenyingxi/my-nerf/data/train1")))]
CmdTest=Annotated[
        NeRFTestingArgs,
        tyro.conf.subcommand(name="test")]
CmdArgs=Union[CmdTrain,CmdTest]

if __name__=="__main__":
    args=tyro.cli(CmdArgs)
    if isinstance(args,NeRFTrainingArgs):
        from train import train
        train(args)
    elif isinstance(args,NeRFTestingArgs):
        pass