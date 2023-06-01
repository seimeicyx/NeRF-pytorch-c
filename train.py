from loguru import logger
from utils.args import NeRFTrainingArgs
from utils.data import load_jsondata
from models.nerf import create_nerf

def train(args:NeRFTrainingArgs):
    args.exp_dir.mkdir(exist_ok=True)
    train_targetImg,scene_train=load_jsondata(args.train_data_filepath)
    create_nerf(args)
    