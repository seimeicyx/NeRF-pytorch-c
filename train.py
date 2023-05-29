from loguru import logger
from utils.args import NeRFTrainingArgs
from utils.data import load_jsondata
def train(args:NeRFTrainingArgs):
    args.exp_dir.mkdir(exist_ok=True)
    train_data,K=load_jsondata(args.train_data_filepath)
    