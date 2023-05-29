from dataclasses import dataclass
from pathlib import Path


    
@dataclass(frozen=True)
class NeRFTrainingArgs():
    #train data path
    train_data_filepath:Path
    #save path
    exp_dir:Path

@dataclass(frozen=True)
class NeRFTestingArgs():
    config:Path
    exp_dir:Path
    