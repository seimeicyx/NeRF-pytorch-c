import torch
from pathlib import Path
import os
from loguru import logger
def save_ckpt(basedir:str,i:int,optimizer,**render_kwargs):
    _dir=Path(basedir)
    if not _dir.exists():
        _dir.mkdir()
    save_path=os.path.join(basedir,'{}.tar'.format(i))
    torch.save(
        {
            'global_step':i,
            'network_fn_state_dict':render_kwargs['network_fn'].state_dict(),
            'network_fine_state_dict':render_kwargs['network_fine'].state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
        },save_path
    )
    logger.info("save ckpt to :{}".format(save_path))

def load_ckpt(ckpt:str,optimizer,model,model_fine):
    ckpt_dict=torch.load(ckpt)
    start=ckpt_dict['global_step']
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    model.load_state_dict(ckpt['network_fn_state_dict'])
    model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    return start,optimizer,model,model_fine
    