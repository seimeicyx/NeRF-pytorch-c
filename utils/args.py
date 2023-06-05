from dataclasses import dataclass,field
from pathlib import Path
from typing import List
 
    
@dataclass(frozen=True)
class NeRFTrainingArgs():
    #train data path
    train_data_filepath:Path
    valid_data_filepath:Path
    #save path
    exp_dir:Path
    
    #L for points
    multires:int=10
    #L for views
    multires_views:int=4
    #layers in network
    netdepth:int=8
    #layers in fine network
    netdepth_fine:int=8
    #channels per layer'
    netwidth:int=256
    #channels per layer in fine network
    netwidth_fine:int=256
    #learning rate
    lrate:float=5e-4
    #number of pts sent through network in parallel, decrease if running out of memory
    netchunk:int=1024*64
    #set to 0. for no jitter, 1. for jitter
    perturb:float=1.
    #number of additional fine samples per ray
    N_importance:int=128
    #number of coarse samples per ray
    N_samples:int=64
    #use full 5D input instead of 3D
    use_viewdirs:bool=True
    #set to render synthetic data on a white bkgd (always use for dvoxels)
    white_bkgd:bool=True
    #std dev of noise added to regularize sigma_a output, 1e0 recommended
    raw_noise_std:float=0.
    #sampling linearly in disparity rather than depth
    lindisp:bool=False
    #number of rays processed in parallel, decrease if running out of memory
    chunk:int=1024*32
    #batch size (number of random rays per gradient step)
    N_rand=32*32*2
    #sample boundary
    near:float=2.
    far:float=6.
    #train iters
    N_iters:int=400000
    #number of steps to train on central crops
    precrop_iters:int=500
    #fraction of img taken for central crops
    precrop_frac:float=0.5
    #exponential learning rate decay (in 1000 steps)
    lrate_decay:int=500
    #iters to render a test img
    i_valid:int=1000
    #iters to save ckpt
    i_ckpt:int=10_0000
    #iters to log loss
    i_loss:int=100
    valid_num:int=1
@dataclass(frozen=True)
class NeRFTestingArgs():
    #test data path
    test_data_filepath:Path
    #ctpk path
    ctpk:Path
    #save path
    exp_dir:Path
    
    #L for points
    multires:int=10
    #L for views
    multires_views:int=4
    #layers in network
    netdepth:int=8
    #layers in fine network
    netdepth_fine:int=8
    #channels per layer'
    netwidth:int=256
    #channels per layer in fine network
    netwidth_fine:int=256
    #learning rate
    lrate:float=5e-4
    #number of pts sent through network in parallel, decrease if running out of memory
    netchunk:int=1024*64
    #set to 0. for no jitter, 1. for jitter
    perturb:float=1.
    #number of additional fine samples per ray
    N_importance:int=128
    #number of coarse samples per ray
    N_samples:int=64
    #use full 5D input instead of 3D
    use_viewdirs:bool=True
    #set to render synthetic data on a white bkgd (always use for dvoxels)
    white_bkgd:bool=True
    #std dev of noise added to regularize sigma_a output, 1e0 recommended
    raw_noise_std:float=0.
    #sampling linearly in disparity rather than depth
    lindisp:bool=False
    #number of rays processed in parallel, decrease if running out of memory
    chunk:int=1024*32
    #batch size (number of random rays per gradient step)
    N_rand=32*32*2
    #sample boundary
    near:float=2.
    far:float=6.
    #train iters
    N_iters:int=400000
    #number of steps to train on central crops
    precrop_iters:int=500
    #fraction of img taken for central crops
    precrop_frac:float=0.5
    #exponential learning rate decay (in 1000 steps)
    lrate_decay:int=500
    #iters to render a test img
    i_valid:int=1000
    #iters to save ckpt
    i_ckpt:int=10_0000
    #iters to log loss
    i_loss:int=100
    valid_num:int=1