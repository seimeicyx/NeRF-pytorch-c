from dataclasses import dataclass,field
from typing import Any,List
import numpy as np
import torch
@dataclass
class FreEncoder():
    L:int
    out_dim:int=0
    fns:List=[]
    def __post_init__(self) -> Any:
        fres=[np.sin,np.cos]
        for i in range(self.L):
            for fre in fres:
                self.fns.append(lambda x:fre(2**i*np.pi*x))
                self.out_dim+=3
        self.fns.append(lambda x:x)
        self.out_dim+=3    
    def __call__(self,inputs:Any) -> Any:
        return torch.cat([fn(inputs) for fn in self.fns],dim=-1)