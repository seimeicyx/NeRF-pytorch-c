import torch.nn as nn
import torch
import torch.nn.functional as F
class NeRF(nn.modules):
    def __init__(self,pts_ch=3,view_ch=3,D=8,W=256,skips=[4]):
        super(NeRF,self).__init__()
        self.pts_ch=pts_ch
        self.view_ch=view_ch
        self.D=D
        self.W=W
        self.skips=skips
        #pts part
        self.pts_linears=nn.ModuleList([nn.Linear(self.pts_ch,self.W)]+
                                       [nn.Linear(self.W,self.W) if i not in self.skips
                                        else nn.Linear(self.W+self.pts_ch,self.W) 
                                        for i in range(self.D-1)])
        #view_dir part
        self.feature=nn.Linear(self.W,self.W)
        self.out_alpha=nn.Linear(self.W,1)
        self.view_linear=nn.Linear(self.W,self.W//2)
        self.out_rgb=nn.Linear(self.W//2,3)
    def forward(self,x):
        pts_x,view_x=torch.split(x,[self.pts_ch,self.view_ch],dim=-1)
        h=pts_x
        for i,linear in enumerate(self.pts_linears):
            h=linear(h)
            h=F.relu(h)
            if i in self.skips:
                h=torch.cat([pts_x,h],dim=-1)
        #view
        h=self.feature(h)
        alpha=self.out_alpha(h)
        h=torch.cat([h,view_x])
        h=self.view_linear(h)
        h=F.relu(h)
        rgb=self.out_rgb(h)
        return torch.cat([rgb,alpha],dim=-1)
            