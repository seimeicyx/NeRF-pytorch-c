import torch.nn as nn
import torch
import torch.nn.functional as F
class NeRF(nn.Module):
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
        self.view_linear=nn.Linear(self.W+self.view_ch,self.W//2)
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
        alpha=self.out_alpha(h)
        h=self.feature(h)
        h=torch.cat([h,view_x],dim=-1)
        
        h=self.view_linear(h)
        h=F.relu(h)
        rgb=self.out_rgb(h)
        return torch.cat([rgb,alpha],dim=-1)

def batch_run_model(pts,views,model,chunk):
    rets=[]
    for i in range(0,pts.shape[0],chunk):
        rets.append(model(torch.cat([pts[i:i+chunk],views[i:i+chunk]],dim=-1)))
    return torch.cat(rets,dim=0)
def apply_model(input_pts,input_views,pts_embed_fn,view_embed_fn,model,batch_chunk):
    pts_flat=torch.reshape(input_pts,shape=([-1,input_pts.shape[-1]]))
    pts_embeded=pts_embed_fn(pts_flat)
    views_exp=input_views[:,None,:].expand(input_pts.shape)
    views_flat=torch.reshape(views_exp,shape=([-1,views_exp.shape[-1]]))
    views_embeded=view_embed_fn(views_flat)
    outputs=batch_run_model(pts_embeded,views_embeded,model,batch_chunk)
    return torch.reshape(outputs,shape=(list(input_pts.shape[:-1])+[outputs.shape[-1]]))    