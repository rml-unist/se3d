"""Differentiable PyTorch equivalent of the upstream horizontal sampling kernel.

Left features repeat at every depth even when the right coordinate is outside.
Right features use horizontal interpolation on [0,width-1], zero elsewhere.
As upstream, calibration-derived shifts have no gradient.
"""
import torch
from torch import nn

class BuildCostVolume(nn.Module):
    def forward(self,left,right,shift):
        if left.shape!=right.shape or shift.ndim!=2 or shift.shape[0]!=left.shape[0]:raise ValueError('Invalid cost-volume input shapes')
        if not torch.all(shift>=0):raise ValueError('Negative stereo shift')
        n,c,h,w=left.shape;d=shift.shape[1]
        x=torch.arange(w,device=left.device,dtype=left.dtype)[None,None,:]-shift.detach().to(left)[:,:,None]
        valid=(x>=0)&(x<=w-1)
        low=x.floor().long().clamp(0,w-1);high=(low+1).clamp(max=w-1)
        alpha=(x-x.floor())[:,None,:,None,:]
        rows=torch.arange(h,device=left.device)[None,None,:,None]*w
        def sample(indices):
            index=(indices[:,:,None,:]+rows).reshape(n,1,d*h*w).expand(n,c,d*h*w)
            return right.reshape(n,c,h*w).gather(2,index).reshape(n,c,d,h,w)
        warped=(sample(low)*(1-alpha)+sample(high)*alpha)*valid[:,None,:,None,:]
        return torch.cat((left[:,:,None].expand(-1,-1,d,-1,-1),warped),dim=1)
