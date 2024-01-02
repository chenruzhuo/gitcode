import torch
import torch.nn as nn

class Resample2d(nn.Module):

    def __init__(self, kernel_size=1):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size  

    def forward(self, input1, input2):    
        return self.warp_right_to_left(input1,input2)
     

    @staticmethod
    def warp_right_to_left(x, disp, warp_grid=None):
        B, C, H, W = x.size()
        # mesh grid
        if warp_grid is not None:
            xx0, yy = warp_grid
            xx = xx0 + disp
            xx = 2.0*xx / max(W-1,1)-1.0
        else:
            xx = torch.arange(0, W, device=disp.device, dtype=x.dtype)
            yy = torch.arange(0, H, device=disp.device, dtype=x.dtype)
           
            xx = xx.view(1,-1).repeat(H,1)
            yy = yy.view(-1,1).repeat(1,W)

            xx = xx.view(1,1,H,W).repeat(B,1,1,1)
            yy = yy.view(1,1,H,W).repeat(B,1,1,1)

            # apply disparity to x-axis
            xx = xx + disp
            xx = 2.0*xx / max(W-1,1)-1.0
            yy = 2.0*yy / max(H-1,1)-1.0

        grid = torch.cat((xx,yy),1)

        vgrid = grid 
       
        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
       
        return output 
    
    

class ChannelNorm(nn.Module):

    def __init__(self, norm_deg=2):
        super(ChannelNorm, self).__init__()
        self.norm_deg = norm_deg

    def forward(self, input1):
        return torch.sqrt(torch.sum(torch.pow(input1, 2), dim=1, keepdim=True) + 1e-8)