
import torch.nn as nn
from .bifpn_lpp import BiFPNModule
# from .bifpn import BiFPNModule

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)    # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in // 5, bias=False),
            nn.Sigmoid()
        ).cuda()

        self.bifpn = BiFPNModule(channels = 1024, levels = 5)

    def forward(self, x):
        b, c, _, _ = x.size()  # 32,5*1024,64,64
        y = self.avg_pool(x).view(b, c)     # shape:torch.Size([32, 5*1024])
        y = self.fc(y).view(b, c // 5, 1, 1)
        x_ = []
        x_.append(x[:,0:1024,:,:])
        x_.append(x[:,1024:2048,:,:])
        x_.append(x[:,2048:3072,:,:])
        x_.append(x[:,3072:4096,:,:])
        x_.append(x[:,4096:5120,:,:])
        out_x = self.bifpn(x_)
        out_x = out_x[0]+out_x[1]+out_x[2]+out_x[3]+out_x[4]
        return out_x * y.expand_as(out_x) 

        # only se_block
        # x_ = x_[0]+x_[1]+x_[2]+x_[3]+x_[4]
        # return x_ * y.expand_as(x_)             
       