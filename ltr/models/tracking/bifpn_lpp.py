import torch.nn as nn
import torch.nn.functional as F

import torch


class BiFPNModule_lpp(nn.Module):
    def __init__(self,
                 levels,
                 init=0.5,
                 activation=None,
                 eps=0.0001):
        super(BiFPNModule_lpp, self).__init__()
        self.activation = activation
        self.eps = eps
        self.levels = levels
        self.bifpn_convs = nn.ModuleList()
        # weighted
        self.w1 = nn.Parameter(torch.Tensor(1, levels).fill_(init))
        self.relu1 = nn.ReLU()
        
    def forward(self, inputs):
 
        # w relu
        w1 = self.relu1(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.eps  # normalize
        x_ = w1[0, 0] * inputs[:,0:1024,:,:] + w1[0, 1] * inputs[:,1024:2048,:,:] + w1[0, 2] * inputs[:,2048:3072,:,:] + w1[0, 3] * inputs[:,3072:4096,:,:] + w1[0, 4] * inputs[:,4096:5120,:,:]
        return x_

class BiFPNModule(nn.Module):
    def __init__(self,
                 channels,
                 levels,
                 init=0.5,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 eps=0.0001):
        super(BiFPNModule, self).__init__()
        self.activation = activation
        self.eps = eps
        self.levels = levels
        # self.bifpn_convs = nn.ModuleList()
        # weighted
        self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init))
        self.relu1 = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, levels - 2).fill_(init))
        self.relu2 = nn.ReLU()

    def forward(self, inputs):
        assert len(inputs) == self.levels
        # build top-down and down-top path with stack
        levels = self.levels
        # w relu
        w1 = self.relu1(self.w1)
        w1 = w1 / (torch.sum(w1, dim=0) + self.eps)  # normalize
        w2 = self.relu2(self.w2)
        w2 = w2 / (torch.sum(w2, dim=0) + self.eps)  # normalize
        # build top-down
        # idx_bifpn = 0
        pathtd = inputs
        inputs_clone = []
        for in_tensor in inputs:
            inputs_clone.append(in_tensor.clone())

        for i in range(levels - 1, 0, -1):
            pathtd[i - 1] = (w1[0, i-1]*pathtd[i - 1] + w1[1, i-1]*(
                        pathtd[i]))/(w1[0, i-1] + w1[1, i-1] + self.eps)
            # pathtd[i - 1] = self.bifpn_convs[idx_bifpn](pathtd[i - 1])          # lpp 去掉conv操作
            # idx_bifpn = idx_bifpn + 1
        # build down-top
        for i in range(0, levels - 2, 1):
            pathtd[i + 1] = (w2[0, i] * pathtd[i + 1] + w2[1, i] * pathtd[i] +
                              w2[2, i] * inputs_clone[i + 1])/(w2[0, i] + w2[1, i] + w2[2, i] + self.eps)
            # pathtd[i + 1] = self.bifpn_convs[idx_bifpn](pathtd[i + 1])      # lpp 去掉conv操作
            # idx_bifpn = idx_bifpn + 1
        pathtd[levels - 1] = (w1[0, levels-1] * pathtd[levels - 1] + w1[1, levels-1] * 
            pathtd[levels - 2])/(w1[0, levels-1] + w1[1, levels-1] + self.eps)
        # pathtd[levels - 1] = self.bifpn_convs[idx_bifpn](pathtd[levels - 1])
        return pathtd