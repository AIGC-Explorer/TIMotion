import torch.nn as nn
import torch
from models.layers import AdaLN

class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)

class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None, first=False):
        super().__init__()
        padding = dilation
        self.norm = norm
        self.first = first
        if norm == "LN":
            if not first:
                self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "AdaLN":
            if not first:
                self.norm1 = AdaLN(n_in)
            self.norm2 = AdaLN(n_in)
        elif norm == "GN":
            if not first:
                self.norm1 = nn.GroupNorm(num_groups=16, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=16, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            if not first:
                self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
            
        elif activation == "silu":
            self.activation1 = nn.SiLU()
            self.activation2 = nn.SiLU()
            
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
            

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0,)     


    def forward(self, x, emb):
        x_orig = x
        if not self.first:
            if self.norm == "LN":
                x = self.norm1(x)
                x = self.activation1(x)
            elif self.norm == "AdaLN":
                x = self.norm1(x, emb)
                x = self.activation1(x)
            else:
                x = self.norm1(x.transpose(1, 2))
                x = self.activation1(x).transpose(1, 2)
        
        x = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()

        if self.norm == "LN":
            x = self.norm2(x)
            x = self.activation2(x)
        elif self.norm == "AdaLN":
            x = self.norm2(x, emb)
            x = self.activation2(x)
        else:
            x = self.norm2(x.transpose(1, 2))
            x = self.activation2(x).transpose(1, 2)

        x = self.conv2(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = x + x_orig
        return x

class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='silu', norm=None, first=False):
        super().__init__()
        
        
        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm, first=first) for depth in range(n_depth)]
        if reverse_dilation:
            self.blocks = blocks[::-1]
        
        self.blocks = nn.ModuleList(blocks)
        
        # self.model = nn.Sequential(*blocks)

    def forward(self, x, emb):   
        for block in self.blocks:
            x = block(x, emb)     
        return x
        # return self.model((x, emb))