from .layers import *
# from mmcls_custom.models.backbones.vrwkv6 import RWKV_Block
# from mmcls_custom.models.backbones_T.vrwkv6_T import RWKV_Block_T
import torch


class TransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim=512,
                 num_heads=8,
                 ff_size=1024,
                 dropout=0.,
                 cond_abl=False,
                 **kargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ca_block = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, x, y, emb=None, key_padding_mask=None):
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        h2 = self.ca_block(h1, y, emb, key_padding_mask)
        h2 = h2 + h1
        out = self.ffn(h2, emb)
        out = out + h2
        return out


class TIMotionTransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim=512,
                 num_heads=8,
                 ff_size=1024,
                 dropout=0.,
                 cond_abl=False,
                 cur_layer=0,
                 **kargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl

        self.sa_block = VanillaSelfAttention(latent_dim*2, num_heads, dropout, embed_dim=latent_dim)
        self.ffn = FFN(latent_dim*2, ff_size, dropout, latent_dim)
        
        self.LPA = kargs.get('LPA', False)
        if self.LPA:
            from mmcls_custom.models.backbones_T.resnet import Resnet1D
            self.linear = nn.Linear(2 * latent_dim, latent_dim)  # nn.Conv1d(n_embd*2, n_embd, 1, 1, 0,)
            self.conv = Resnet1D(latent_dim, kargs['cfg'].conv_layers, kargs['cfg'].dilation_rate, norm=kargs['cfg'].norm, first=(cur_layer==0))

    def forward(self, x, y, emb=None, key_padding_mask=None):
        b, t, c = x.size(0), x.size(1), x.size(2)

        inputs = torch.empty((b, t*2, 2*c), device=x.device)
        inputs[:, ::2, :c] = x
        inputs[:, 1::2, :c] = y
        inputs[:, ::2, c:] = y
        inputs[:, 1::2, c:] = x

        mask = torch.ones((b, t*2), dtype=torch.bool, device=x.device)
        index = torch.arange(t*2).to(x.device).unsqueeze(0).repeat(b, 1)
        valid_idx = (~key_padding_mask).sum(-1) * 2
        mask[index < valid_idx.unsqueeze(1)] = False

        h1 = self.sa_block(inputs, emb, mask)
        h1 = h1 + inputs
        
        if self.LPA:
            scan_1, scan_2 = h1[:, :, :c], h1[:, :, c:]
            outputs = torch.empty((b, t*2, 2*c), device=x.device)
            
            x[key_padding_mask] = 0.0
            y[key_padding_mask] = 0.0
            out_conv_x = self.conv(x, emb)
            out_conv_y = self.conv(y, emb)
            
            final_out_x = self.linear(torch.cat((scan_1[:, ::2]+scan_2[:, 1::2], out_conv_x), -1))
            final_out_y = self.linear(torch.cat((scan_2[:, ::2]+scan_1[:, 1::2], out_conv_y), -1))

            outputs[:, ::2, :c] = final_out_x
            outputs[:, 1::2, :c] = final_out_y
            outputs[:, ::2, c:] = final_out_y
            outputs[:, 1::2, c:] = final_out_x
            
            out = self.ffn(outputs, emb)
            out = out + outputs
            scan_1, scan_2 = out[:, :, :c], out[:, :, c:]
            
            return scan_1[:, ::2]+scan_2[:, 1::2], scan_2[:, ::2]+scan_1[:, 1::2]
        else:
            out = self.ffn(h1, emb)
            out = out + h1
            scan_1, scan_2 = out[:, :, :c], out[:, :, c:]
            
            return scan_1[:, ::2]+scan_2[:, 1::2], scan_2[:, ::2]+scan_1[:, 1::2]

