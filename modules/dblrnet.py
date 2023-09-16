import os

import torch
import torch.nn as nn

from modules.bde_v2 import BDENet
from utilities.warper import back_warp_disp


def pixel_reshuffle(inputs, upscale_factor):
    """Rearranges elements in a tensor of shape ``[*, C, H, W]`` to a
    tensor of shape ``[C*r^2, H/r, W/r]``.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples:
        >>> input = autograd.Variable(torch.Tensor(1, 3, 12, 12))
        >>> output = pixel_reshuffle(input,2)
        >>> print(output.size())
        torch.Size([1, 12, 6, 6])
    """
    batch_size, channels, in_height, in_width = inputs.size()

    # // division is to keep data type unchanged. In this way, the out_height is still int type
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = inputs.contiguous().view(batch_size, channels, out_height, upscale_factor, out_width, upscale_factor)
    channels = channels * upscale_factor * upscale_factor

    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class Res_Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Res_Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        f = self.conv1(x)
        f = self.conv2(f)
        return f + x


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class FAB(nn.Module):
    def __init__(self, planes):
        super(FAB, self).__init__()
        self.conv1 = conv(planes, planes)
        self.resblock = Res_Conv2(planes, planes)
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, Feat_1, Feat_2):
        Feat = Feat_1 - Feat_2
        x = self.conv1(Feat)
        x = self.resblock(x)
        x = self.conv2(x)
        return x


class EDNet(nn.Module):
    def __init__(self, baseline):
        super(EDNet, self).__init__()
        self.G0 = 72
        kSize = 3
        self.S = 6
        self.baseline = baseline

        # number of RDB blocks, conv layers, out channels
        self.D = 6
        self.C = 5
        self.G = 48

        self.bde = BDENet(self.G0)
        # Shallow feature extraction net
        self.SFENet1_b = nn.Conv2d(3 * 4, self.G0, 5, padding=2, stride=1)
        self.SFENet2_b = nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize - 1) // 2, stride=1)

        self.SFENet1_e = nn.Conv2d(6 * 6 * 4, self.G0, 5, padding=2, stride=1)
        self.SFENet2_e = nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs_e = nn.ModuleList()
        for i in range(self.D):
            self.RDBs_e.append(
                RDB(growRate0=self.G0, growRate=self.G, nConvLayers=self.C)
            )

        self.RDBs_b = nn.ModuleList()
        for i in range(self.D):
            self.RDBs_b.append(
                RDB(growRate0=self.G0, growRate=self.G, nConvLayers=self.C)
            )
        
        
        # Mask-Net
        self.atten_e = nn.ModuleList()
        for i in range(self.S):
            self.atten_e.append(
                FAB(self.G0 // self.S)
            )
        
        self.atten_b = nn.ModuleList()
        for i in range(self.S):
            self.atten_b.append(
                FAB(self.G0 // self.S)
            )
        
        # Fuse
        self.fuse_e = nn.ModuleList()
        for i in range(self.D):
            self.fuse_e.append(
                nn.Conv2d(2 * self.G0, self.G0, kernel_size=1, stride=1, padding=0)
            )
        
        self.fuse_b = nn.ModuleList()
        for i in range(self.D):
            self.fuse_b.append(
                nn.Conv2d(2 * self.G0, self.G0, kernel_size=1, stride=1, padding=0)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d((self.D + 1) * self.G0, self.G0, 1, padding=0, stride=1),
            nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # Up-sampling net
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(self.G0, 256, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 21, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

    def forward(self, blurry, events):
        B_shuffle_events = pixel_reshuffle(events, 2)
        B_shuffle_blurry = pixel_reshuffle(blurry, 2)

        f__1_e = self.SFENet1_e(B_shuffle_events)
        x_e = self.SFENet2_e(f__1_e)
        f__1_b = self.SFENet1_b(B_shuffle_blurry)
        x_b = self.SFENet2_b(f__1_b)
        
        feat_e = x_e  # [bs, G0, H/2, W/2]
        feat_b = x_b
        RDBs_out_b = []

        for i in range(self.D):
            mid_e = self.RDBs_e[i](feat_e)  # [bs, G0, H/2, W/2]
            mid_b = self.RDBs_b[i](feat_b)  # [bs, G0, H/2, W/2]
            
            res_disps = self.bde(mid_b, mid_e) - 0.5  # [bs, 24, H/2, W/2]
            
            warped_0_e, _ = back_warp_disp(mid_e[:, 0                 : self.G0  //self.S, ...], res_disps[:, 0 :1 , ...]*self.baseline)
            warped_1_e, _ = back_warp_disp(mid_e[:, self.G0  //self.S : self.G0*2//self.S, ...], res_disps[:, 2 :3 , ...]*self.baseline)
            warped_2_e, _ = back_warp_disp(mid_e[:, self.G0*2//self.S : self.G0*3//self.S, ...], res_disps[:, 4 :5 , ...]*self.baseline)
            warped_3_e, _ = back_warp_disp(mid_e[:, self.G0*3//self.S : self.G0*4//self.S, ...], res_disps[:, 6 :7 , ...]*self.baseline)
            warped_4_e, _ = back_warp_disp(mid_e[:, self.G0*4//self.S : self.G0*5//self.S, ...], res_disps[:, 8 :9 , ...]*self.baseline)
            warped_5_e, _ = back_warp_disp(mid_e[:, self.G0*5//self.S : self.G0*6//self.S, ...], res_disps[:, 10:11, ...]*self.baseline)
            
            warped_0_b, _ = back_warp_disp(mid_b[:, 0                 : self.G0  //self.S, ...], res_disps[:, 1 :2 , ...]*self.baseline)
            warped_1_b, _ = back_warp_disp(mid_b[:, self.G0  //self.S : self.G0*2//self.S, ...], res_disps[:, 3 :4 , ...]*self.baseline)
            warped_2_b, _ = back_warp_disp(mid_b[:, self.G0*2//self.S : self.G0*3//self.S, ...], res_disps[:, 5 :6 , ...]*self.baseline)
            warped_3_b, _ = back_warp_disp(mid_b[:, self.G0*3//self.S : self.G0*4//self.S, ...], res_disps[:, 7 :8 , ...]*self.baseline)
            warped_4_b, _ = back_warp_disp(mid_b[:, self.G0*4//self.S : self.G0*5//self.S, ...], res_disps[:, 9 :10, ...]*self.baseline)
            warped_5_b, _ = back_warp_disp(mid_b[:, self.G0*5//self.S : self.G0*6//self.S, ...], res_disps[:, 11:12, ...]*self.baseline)

            mask_e_0 = self.atten_e[0](mid_e[:, 0                 : self.G0  //self.S, ...], warped_0_b)
            mask_e_1 = self.atten_e[1](mid_e[:, self.G0  //self.S : self.G0*2//self.S, ...], warped_1_b)
            mask_e_2 = self.atten_e[2](mid_e[:, self.G0*2//self.S : self.G0*3//self.S, ...], warped_2_b)
            mask_e_3 = self.atten_e[3](mid_e[:, self.G0*3//self.S : self.G0*4//self.S, ...], warped_3_b)
            mask_e_4 = self.atten_e[4](mid_e[:, self.G0*4//self.S : self.G0*5//self.S, ...], warped_4_b)
            mask_e_5 = self.atten_e[5](mid_e[:, self.G0*5//self.S : self.G0*6//self.S, ...], warped_5_b)
            masked_wb = torch.cat((mask_e_0*warped_0_b, mask_e_1*warped_1_b, mask_e_2*warped_2_b, mask_e_3*warped_3_b, mask_e_4*warped_4_b, mask_e_5*warped_5_b), 1)

            mask_b_0 = self.atten_b[0](mid_b[:, 0                 : self.G0  //self.S, ...], warped_0_e)
            mask_b_1 = self.atten_b[1](mid_b[:, self.G0  //self.S : self.G0*2//self.S, ...], warped_1_e)
            mask_b_2 = self.atten_b[2](mid_b[:, self.G0*2//self.S : self.G0*3//self.S, ...], warped_2_e)
            mask_b_3 = self.atten_b[3](mid_b[:, self.G0*3//self.S : self.G0*4//self.S, ...], warped_3_e)
            mask_b_4 = self.atten_b[4](mid_b[:, self.G0*4//self.S : self.G0*5//self.S, ...], warped_4_e)
            mask_b_5 = self.atten_b[5](mid_b[:, self.G0*5//self.S : self.G0*6//self.S, ...], warped_5_e)
            masked_we = torch.cat((mask_b_0*warped_0_e, mask_b_1*warped_1_e, mask_b_2*warped_2_e, mask_b_3*warped_3_e, mask_b_4*warped_4_e, mask_b_5*warped_5_e), 1)

            mid_e = torch.cat((mid_e, masked_wb), 1)
            mid_b = torch.cat((mid_b, masked_we), 1)

            feat_e = self.fuse_e[i](mid_e)
            feat_b = self.fuse_b[i](mid_b)
            RDBs_out_b.append(feat_b)
        
        RDBs_out_b.append(feat_e)
        x = self.GFF(torch.cat(RDBs_out_b, 1))
        x += f__1_b

        pred_shape_images = torch.split(self.UPNet(x) + torch.cat((blurry, blurry, blurry, blurry, blurry, blurry, blurry), 1), 3, 1)
        
        return pred_shape_images


