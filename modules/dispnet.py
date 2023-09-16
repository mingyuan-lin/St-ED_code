import os

import torch
import torch.nn as nn
import torch.nn.functional as F

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

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.1,inplace=True)
    )


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Deconv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Deconv2, self).__init__()
        self.conv1 = deconv(in_planes, out_planes)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class PyramidAttention(nn.Module):
    def __init__(self, planes, stride=1):
        super(PyramidAttention, self).__init__()
        self.conv01 = nn.Conv2d(planes, planes // 2, kernel_size=1, stride=stride, padding=0)
        self.conv03 = nn.Conv2d(planes, planes // 2, kernel_size=3, stride=stride, padding=1)
        self.conv05 = nn.Conv2d(planes, planes // 2, kernel_size=5, stride=stride, padding=2)

        self.conv11 = nn.Conv2d(planes, planes // 2, kernel_size=1, stride=stride, padding=0)
        self.conv13 = nn.Conv2d(planes, planes // 2, kernel_size=3, stride=stride, padding=1)
        self.conv15 = nn.Conv2d(planes, planes // 2, kernel_size=5, stride=stride, padding=2)

        self.branch1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(planes // 2, planes // 2, kernel_size=1, stride=stride, padding=0)
        )

        self.branch3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(planes // 2, planes // 2, kernel_size=3, stride=stride, padding=1)
        )

        self.branch5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(planes // 2, planes // 2, kernel_size=5, stride=stride, padding=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(planes // 2, planes // 4, kernel_size=1, stride=stride, padding=0),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(planes // 4, 1, kernel_size=1, stride=stride, padding=0),
            nn.Sigmoid()
        )

    def forward(self, feat_blurry, feat_events):  # target_view, ref_view
        w01 = self.conv01(feat_blurry)
        w03 = self.conv03(feat_blurry)
        w05 = self.conv05(feat_blurry)

        w11 = self.conv11(feat_events)
        w13 = self.conv13(feat_events)
        w15 = self.conv15(feat_events)

        w1 = w01 + w11
        w3 = w03 + w13
        w5 = w05 + w15

        w = self.branch1(w1) + self.branch1(w3) + self.branch1(w5)
        w = self.conv2(w)

        weight = self.conv3(w)
        feat_fuse = weight * feat_blurry + (1 - weight) * feat_events
        return feat_fuse


class StNet(nn.Module):
    def __init__(self, baseline):
        super(StNet, self).__init__()
        self.S = 4
        self.G0 = 16
        self.bl = baseline

        """ -------------------- Shallow Feature Extraction -------------------- """
        self.SFENet_b_1 = nn.ModuleList()
        for i in range(self.S):
            self.SFENet_b_1.append(
                nn.Conv2d(3*pow(4,i), pow(2,i)*self.G0, 5, padding=2, stride=1)  # 3*1,3*4,3*16 -> G0,2*G0,4*G0
            )
        self.SFENet_b_2 = nn.ModuleList()
        self.SFENet_b_2.append(
            nn.Conv2d(self.G0, self.G0, 3, padding=(3 - 1) // 2, stride=1)
        )
        self.down_b_01 = Conv2(self.G0, 2*self.G0)
        self.SFENet_b_2.append(
            nn.Conv2d(4*self.G0, 2*self.G0, 3, padding=(3 - 1) // 2, stride=1)
        )
        self.down_b_12 = Conv2(2*self.G0, 4*self.G0)
        self.SFENet_b_2.append(
            nn.Conv2d(8*self.G0, 4*self.G0, 3, padding=(3 - 1) // 2, stride=1)
        )
        self.down_b_23 = Conv2(4*self.G0, 8*self.G0)
        self.SFENet_b_2.append(
            nn.Conv2d(16*self.G0, 8*self.G0, 3, padding=(3 - 1) // 2, stride=1)
        )

        self.SFENet_e_1 = nn.ModuleList()
        for i in range(self.S):
            self.SFENet_e_1.append(
                nn.Conv2d(6*6*pow(4,i), pow(2,i)*self.G0, 5, padding=2, stride=1)  # 36*1,36*4,36*16 -> G0,2*G0,4*G0
            )
        self.SFENet_e_2 = nn.ModuleList()
        self.SFENet_e_2.append(
            nn.Conv2d(self.G0, self.G0, 3, padding=(3 - 1) // 2, stride=1)
        )
        self.down_e_01 = Conv2(self.G0, 2*self.G0)
        self.SFENet_e_2.append(
            nn.Conv2d(4*self.G0, 2*self.G0, 3, padding=(3 - 1) // 2, stride=1)
        )
        self.down_e_12 = Conv2(2*self.G0, 4*self.G0)
        self.SFENet_e_2.append(
            nn.Conv2d(8*self.G0, 4*self.G0, 3, padding=(3 - 1) // 2, stride=1)
        )
        self.down_e_23 = Conv2(4*self.G0, 8*self.G0)
        self.SFENet_e_2.append(
            nn.Conv2d(16*self.G0, 8*self.G0, 3, padding=(3 - 1) // 2, stride=1)
        )

        """ -------------------- Disparity Estimation -------------------- """
        self.PA = nn.ModuleList()
        for i in range(self.S):
            self.PA.append(
                PyramidAttention(pow(2,i)*self.G0),
            )
        self.DE = nn.ModuleList()
        for i in range(self.S):
            self.DE.append(
                nn.Sequential(
                    nn.Conv2d(pow(2,i)*self.G0, self.G0, 3, padding=(3 - 1) // 2, stride=1),
                    nn.Conv2d(self.G0, 1, 3, padding=(3 - 1) // 2, stride=1),
                )
            )
        self.sigmoid = nn.Sigmoid()

        """ -------------------- Upsampling -------------------- """
        self.up_e_32 = Deconv2(8*self.G0, 4*self.G0)
        self.up_e_21 = Deconv2(4*self.G0, 2*self.G0)
        self.up_e_10 = Deconv2(2*self.G0, self.G0)

        self.up_b_32 = Deconv2(8*self.G0, 4*self.G0)
        self.up_b_21 = Deconv2(4*self.G0, 2*self.G0)
        self.up_b_10 = Deconv2(2*self.G0, self.G0)


    def forward(self, blurry_0, events_0):
        """ -------------------- Shallow Feature Extraction -------------------- """
        # events path
        events_1 = pixel_reshuffle(events_0, 2)
        events_2 = pixel_reshuffle(events_0, 4)
        events_3 = pixel_reshuffle(events_0, 8)

        feat_e_0 = self.SFENet_e_1[0](events_0)
        feat_e_1 = self.SFENet_e_1[1](events_1)
        feat_e_2 = self.SFENet_e_1[2](events_2)
        feat_e_3 = self.SFENet_e_1[3](events_3)

        feat_e_0 = self.SFENet_e_2[0](feat_e_0)  # [bs, G0, H, W]
        feat_e_1 = self.SFENet_e_2[1](torch.cat((feat_e_1, self.down_e_01(feat_e_0)), 1))  # [bs, 2G0, H/2, W/2]
        feat_e_2 = self.SFENet_e_2[2](torch.cat((feat_e_2, self.down_e_12(feat_e_1)), 1))  # [bs, 4G0, H/4, W/4]
        feat_e_3 = self.SFENet_e_2[3](torch.cat((feat_e_3, self.down_e_23(feat_e_2)), 1))  # [bs, 8G0, H/8, W/8]

        # blurry path
        blurry_1 = pixel_reshuffle(blurry_0, 2)
        blurry_2 = pixel_reshuffle(blurry_0, 4)
        blurry_3 = pixel_reshuffle(blurry_0, 8)

        feat_b_0 = self.SFENet_b_1[0](blurry_0)
        feat_b_1 = self.SFENet_b_1[1](blurry_1)
        feat_b_2 = self.SFENet_b_1[2](blurry_2)
        feat_b_3 = self.SFENet_b_1[3](blurry_3)

        feat_b_0 = self.SFENet_b_2[0](feat_b_0)
        feat_b_1 = self.SFENet_b_2[1](torch.cat((feat_b_1, self.down_b_01(feat_b_0)), 1))
        feat_b_2 = self.SFENet_b_2[2](torch.cat((feat_b_2, self.down_b_12(feat_b_1)), 1))
        feat_b_3 = self.SFENet_b_2[3](torch.cat((feat_b_3, self.down_b_23(feat_b_2)), 1))

        """ -------------------- Disparity Estimation -------------------- """
        disps_3 = self.PA[3](feat_b_3, feat_e_3)
        disps_3 = self.sigmoid(self.DE[3](disps_3)) * (self.bl/8.)

        disps_3_up = F.interpolate(disps_3, scale_factor=2, mode='bilinear', recompute_scale_factor=False) * 2.
        mid_e_2 = self.up_e_32(feat_e_3) + feat_e_2
        mid_b_2 = self.up_b_32(feat_b_3) + feat_b_2
        warped_e_2, _ = back_warp_disp(mid_e_2, disps_3_up)
        disps_2 = self.PA[2](mid_b_2, warped_e_2)
        disps_2 = self.DE[2](disps_2) + disps_3_up
        disps_2 = disps_2.clamp(min=0, max=(self.bl/4.))

        disps_2_up = F.interpolate(disps_2, scale_factor=2, mode='bilinear', recompute_scale_factor=False) * 2.
        mid_e_1 = self.up_e_21(mid_e_2) + feat_e_1
        mid_b_1 = self.up_b_21(mid_b_2) + feat_b_1
        warped_e_1, _ = back_warp_disp(mid_e_1, disps_2_up)
        disps_1 = self.PA[1](mid_b_1, warped_e_1)
        disps_1 = self.DE[1](disps_1) + disps_2_up
        disps_1 = disps_1.clamp(min=0, max=(self.bl/2.))

        disps_1_up = F.interpolate(disps_1, scale_factor=2, mode='bilinear', recompute_scale_factor=False) * 2.
        mid_e_0 = self.up_e_10(mid_e_1) + feat_e_0
        mid_b_0 = self.up_b_10(mid_b_1) + feat_b_0
        warped_e_0, _ = back_warp_disp(mid_e_0, disps_1_up)
        disps_0 = self.PA[0](mid_b_0, warped_e_0)
        disps_0 = self.DE[0](disps_0) + disps_1_up
        disps_0 = disps_0.clamp(min=0, max=self.bl)

        return disps_0