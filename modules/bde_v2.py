from cmath import tanh
import torch
import torch.nn as nn


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.1,inplace=True)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
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


class BDENet(nn.Module):
    def __init__(self, c=84):
        super(BDENet, self).__init__()
        # Encoders not sharing weights
        #self.down00 = Conv2(16, c)
        self.down01 = Conv2(c, 2 * c)
        #self.down02 = Conv2(2 * c, 4 * c)
        #self.down03 = Conv2(4 * c, 8 * c)
        #self.down04 = Conv2(8 * c, 16 * c)

        #self.down10 = Conv2(3, c)
        self.down11 = Conv2(c, 2 * c)
        #self.down12 = Conv2(2 * c, 4 * c)
        #self.down13 = Conv2(4 * c, 8 * c)
        #self.down14 = Conv2(8 * c, 16 * c)

        self.pa0 = PyramidAttention(c)
        self.pa1 = PyramidAttention(2 * c)
        #self.pa2 = PyramidAttention(4 * c)
        #self.pa3 = PyramidAttention(8 * c)

        #self.up0 = deconv(32 * c, 8 * c)
        #self.up1 = deconv(16 * c, 4 * c)
        #self.up2 = deconv(12 * c, 2 * c)
        self.up3 = deconv(6 * c, c)
        self.conv = nn.Sequential(
            nn.Conv2d(4 * c, c, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c, 12, 3, padding=(3 - 1) // 2, stride=1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_b, feat_e):
        e00 = feat_b  # [bs, c, H/2, W/2]
        e01 = self.down01(e00)  # [bs, 2c, H/4, W/4]
        #e02 = self.down02(e01)  # [bs, 4c, H/8, W/8]
        #e03 = self.down03(e02)  # [bs, 8c, H/16, W/16]
        #e04 = self.down04(e03)

        e10 = feat_e
        e11 = self.down11(e10)
        #e12 = self.down12(e11)
        #e13 = self.down13(e12)
        #e14 = self.down14(e13)

        w0 = self.pa0(e00, e10)  # [bs, c, H/2, W/2]
        w1 = self.pa1(e01, e11)  # [bs, 2c, H/4, W/4]
        #w2 = self.pa2(e02, e12)  # [bs, 4c, H/8, W/8]
        #w3 = self.pa3(e13, e03)

        #d0 = self.up0(torch.cat((e14, e04), 1))
        #d1 = self.up1(torch.cat((e13, e03), 1))  # [bs, 4c, H/8, W/8]
        #d2 = self.up2(torch.cat((e02, e12, w2), 1))  # [bs, 2c, H/4, W/4]
        d3 = self.up3(torch.cat((e01, e11, w1), 1))  # [bs, c, H/2, W/2]
        disps = self.sigmoid(self.conv(torch.cat((d3, e00, e10, w0), 1)))  # [bs, 16, H/2, W/2]  dsec: 5 for left; 65 for right

        return disps


class DispNet(nn.Module):
    def __init__(self):
        super(DispNet, self).__init__()
        self.bde = BDENet()

    def forward(self, feat_e, feat_b):
        disps = self.bde(feat_e, feat_b)

        return disps


if __name__ == "__main__":
    model = DispNet().to('cpu')

    bs = 2

    events = torch.rand(bs, 96, 64, 64).to('cpu')
    blurry = torch.rand(bs, 96, 64, 64).to('cpu')

    outputs = model(blurry, events)
    print(outputs.shape)
