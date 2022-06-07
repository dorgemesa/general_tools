import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderBlock, self).__init__()
        self.encodconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.encodconv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.decodblock = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2))
       
        x = torch.cat([x2, x1], dim=1)
        return self.decodblock(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()

        self.first = DoubleConv(in_channels, 64)
        self.encod1 = EncoderBlock(64, 128)
        self.encod2 = EncoderBlock(128, 256)
        self.encod3 = EncoderBlock(256, 256)
        self.decod3 = DecoderBlock(512, 128)
        self.decod2 = DecoderBlock(256, 64)
        self.decod1 = DecoderBlock(128, 64)
        self.last = OutConv(64, num_classes)
    
    def forward(self, input):
        x1 = self.first(input)
        x2 = self.encod1(x1)
        x3 = self.encod2(x2)
        x4 = self.encod3(x3)
        x5 = self.decod3(x4, x3)
        x6 = self.decod2(x5, x2)
        x6 = self.decod1(x6, x1)
        output = self.last(x6)

        return output
    