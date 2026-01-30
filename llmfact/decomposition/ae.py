import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels=32, kernel_size=9, padding=4, groups=1, layer_num=3):
        super().__init__()
        self.conv = nn.ModuleList([nn.Conv3d(in_channels=in_channels, out_channels=in_channels,
                                             kernel_size=kernel_size, padding=padding, groups=groups) for _ in range(layer_num)])
        self.bn = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        x_in = x
        for layer in self.conv:
            x = F.gelu(layer(x))
        x = x + x_in
        x = self.bn(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, padding=4, kernel_size=9)
        self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, padding=1, kernel_size=3)
        # self.conv3 = nn.Conv3d(in_channels=in_channels, out_channels=1, padding=0, kernel_size=1)

    def forward(self, x): #(batch_size, in_channels, D, W, H)
        x = F.gelu(self.conv1(x))
        sa_weight = self.conv2(x)
        # sa_weight = self.conv3(sa_weight)
        sa = torch.sigmoid(sa_weight) # (batch_size, in_channels, D, W, H)
        return sa, sa_weight

class STCA(nn.Module):
    def __init__(self, time_step=284, out_map=32):
        super().__init__()
        self.conv_time = nn.Conv3d(in_channels=time_step, out_channels=out_map, kernel_size=9, padding=4)
        self.sa = SpatialAttention(in_channels=out_map)

    def forward(self, x):
        x = F.gelu(self.conv_time(x))
        sa, sa_weight = self.sa(x)
        # x = sa.expand_as(x).clone()
        return x, sa_weight

class ConvDown(nn.Module):
    def __init__(self, layer_num=3, kernel_size=3, padding=1, groups=1, in_channels=32):
        super(ConvDown, self).__init__()
        self.convBlock = ConvBlock(layer_num=layer_num, kernel_size=kernel_size,
                                   padding=padding, groups=groups, in_channels=in_channels)
        self.downsample = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        x = self.convBlock(x)
        x = self.downsample(x)
        return x

class ConvUp(nn.Module):
    def __init__(self, layer_num=3, kernel_size=3, padding=1, groups=1, in_channels=32):
        super(ConvUp, self).__init__()
        self.convBlock = ConvBlock(layer_num=layer_num, kernel_size=kernel_size,
                                   padding=padding, groups=groups, in_channels=in_channels)

    def forward(self, x):
        x = F.interpolate(x,  scale_factor=(1.0, 2.0, 2.0), mode='trilinear', align_corners=False)
        x = self.convBlock(x)
        return x

class Encoder(nn.Module):
    def __init__(self, out_map=32):
        super().__init__()
        self.encode_list = nn.ModuleList([ConvDown(in_channels=out_map) for _ in range(3)]) #(32,11008)

    def forward(self, x):
        for layer in self.encode_list:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, time_step=284, out_map=32):
        super(Decoder, self).__init__()
        self.decode_list = nn.ModuleList([ConvUp(in_channels=out_map) for _ in range(3)])
        self.conv_time = nn.Sequential(nn.Conv3d(in_channels=out_map, out_channels=time_step, kernel_size=9, padding=4),
                                       nn.GELU(),
                                       nn.Conv3d(in_channels=time_step, out_channels=time_step, kernel_size=1, padding=0))

    def forward(self, x):
        for layer in self.decode_list:
            x = layer(x)
        x = self.conv_time(x)
        return x

class LLaMASTCA(nn.Module):
    def __init__(self, time_step=20, out_map=32):
        super().__init__()
        self.stca = STCA(time_step=time_step, out_map=out_map) #(batch_size, time_step, D, W, H)
        self.encoder = Encoder(out_map=out_map)
        self.decoder = Decoder(time_step=time_step, out_map=out_map)

    def forward(self, x):
        x, sa_weight = self.stca(x)
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode

    def get_fbn(self, x):
        _, sa_weight = self.stca(x)
        return sa_weight