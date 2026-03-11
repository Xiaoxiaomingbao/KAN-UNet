import torch
import torch.nn as nn
from efficient_kan.conv import KANConv2d


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0., max_pooling=True):
        super(DownBlock, self).__init__()
        self.conv = KANConv2d(in_channels, out_channels, 3)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.maxpool = nn.MaxPool2d(2) if max_pooling else None
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else None

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        if self.dropout:
            x = self.dropout(x)
        skip = x
        if self.maxpool:
            x = self.maxpool(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = KANConv2d(out_channels * 2, out_channels, 3)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.norm(self.conv(x)))
        return x


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, n_filters=32):
        super(UNet, self).__init__()

        # Encoder (更浅)
        self.down1 = DownBlock(n_channels, n_filters)
        self.down2 = DownBlock(n_filters, n_filters * 2)
        self.down3 = DownBlock(n_filters * 2, n_filters * 4)
        self.down4 = DownBlock(n_filters * 4, n_filters * 8)

        # Bottleneck
        self.bottleneck = DownBlock(
            n_filters * 8,
            n_filters * 16,
            dropout_prob=0.3,
            max_pooling=False,
        )

        # Decoder
        self.up1 = UpBlock(n_filters * 16, n_filters * 8)
        self.up2 = UpBlock(n_filters * 8, n_filters * 4)
        self.up3 = UpBlock(n_filters * 4, n_filters * 2)
        self.up4 = UpBlock(n_filters * 2, n_filters)

        # Output
        self.outc = nn.Conv2d(n_filters, n_classes, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x4, skip4 = self.down4(x3)

        x5, _ = self.bottleneck(x4)

        x = self.up1(x5, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        x = self.outc(x)

        return self.sigmoid(x)
