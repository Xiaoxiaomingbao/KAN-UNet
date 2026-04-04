import torch.nn as nn
from net import UpBlock, DownBlock
from net2 import DownBlock as KANDownBlock


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, n_filters=32):
        super(UNet, self).__init__()

        # Encoder
        self.down1 = DownBlock(n_channels, n_filters)
        self.down2 = DownBlock(n_filters, n_filters * 2)
        self.down3 = DownBlock(n_filters * 2, n_filters * 4)
        self.down4 = DownBlock(n_filters * 4, n_filters * 8)

        # Bottleneck
        self.bottleneck = KANDownBlock(
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

    def forward(self, x, update_grid=False):
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x4, skip4 = self.down4(x3)

        x5, _ = self.bottleneck(x4, update_grid=update_grid)

        x = self.up1(x5, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        x = self.outc(x)

        return self.sigmoid(x)

    def regularization_loss(self):
        reg_loss = 0
        for module in self.modules():
            if hasattr(module, "regularization_loss") and module != self:
                reg_loss += module.regularization_loss()
        return reg_loss
