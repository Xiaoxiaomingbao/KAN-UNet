import torch.nn as nn
import torch.nn.functional as F

from efficient_kan import KANLinear


class KANConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        grid_size=5,
        spline_order=3,
        base_activation=nn.SiLU,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.in_channels = in_channels
        self.out_channels = out_channels

        # patch 向量长度
        patch_dim = in_channels * kernel_size[0] * kernel_size[1]

        # 核心：KANLinear
        self.kan_linear = KANLinear(
            patch_dim,
            out_channels,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation,
        )

    def forward(self, x, update_grid=False):

        B, C, H, W = x.shape

        # 1 unfold
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        # patches shape
        # (B, Cin*K*K, L)

        patches = patches.transpose(1, 2)

        # (B, L, patch_dim)

        B, L, D = patches.shape

        patches = patches.reshape(B * L, D)

        # 2 KANLinear
        if update_grid:
            self.kan_linear.update_grid(patches)
        out = self.kan_linear(patches)

        # (B*L, Cout)

        out = out.reshape(B, L, self.out_channels)

        out = out.transpose(1, 2)

        # 3 计算输出尺寸
        H_out = (
            (H + 2 * self.padding - self.dilation * (self.kernel_size[0] - 1) - 1)
            // self.stride
            + 1
        )

        W_out = (
            (W + 2 * self.padding - self.dilation * (self.kernel_size[1] - 1) - 1)
            // self.stride
            + 1
        )

        out = out.reshape(B, self.out_channels, H_out, W_out)

        return out

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.kan_linear.regularization_loss(
            regularize_activation,
            regularize_entropy
        )
