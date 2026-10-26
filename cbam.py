import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.mlp[2].bias, 5.0)

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.mlp(self.avg_pool(x))
        # Use flatten+max instead of AdaptiveMaxPool2d to avoid non-deterministic
        # backward NaN that occurs with AdaptiveMaxPool2d on early backbone features.
        max_out = self.mlp(x.flatten(2).max(dim=2)[0].view(b, c, 1, 1))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=True)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv.bias, 5.0)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, c1, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(c1, reduction)
        self.sa = SpatialAttention(7)

    def forward(self, x):
        return self.sa(self.ca(x))
