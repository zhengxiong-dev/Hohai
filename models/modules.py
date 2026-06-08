"""
Custom modules for UAV-YOLO.
EMA: Efficient Multi-Scale Attention (ICASSP 2023)
BiFPN_Concat: Weighted feature fusion from EfficientDet
"""

import torch
import torch.nn as nn


class EMA(nn.Module):
    """
    Efficient Multi-Scale Attention Module.
    Paper: Efficient Multi-Scale Attention Module with Cross-Spatial Learning (ICASSP 2023)

    Args:
        c1 (int): Input channels (required by ultralytics yaml parser).
        c2 (int): Unused, kept for compatibility. Output channels = c1.
        groups (int): Number of groups for grouped operations.
    """

    def __init__(self, c1, c2=None, groups=32):
        super().__init__()
        # Ensure groups divides c1
        while c1 % groups != 0 and groups > 1:
            groups //= 2
        self.groups = groups
        gc = c1 // self.groups

        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(gc, gc)
        self.conv1x1 = nn.Conv2d(gc, gc, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(gc, gc, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        gc = c // self.groups

        # Reshape into groups: (B*G, C//G, H, W)
        group_x = x.reshape(b * self.groups, gc, h, w)

        # Spatial pooling
        x_h = self.pool_h(group_x)  # (B*G, gc, H, 1)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # (B*G, gc, W, 1)

        # Cross-spatial: concat along H dimension, then split
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # (B*G, gc, H+W, 1)
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # Branch 1: spatial attention
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        # Branch 2: local context
        x2 = self.conv3x3(group_x)

        # Cross-attention weights
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, gc, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, gc, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(
            b * self.groups, 1, h, w
        )
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class BiFPN_Concat(nn.Module):
    """
    Weighted Concatenation for BiFPN-style feature fusion.
    Uses fast normalized fusion from EfficientDet paper.

    Args:
        dimension (int): Concatenation dimension. Default: 1.
    """

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.epsilon = 1e-4
        # Initialize weights for 2 inputs (most common case)
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self._num_inputs = 2

    def forward(self, x):
        n = len(x)
        # Recreate weights if number of inputs changed
        if n != self._num_inputs:
            self.w = nn.Parameter(
                torch.ones(n, dtype=torch.float32, device=x[0].device)
            )
            self._num_inputs = n

        # Fast normalized fusion (non-inplace relu)
        w = torch.relu(self.w.clone())
        w = w / (w.sum() + self.epsilon)

        return torch.cat([w[i] * x[i] for i in range(n)], self.d)
