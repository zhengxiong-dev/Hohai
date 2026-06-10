"""
Custom modules for UAV-YOLO and AFA-YOLO11.
EMA: Efficient Multi-Scale Attention (ICASSP 2023)
BiFPN_Concat: Weighted feature fusion from EfficientDet
SGFA: Small-object Guided Feature Alignment (soft alignment)
SOA_SAFM: Small-object Aware Spatial Adaptive Fusion Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SGFA(nn.Module):
    """
    Small-object Guided Feature Alignment (soft alignment version).

    A lightweight cross-scale fusion module for YOLO neck that uses spatial gating
    to align deep semantic features with shallow spatial features, guided by
    small-object response maps.

    Designed for AFA-YOLO11 to improve small object detection in UAV imagery.

    Args:
        dimension (int): Concatenation dimension. Default: 1 (channel dim).
        ratio (float): Channel reduction ratio for projection. Default: 1.0.

    Input:
        x: List[Tensor] of length 2
           x[0] = F_high: deep semantic feature (will be upsampled if needed)
           x[1] = F_low:  shallow spatial feature (target resolution)

    Output:
        Tensor: Concatenation of [F_low_projected, F_high_soft_aligned]

    Forward Flow:
        1. Upsample F_high to match F_low spatial size
        2. Project both features via 1x1 conv (channel reduction)
        3. Generate small-object response map M_small from F_low
        4. Predict alignment gate A from [F_high, F_low, M_small]
        5. Apply gate: F_high_soft = F_high * A (soft alignment)
        6. Return concat([F_low, F_high_soft])

    Note:
        This version uses lazy build (builds layers on first forward).
        Output channels = c_low * ratio + c_high * ratio
    """

    def __init__(self, dimension=1, ratio=1.0):
        super().__init__()
        self.d = dimension
        self.ratio = ratio
        self._built = False

    def _build(self, c_high, c_low, device):
        """Build layers on first forward pass (lazy initialization)."""
        c_mid_high = max(1, int(c_high * self.ratio))
        c_mid_low = max(1, int(c_low * self.ratio))

        # Project high-level features
        self.high_proj = nn.Conv2d(c_high, c_mid_high, 1, 1, 0).to(device)

        # Project low-level features
        self.low_proj = nn.Conv2d(c_low, c_mid_low, 1, 1, 0).to(device)

        # Small-object response generator (from low-level features)
        self.small_gate = nn.Sequential(
            nn.Conv2d(c_mid_low, 1, 3, 1, 1),
            nn.Sigmoid(),
        ).to(device)

        # Alignment gate predictor (guided by small-object response)
        # Input: [fh, fl, m_small] -> Output: spatial gate for fh
        self.align_gate = nn.Sequential(
            nn.Conv2d(c_mid_high + c_mid_low + 1, c_mid_high, 3, 1, 1),
            nn.BatchNorm2d(c_mid_high),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid_high, c_mid_high, 1, 1, 0),
            nn.Sigmoid(),
        ).to(device)

        self._built = True

    def forward(self, x):
        """
        Args:
            x: List[Tensor, Tensor]
               x[0] = F_high (B, C_high, H_high, W_high)
               x[1] = F_low  (B, C_low, H_low, W_low)

        Returns:
            Tensor: (B, C_mid_low + C_mid_high, H_low, W_low)
        """
        f_high, f_low = x

        # Ensure spatial size match (upsample F_high if needed)
        if f_high.shape[-2:] != f_low.shape[-2:]:
            f_high = F.interpolate(
                f_high, size=f_low.shape[-2:], mode="nearest"
            )

        # Lazy build on first forward
        if not self._built:
            self._build(f_high.shape[1], f_low.shape[1], f_high.device)

        # Project features
        fh = self.high_proj(f_high)  # (B, C_mid_high, H, W)
        fl = self.low_proj(f_low)    # (B, C_mid_low, H, W)

        # Generate small-object response map from shallow features
        m_small = self.small_gate(fl)  # (B, 1, H, W)

        # Predict alignment gate guided by [fh, fl, m_small]
        gate = self.align_gate(torch.cat([fh, fl, m_small], dim=1))  # (B, C_mid_high, H, W)

        # Soft alignment: suppress background, enhance small-object regions
        fh_soft = fh * gate

        # Concat aligned features
        return torch.cat([fl, fh_soft], self.d)


class SOA_SAFM(nn.Module):
    """
    Small-object Aware Spatial Adaptive Fusion Module.

    A spatial-adaptive multi-scale fusion module for YOLO detection heads that
    dynamically adjusts scale weights at each spatial location, guided by
    small-object response maps.

    Designed for AFA-YOLO11 to enhance P2/P3 detection layers for small object
    detection in UAV imagery.

    Args:
        target_index (int): Index of the target scale in input list. Default: 0.
                           The output will have the same spatial size as this input.
        out_channels (int): Output channels. If None, uses target input channels.
        ratio (float): Channel reduction ratio for intermediate features. Default: 1.0.

    Input:
        x: List[Tensor] of N multi-scale features
           Example for P2 output: [P2, P3, P4] where len(x) = 3
           Example for P3 output: [P2, P3, P4, P5] where len(x) = 4

    Output:
        Tensor: Fused feature at target scale
                Shape: (B, out_channels, H_target, W_target)

    Forward Flow:
        1. Resize all inputs to target_scale spatial size
        2. Project all inputs to c_mid channels via 1x1 conv
        3. Generate small-object response M_small from target feature
        4. Predict spatial scale weights W from [F1, F2, ..., FN, M_small]
        5. Apply softmax over scale dimension: sum(Wi) = 1 at each pixel
        6. Weighted sum: Out = sum(Wi * Fi)
        7. Project to output channels

    Key Difference from ASFF:
        - ASFF: learns weights from multi-scale features only
        - SOA-SAFM: explicitly guides weights with small-object response prior
        - Focus: small-object regions favor P2/P3, background suppressed

    Note:
        Uses lazy build (constructs layers on first forward).
        Recommended to apply only to P2/P3 outputs in first version.
    """

    def __init__(self, target_index=0, out_channels=None, ratio=1.0):
        super().__init__()
        self.target_index = target_index
        self.out_channels = out_channels
        self.ratio = ratio
        self._built = False

    def _build(self, channels, device):
        """Build layers on first forward pass (lazy initialization)."""
        # Determine output channels
        if self.out_channels is None:
            c_out = channels[self.target_index]
        else:
            c_out = self.out_channels

        c_mid = max(1, int(c_out * self.ratio))
        self.c_out = c_out
        self.n = len(channels)

        # Project all inputs to same channel dimension
        self.proj = nn.ModuleList([
            nn.Conv2d(c, c_mid, 1, 1, 0) for c in channels
        ]).to(device)

        # Small-object response generator (from target scale feature)
        self.small_gate = nn.Sequential(
            nn.Conv2d(c_mid, 1, 3, 1, 1),
            nn.Sigmoid(),
        ).to(device)

        # Spatial scale weight predictor
        # Input: [F1, F2, ..., FN, M_small]
        # Output: N weight maps (one per scale)
        self.weight_head = nn.Sequential(
            nn.Conv2d(c_mid * self.n + 1, c_mid, 3, 1, 1),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, self.n, 1, 1, 0),  # Output N channels for N scales
        ).to(device)

        # Final projection to output channels
        self.out_conv = nn.Conv2d(c_mid, c_out, 1, 1, 0).to(device)

        self._built = True

    def forward(self, x):
        """
        Args:
            x: List[Tensor] of N multi-scale features
               x[i] shape: (B, C_i, H_i, W_i)

        Returns:
            Tensor: (B, out_channels, H_target, W_target)
        """
        target = x[self.target_index]
        target_size = target.shape[-2:]

        # Lazy build on first forward
        if not self._built:
            channels = [feat.shape[1] for feat in x]
            self._build(channels, target.device)

        # Step 1: Resize all features to target size and project
        feats = []
        for i, feat in enumerate(x):
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode="nearest"
                )
            feats.append(self.proj[i](feat))

        b, _, h, w = feats[self.target_index].shape

        # Step 2: Generate small-object response from target scale
        m_small = self.small_gate(feats[self.target_index])  # (B, 1, H, W)

        # Step 3: Predict spatial scale weights
        weight_in = torch.cat(feats + [m_small], dim=1)  # (B, N*C_mid + 1, H, W)
        weight_logits = self.weight_head(weight_in)      # (B, N, H, W)

        # Reshape for softmax over scale dimension
        weights = weight_logits.view(b, self.n, 1, h, w)  # (B, N, 1, H, W)
        weights = torch.softmax(weights, dim=1)            # Sum over N = 1

        # Step 4: Weighted sum over scales
        out = 0
        for i in range(self.n):
            out = out + weights[:, i] * feats[i]  # (B, C_mid, H, W)

        # Step 5: Project to output channels
        return self.out_conv(out)  # (B, out_channels, H, W)
