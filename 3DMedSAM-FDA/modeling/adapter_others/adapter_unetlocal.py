import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """(간단한) 3D Conv Block
    - 기본적으로 2.5D 느낌을 살리기 위해 (1,3,3) + pointwise(1,1,1) 구조 사용
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # depthwise in-plane conv
        self.depthwise = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=in_channels,
        )
        # pointwise mixing
        self.pointwise = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class LocalUNetPPPath3D(nn.Module):
    """
    Local Path with UNet++-style Skip Connections (간단 2-level 버전)

    노드 구조 (UNet++ notation):
        x_0_0 : high-res encoder feature
        x_1_0 : low-res encoder feature
        x_0_1 : decoder node = conv( [x_0_0, up(x_1_0)] )

    Input  : (B, C, D, H, W)  (C = mid_dim)
    Output : (B, C, D, H, W)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # encoder
        self.conv_0_0 = ConvBlock3D(channels, channels)
        self.conv_1_0 = ConvBlock3D(channels, channels)

        # decoder (UNet++ style: concat(x_0_0, up(x_1_0)) → conv)
        self.conv_0_1 = ConvBlock3D(channels * 2, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, D, H, W)
        """
        # encoder level 0
        x_0_0 = self.conv_0_0(x)  # (B, C, D, H, W)

        # encoder level 1 (downsample H,W)
        x_1_0_in = self.pool(x_0_0)         # (B, C, D, H/2, W/2)
        x_1_0 = self.conv_1_0(x_1_0_in)     # (B, C, D, H/2, W/2)

        # upsample back to level 0 resolution
        up_x_1_0 = F.interpolate(
            x_1_0,
            size=x_0_0.shape[2:],  # (D, H, W)
            mode="trilinear",
            align_corners=False,
        )

        # UNet++ style: concat encoder & upsampled deeper feature
        x_0_1_in = torch.cat([x_0_0, up_x_1_0], dim=1)  # (B, 2C, D, H, W)
        x_0_1 = self.conv_0_1(x_0_1_in)                 # (B, C, D, H, W)

        return x_0_1


class Adapter(nn.Module):
    """
    Dual-Path 3D Adapter with
    - Global 3D Context Path (저주파 / 형태 정보)
    - Local UNet++-style 2.5D Texture Path (고주파 / 텍스처 + 경계)
    - Gated Fusion (Global이 Local을 조절)

    Input / Output:
        features: (B, D, H, W, C)  # C = input_dim
    """

    def __init__(
        self,
        input_dim: int,
        mid_dim: int
    ):
        super().__init__()

        # 공통: token dim -> mid_dim
        self.linear1 = nn.Linear(input_dim, mid_dim)

        # (1) Global 3D Context Path
        self.global_conv = nn.Conv3d(
            in_channels=mid_dim,
            out_channels=mid_dim,
            kernel_size=3,
            padding=1,
            groups=mid_dim,  # depthwise 3D conv
        )

        # (2) Local Path: UNet++ style
        self.local_unetpp = LocalUNetPPPath3D(mid_dim)

        # (3) Fusion: Global → Gate → Local 조절
        self.fusion_gate = nn.Conv3d(
            in_channels=mid_dim,
            out_channels=mid_dim,
            kernel_size=1,
        )

        # 출력 proj
        self.linear2 = nn.Linear(mid_dim, input_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B, D, H, W, C)
        """
        # token 차원 축소 (공통 인코딩)
        x = self.linear1(features)          # (B, D, H, W, mid_dim)
        x = F.relu(x)

        # Conv3d용 형태로 변경
        x_3d = x.permute(0, 4, 1, 2, 3)     # (B, mid_dim, D, H, W)

        # ============================
        # (1) Global 3D Context Path
        # ============================
        global_feat = self.global_conv(x_3d)
        global_feat = F.relu(global_feat)   # (B, mid_dim, D, H, W)

        # ============================
        # (2) Local UNet++ Path
        # ============================
        local_feat = self.local_unetpp(x_3d)  # (B, mid_dim, D, H, W)

        # ============================
        # (3) Gated Fusion
        # ============================
        gate = torch.sigmoid(self.fusion_gate(global_feat))  # (B, mid_dim, D, H, W)
        fused_3d = global_feat + gate * local_feat           # (B, mid_dim, D, H, W)

        # 원래 형태로 복구: (B, D, H, W, mid_dim)
        fused = fused_3d.permute(0, 2, 3, 4, 1)

        # proj back + residual
        out = self.linear2(fused)
        out = F.relu(out)

        out = features + out
        return out

