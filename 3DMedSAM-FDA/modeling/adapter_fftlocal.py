import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalFFTPath3D(nn.Module):
    """
    Local Path with 3D FFT-based High-Frequency Enhancement

    Input  : (B, C, D, H, W)
    Output : (B, C, D, H, W)
    """
    def __init__(
        self,
        channels: int,
        high_freq_boost: float = 2.0,      # 고주파를 얼마나 더 강조할지
        min_radius_ratio: float = 0.25,    # 이 이상인 radius부터 고주파로 간주
    ):
        super().__init__()
        self.high_freq_boost = high_freq_boost
        self.min_radius_ratio = min_radius_ratio

        # FFT 후 공간 도메인으로 돌아온 다음, 가볍게 정제하는 conv
        self.post_conv = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
        )
        self.norm = nn.InstanceNorm3d(channels, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape

        # 1) 주파수 도메인으로 이동 (공간 3축에 대해 FFT)
        #    dtype은 복소수(complex)
        x_fft = torch.fft.fftn(x, dim=(-3, -2, -1))

        # 2) 주파수 그리드 생성 (normalized radius)
        device = x.device
        # dtype은 실수형으로
        base_dtype = x.real.dtype if torch.is_complex(x) else x.dtype

        freq_d = torch.fft.fftfreq(D, device=device, dtype=base_dtype).view(1, 1, D, 1, 1)
        freq_h = torch.fft.fftfreq(H, device=device, dtype=base_dtype).view(1, 1, 1, H, 1)
        freq_w = torch.fft.fftfreq(W, device=device, dtype=base_dtype).view(1, 1, 1, 1, W)

        # radius ~ 0 이면 저주파, radius ~ max 이면 고주파
        radius = torch.sqrt(freq_d ** 2 + freq_h ** 2 + freq_w ** 2)  # (1,1,D,H,W)
        radius_max = radius.max().clamp(min=1e-6)
        radius_norm = radius / radius_max  # [0, 1]

        # 3) 고주파 강조 mask 생성
        #    - min_radius_ratio 이하: 거의 1.0 유지 (저주파 유지)
        #    - 그 이상: 1.0 ~ 1.0 + high_freq_boost 까지 선형 증가
        #      weight = 1 + high_freq_boost * ((r - r0) / (1 - r0))
        r0 = self.min_radius_ratio
        high_region = torch.clamp(radius_norm - r0, min=0.0) / (1.0 - r0 + 1e-6)
        mask = 1.0 + self.high_freq_boost * high_region  # (1,1,D,H,W), real

        # 4) 주파수 도메인에서 고주파 강조
        x_fft_hf = x_fft * mask  # broadcasting; complex * real

        # 5) 다시 공간 도메인으로 (real part 사용)
        x_hf = torch.fft.ifftn(x_fft_hf, dim=(-3, -2, -1)).real  # (B, C, D, H, W)

        # 6) conv + norm + ReLU로 정제
        x_hf = self.post_conv(x_hf)
        x_hf = self.norm(x_hf)
        x_hf = self.act(x_hf)

        return x_hf


class Adapter(nn.Module):
    """
    Dual-Path 3D Adapter with FFT-based Local Path

    - Input / Output: features (B, D, H, W, C)
      C = input_dim

    Paths:
      (1) Global 3D Context Path: 3x3x3 depthwise conv (저주파 / 형태 정보)
      (2) Local FFT Path: 3D FFT로 고주파 강조 (텍스처 / 경계 정보)
      (3) Gated Fusion: global → gate → local_fft 조절
    """

    def __init__(
        self,
        input_dim: int,
        mid_dim: int
    ):
        super().__init__()

        # 공통: token 차원 -> mid_dim
        self.linear1 = nn.Linear(input_dim, mid_dim)

        # (1) Global 3D Context Path
        self.global_conv = nn.Conv3d(
            in_channels=mid_dim,
            out_channels=mid_dim,
            kernel_size=3,
            padding=1,
            groups=mid_dim,   # depthwise 3D conv
        )

        # (2) Local FFT Path
        self.local_fft = LocalFFTPath3D(
            channels=mid_dim,
            high_freq_boost=2.0,
            min_radius_ratio=0.25,
        )

        # (3) Fusion: Global → Gate
        self.fusion_gate = nn.Conv3d(
            in_channels=mid_dim,
            out_channels=mid_dim,
            kernel_size=1,
        )

        # 최종 proj: mid_dim -> input_dim
        self.linear2 = nn.Linear(mid_dim, input_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B, D, H, W, C)
        """
        # 0) token dim 축소
        x = self.linear1(features)          # (B, D, H, W, mid_dim)
        x = F.relu(x)

        # Conv3d형태로: (B, C, D, H, W)
        x_3d = x.permute(0, 4, 1, 2, 3)

        # ============================
        # (1) Global 3D Context Path
        # ============================
        global_feat = self.global_conv(x_3d)   # (B, mid_dim, D, H, W)
        global_feat = F.relu(global_feat)

        # ============================
        # (2) Local FFT Path
        # ============================
        local_feat = self.local_fft(x_3d)      # (B, mid_dim, D, H, W)

        # ============================
        # (3) Gated Fusion
        # ============================
        gate = torch.sigmoid(self.fusion_gate(global_feat))  # (B, mid_dim, D, H, W)
        fused_3d = global_feat + gate * local_feat           # (B, mid_dim, D, H, W)

        # 다시 (B, D, H, W, mid_dim)로
        fused = fused_3d.permute(0, 2, 3, 4, 1)

        # 최종 proj + residual
        out = self.linear2(fused)
        out = F.relu(out)

        out = features + out
        return out