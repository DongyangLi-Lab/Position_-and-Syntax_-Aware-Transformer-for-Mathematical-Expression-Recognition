import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm for feature maps: [B, C, H, W]."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)   # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)   # [B, C, H, W]
        return x


def _make_reference_grid(num_ref_points: int) -> torch.Tensor:
    """
    Create approximately uniform learned reference point initialization
    in normalized grid coordinates [-1, 1].

    Return:
        [R, 2], each point is (x, y).
    """
    side = int(math.ceil(math.sqrt(num_ref_points)))
    ys = torch.linspace(-0.75, 0.75, steps=side)
    xs = torch.linspace(-0.75, 0.75, steps=side)

    points = []
    for y in ys:
        for x in xs:
            points.append([x.item(), y.item()])
            if len(points) >= num_ref_points:
                return torch.tensor(points, dtype=torch.float32)

    return torch.tensor(points[:num_ref_points], dtype=torch.float32)


class SparseDeformableRefAttentionLite(nn.Module):
    """
    Lightweight reference-point deformable sparse attention.

    Paper-compatible components:
    - learned reference points
    - sparse sampling with grid_sample
    - deformable offsets predicted from global context
    - attention-weighted aggregation of sampled features
    - low-risk residual scaling

    Input/output:
        [B, C, H, W] -> [B, C, H, W]
    """

    def __init__(
        self,
        dim: int,
        num_ref_points: int = 8,
        num_samples: int = 4,
        offset_scale: float = 0.25,
        residual_scale_init: float = 1e-4,
        gaussian_sigma: float = 0.35,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_ref_points = num_ref_points
        self.num_samples = num_samples
        self.offset_scale = offset_scale
        self.gaussian_sigma = gaussian_sigma

        ref = _make_reference_grid(num_ref_points)
        self.reference_points = nn.Parameter(ref)  # [R, 2], normalized [-1, 1]

        hidden = max(32, dim // 4)

        self.context_proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
        )
        self.offset_head = nn.Linear(hidden, num_ref_points * num_samples * 2)
        self.attn_head = nn.Linear(hidden, num_ref_points * num_samples)

        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.Dropout(dropout),
        )

        self.residual_scale = nn.Parameter(torch.ones(1) * residual_scale_init)

        self._init_low_risk()

    def _init_low_risk(self):
        # Start with zero offsets and uniform attention.
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

        nn.init.zeros_(self.attn_head.weight)
        nn.init.zeros_(self.attn_head.bias)

        # Initial branch output is zero, making this module near identity.
        conv = self.out_proj[0]
        nn.init.zeros_(conv.weight)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    def _make_spatial_ref_maps(self, h: int, w: int, device, dtype) -> torch.Tensor:
        """
        Convert sparse reference contexts into a dense spatial map using
        Gaussian maps centered at learned reference points.

        Return:
            [R, H, W], normalized over R at each spatial location.
        """
        y = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)

        yy = y.view(1, h, 1)
        xx = x.view(1, 1, w)

        ref = self.reference_points.to(device=device, dtype=dtype).clamp(-1.0, 1.0)
        rx = ref[:, 0].view(-1, 1, 1)
        ry = ref[:, 1].view(-1, 1, 1)

        dist2 = (xx - rx) ** 2 + (yy - ry) ** 2
        maps = torch.exp(-dist2 / (2 * self.gaussian_sigma ** 2))
        maps = maps / maps.sum(dim=0, keepdim=True).clamp_min(1e-6)
        return maps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        device = x.device
        dtype = x.dtype

        pooled = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)  # [B, C]
        ctx = self.context_proj(pooled)

        offsets = self.offset_head(ctx).view(
            b, self.num_ref_points, self.num_samples, 2
        )
        offsets = torch.tanh(offsets) * self.offset_scale

        attn_logits = self.attn_head(ctx).view(
            b, self.num_ref_points, self.num_samples
        )
        attn = torch.softmax(attn_logits, dim=-1)

        ref = self.reference_points.to(device=device, dtype=dtype).clamp(-1.0, 1.0)
        sample_grid = ref.view(1, self.num_ref_points, 1, 2) + offsets
        sample_grid = sample_grid.clamp(-1.0, 1.0)

        # [B, R*K, 1, 2]
        grid = sample_grid.view(b, self.num_ref_points * self.num_samples, 1, 2)

        sampled = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [B, C, R*K, 1]

        sampled = sampled.view(b, c, self.num_ref_points, self.num_samples)
        context = (sampled * attn.unsqueeze(1)).sum(dim=-1)  # [B, C, R]

        ref_maps = self._make_spatial_ref_maps(h, w, device, dtype)  # [R, H, W]
        dense = torch.einsum("bcr,rhw->bchw", context, ref_maps)

        return self.residual_scale * self.out_proj(dense)


class BCFFN(nn.Module):
    """
    Blend-Convolution Feed-Forward Network.

    Uses:
    - 1x1 pointwise projection
    - 7x7 depthwise convolution
    - GELU
    - 1x1 pointwise projection
    """

    def __init__(
        self,
        dim: int,
        hidden_ratio: float = 0.25,
        kernel_size: int = 7,
        dropout: float = 0.0,
        residual_scale_init: float = 1e-4,
    ):
        super().__init__()

        hidden_dim = max(16, int(dim * hidden_ratio))
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
                groups=hidden_dim,
                bias=True,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True),
            nn.Dropout(dropout),
        )

        self.residual_scale = nn.Parameter(torch.ones(1) * residual_scale_init)
        self._init_low_risk()

    def _init_low_risk(self):
        last_conv = None
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is not None:
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual_scale * self.net(x)


class PaperLiteLSTBlock(nn.Module):
    """
    Minimal paper-compatible LST block:

        x = x + SparseDeformableRefAttentionLite(LN(x))
        x = x + BCFFN(LN(x))

    Low-risk initialization makes this block near identity at the beginning.
    """

    def __init__(
        self,
        dim: int,
        num_ref_points: int = 8,
        num_samples: int = 4,
        offset_scale: float = 0.25,
        ffn_hidden_ratio: float = 0.25,
        ffn_kernel_size: int = 7,
        dropout: float = 0.0,
        residual_scale_init: float = 1e-4,
        gaussian_sigma: float = 0.35,
    ):
        super().__init__()

        self.norm1 = LayerNorm2d(dim)
        self.attn = SparseDeformableRefAttentionLite(
            dim=dim,
            num_ref_points=num_ref_points,
            num_samples=num_samples,
            offset_scale=offset_scale,
            residual_scale_init=residual_scale_init,
            gaussian_sigma=gaussian_sigma,
            dropout=dropout,
        )

        self.norm2 = LayerNorm2d(dim)
        self.ffn = BCFFN(
            dim=dim,
            hidden_ratio=ffn_hidden_ratio,
            kernel_size=ffn_kernel_size,
            dropout=dropout,
            residual_scale_init=residual_scale_init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PaperLiteLST(nn.Module):
    """Stack of PaperLiteLSTBlock."""
    def __init__(
        self,
        dim: int,
        depth: int = 1,
        num_ref_points: int = 8,
        num_samples: int = 4,
        offset_scale: float = 0.25,
        ffn_hidden_ratio: float = 0.25,
        ffn_kernel_size: int = 7,
        dropout: float = 0.0,
        residual_scale_init: float = 1e-4,
        gaussian_sigma: float = 0.35,
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[
            PaperLiteLSTBlock(
                dim=dim,
                num_ref_points=num_ref_points,
                num_samples=num_samples,
                offset_scale=offset_scale,
                ffn_hidden_ratio=ffn_hidden_ratio,
                ffn_kernel_size=ffn_kernel_size,
                dropout=dropout,
                residual_scale_init=residual_scale_init,
                gaussian_sigma=gaussian_sigma,
            )
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
