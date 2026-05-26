import torch
import math
from vit_pytorch.cvt import group_by_key_prefix_and_remove_prefix, LayerNorm, Transformer
from torch import nn
from Lst import PaperLiteLST


class CvT(nn.Module):
    def __init__(
        self,
        *,
        s1_emb_dim = 64,
        s1_emb_kernel = 7,
        s1_emb_stride = 4,
        s1_proj_kernel = 3,
        s1_kv_proj_stride = 2,
        s1_heads = 1,
        s1_depth = 1,
        s1_mlp_mult = 4,
        s2_emb_dim = 192,
        s2_emb_kernel = 3,
        s2_emb_stride = 2,
        s2_proj_kernel = 3,
        s2_kv_proj_stride = 2,
        s2_heads = 3,
        s2_depth = 2,
        s2_mlp_mult = 4,
        s3_emb_dim = 384,
        s3_emb_kernel = 3,
        s3_emb_stride = 2,
        s3_proj_kernel = 3,
        s3_kv_proj_stride = 2,
        s3_heads = 6,
        s3_depth = 10,
        s3_mlp_mult = 4,
        dropout = 0.,

        # Paper-compatible lightweight LST options.
        use_lst = False,
        lst_depths = (0, 1, 1),
        lst_num_ref_points = 8,
        lst_num_samples = 4,
        lst_offset_scale = 0.25,
        lst_ffn_hidden_ratio = 0.25,
        lst_ffn_kernel_size = 7,
        lst_dropout = 0.,
        lst_residual_scale_init = 1e-4,
        lst_gaussian_sigma = 0.35,
    ):
        super().__init__()
        kwargs = dict(locals())

        # Remove LST-only keys before vit_pytorch's group_by_key_prefix_and_remove_prefix
        # processes stage-specific CvT arguments.
        for key in [
            "use_lst",
            "lst_depths",
            "lst_num_ref_points",
            "lst_num_samples",
            "lst_offset_scale",
            "lst_ffn_hidden_ratio",
            "lst_ffn_kernel_size",
            "lst_dropout",
            "lst_residual_scale_init",
            "lst_gaussian_sigma",
            "__class__",
        ]:
            kwargs.pop(key, None)

        dim = 1
        layers = []
        stage_dims = []

        for prefix in ('s1', 's2', 's3'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)

            layers.append(nn.Sequential(
                nn.Conv2d(
                    dim,
                    config['emb_dim'],
                    kernel_size = config['emb_kernel'],
                    padding = (config['emb_kernel'] // 2),
                    stride = config['emb_stride']
                ),
                LayerNorm(config['emb_dim']),
                Transformer(
                    dim = config['emb_dim'],
                    proj_kernel = config['proj_kernel'],
                    kv_proj_stride = config['kv_proj_stride'],
                    depth = config['depth'],
                    heads = config['heads'],
                    mlp_mult = config['mlp_mult'],
                    dropout = dropout
                )
            ))

            dim = config['emb_dim']
            stage_dims.append(dim)

        # Use ModuleList instead of one big Sequential so LST can be inserted
        # after selected CvT stages.
        self.layers = nn.ModuleList(layers)

        if isinstance(lst_depths, int):
            lst_depths = (0, 0, lst_depths)
        elif isinstance(lst_depths, list):
            lst_depths = tuple(lst_depths)

        if len(lst_depths) != len(self.layers):
            raise ValueError(
                f"lst_depths should have {len(self.layers)} values, got {lst_depths}"
            )

        self.use_lst = use_lst
        self.lst_stages = nn.ModuleList()

        for stage_idx, stage_dim in enumerate(stage_dims):
            depth = int(lst_depths[stage_idx])
            if self.use_lst and depth > 0:
                self.lst_stages.append(
                    PaperLiteLST(
                        dim=stage_dim,
                        depth=depth,
                        num_ref_points=lst_num_ref_points,
                        num_samples=lst_num_samples,
                        offset_scale=lst_offset_scale,
                        ffn_hidden_ratio=lst_ffn_hidden_ratio,
                        ffn_kernel_size=lst_ffn_kernel_size,
                        dropout=lst_dropout,
                        residual_scale_init=lst_residual_scale_init,
                        gaussian_sigma=lst_gaussian_sigma,
                    )
                )
            else:
                self.lst_stages.append(nn.Identity())

    @staticmethod
    def positionalencoding2d(d_model, height, width):
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    
        return pe
    
    def forward(self, x):
        latents = x

        for stage, lst in zip(self.layers, self.lst_stages):
            latents = stage(latents)
            latents = lst(latents)

        positional_encoding = self.positionalencoding2d(
            latents.size(1),
            latents.size(2),
            latents.size(3)
        )
        positional_encoding = positional_encoding.unsqueeze(0)
        positional_encoding = positional_encoding.to(latents.device)

        positional_encoding = latents + positional_encoding
        positional_encoding = positional_encoding.view(
            positional_encoding.size(0),
            positional_encoding.size(1),
            -1
        )
        positional_encoding = positional_encoding.permute(2, 0, 1)

        return positional_encoding
