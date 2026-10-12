"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core import register
from src.nn.backbone.NewCWAM import CrossWinBasedAttention


__all__ = ['DFINE', ]


def _as_list(value, length):
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(f"Expected {length} values, got {len(value)}: {value}")
        return list(value)
    return [value for _ in range(length)]


def _infer_backbone_channels(backbone):
    if hasattr(backbone, 'out_channels'):
        return list(backbone.out_channels)

    if hasattr(backbone, '_out_channels'):
        channels = list(backbone._out_channels)
        return_idx = getattr(backbone, 'return_idx', range(len(channels)))
        return [channels[i] for i in return_idx]

    raise ValueError(
        "fusion_channels must be set because the backbone does not expose "
        "out_channels or _out_channels."
    )


class Rot180CrossViewFusion(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=8,
        window_size=8,
        shift_size=0,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        reverse_aux_windows=False,
        gate_init=-4.0,
        use_similarity_gate=True,
        reverse_similarity_gate=False,
    ):
        super().__init__()
        self.channels = list(channels)
        num_heads = _as_list(num_heads, len(self.channels))
        window_size = _as_list(window_size, len(self.channels))
        shift_size = _as_list(shift_size, len(self.channels))
        self.reverse_aux_windows = reverse_aux_windows
        self.use_similarity_gate = use_similarity_gate
        self.reverse_similarity_gate = reverse_similarity_gate

        self.fusions = nn.ModuleList([
            CrossWinBasedAttention(
                dim=dim,
                num_heads=heads,
                window_size=win,
                shift_size=shift,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                reverse_aux_windows=reverse_aux_windows,
            )
            for dim, heads, win, shift in zip(self.channels, num_heads, window_size, shift_size)
        ])
        self.gate_logits = nn.ParameterList([
            nn.Parameter(torch.tensor(float(gate_init))) for _ in self.channels
        ])
        self.window_size = window_size

    @staticmethod
    def _pad_to_window(x, window_size):
        _, _, h, w = x.shape
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        if pad_h == 0 and pad_w == 0:
            return x, (h, w)
        return F.pad(x, (0, pad_w, 0, pad_h)), (h, w)

    def forward(self, main_feats, aux_feats):
        if len(main_feats) != len(aux_feats):
            raise ValueError(f"Feature level mismatch: {len(main_feats)} vs {len(aux_feats)}")
        if len(main_feats) != len(self.fusions):
            raise ValueError(f"Fusion level mismatch: got {len(main_feats)} feature levels, configured {len(self.fusions)}")

        fused_feats = []
        for main, aux, fusion, gate_logit, win in zip(main_feats, aux_feats, self.fusions, self.gate_logits, self.window_size):
            main_pad, (h, w) = self._pad_to_window(main, win)
            aux_pad, _ = self._pad_to_window(aux, win)
            candidate = fusion(main_pad, aux_pad)
            delta = candidate - main_pad

            if self.use_similarity_gate:
                aux_for_similarity = aux_pad
                if self.reverse_aux_windows and self.reverse_similarity_gate:
                    aux_for_similarity, _ = self._pad_to_window(aux[..., :h, :w].flip(-2, -1), win)
                main_norm = F.normalize(main_pad.float(), dim=1)
                aux_norm = F.normalize(aux_for_similarity.float(), dim=1)
                similarity = (main_norm * aux_norm).sum(dim=1, keepdim=True).add(1.).mul(0.5)
                similarity = similarity.to(dtype=main_pad.dtype)
            else:
                similarity = 1.

            fused = main_pad + torch.sigmoid(gate_logit).to(dtype=main_pad.dtype) * similarity * delta
            fused_feats.append(fused[..., :h, :w])

        return fused_feats


class LightweightRot180Fusion(nn.Module):
    def __init__(
        self,
        channels,
        reverse_aux_windows=False,
        gate_init=-4.0,
        use_similarity_gate=True,
        reverse_similarity_gate=False,
    ):
        super().__init__()
        self.channels = list(channels)
        self.reverse_aux_windows = reverse_aux_windows
        self.use_similarity_gate = use_similarity_gate
        self.reverse_similarity_gate = reverse_similarity_gate
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            )
            for dim in self.channels
        ])
        self.gate_logits = nn.ParameterList([
            nn.Parameter(torch.tensor(float(gate_init))) for _ in self.channels
        ])

    def forward(self, main_feats, aux_feats):
        if len(main_feats) != len(aux_feats):
            raise ValueError(f"Feature level mismatch: {len(main_feats)} vs {len(aux_feats)}")
        if len(main_feats) != len(self.blocks):
            raise ValueError(f"Fusion level mismatch: got {len(main_feats)} feature levels, configured {len(self.blocks)}")

        fused_feats = []
        for main, aux, block, gate_logit in zip(main_feats, aux_feats, self.blocks, self.gate_logits):
            aux_aligned = aux
            if self.reverse_aux_windows:
                aux_aligned = aux.flip(-2, -1)

            if self.use_similarity_gate:
                aux_for_similarity = aux_aligned if self.reverse_similarity_gate else aux
                main_norm = F.normalize(main.float(), dim=1)
                aux_norm = F.normalize(aux_for_similarity.float(), dim=1)
                similarity = (main_norm * aux_norm).sum(dim=1, keepdim=True).add(1.).mul(0.5)
                similarity = similarity.to(dtype=main.dtype)
            else:
                similarity = 1.

            delta = block(aux_aligned - main)
            fused = main + torch.sigmoid(gate_logit).to(dtype=main.dtype) * similarity * delta
            fused_feats.append(fused)

        return fused_feats


class BottleneckRot180CrossViewFusion(nn.Module):
    def __init__(
        self,
        channels,
        bottleneck_dim=128,
        num_heads=8,
        window_size=8,
        shift_size=0,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        reverse_aux_windows=False,
        gate_init=-4.0,
        use_similarity_gate=True,
        reverse_similarity_gate=False,
    ):
        super().__init__()
        self.channels = list(channels)
        bottleneck_dim = _as_list(bottleneck_dim, len(self.channels))
        num_heads = _as_list(num_heads, len(self.channels))
        window_size = _as_list(window_size, len(self.channels))
        shift_size = _as_list(shift_size, len(self.channels))
        self.reverse_aux_windows = reverse_aux_windows
        self.use_similarity_gate = use_similarity_gate
        self.reverse_similarity_gate = reverse_similarity_gate

        self.main_proj = nn.ModuleList([
            nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
            for dim, hidden in zip(self.channels, bottleneck_dim)
        ])
        self.aux_proj = nn.ModuleList([
            nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
            for dim, hidden in zip(self.channels, bottleneck_dim)
        ])
        self.fusions = nn.ModuleList([
            CrossWinBasedAttention(
                dim=hidden,
                num_heads=heads,
                window_size=win,
                shift_size=shift,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                reverse_aux_windows=reverse_aux_windows,
            )
            for hidden, heads, win, shift in zip(bottleneck_dim, num_heads, window_size, shift_size)
        ])
        self.out_proj = nn.ModuleList([
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True)
            for dim, hidden in zip(self.channels, bottleneck_dim)
        ])
        self.gate_logits = nn.ParameterList([
            nn.Parameter(torch.tensor(float(gate_init))) for _ in self.channels
        ])
        self.window_size = window_size

    @staticmethod
    def _pad_to_window(x, window_size):
        _, _, h, w = x.shape
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        if pad_h == 0 and pad_w == 0:
            return x, (h, w)
        return F.pad(x, (0, pad_w, 0, pad_h)), (h, w)

    def forward(self, main_feats, aux_feats):
        if len(main_feats) != len(aux_feats):
            raise ValueError(f"Feature level mismatch: {len(main_feats)} vs {len(aux_feats)}")
        if len(main_feats) != len(self.fusions):
            raise ValueError(f"Fusion level mismatch: got {len(main_feats)} feature levels, configured {len(self.fusions)}")

        fused_feats = []
        for main, aux, main_proj, aux_proj, fusion, out_proj, gate_logit, win in zip(
            main_feats, aux_feats, self.main_proj, self.aux_proj, self.fusions,
            self.out_proj, self.gate_logits, self.window_size
        ):
            main_pad, (h, w) = self._pad_to_window(main, win)
            aux_pad, _ = self._pad_to_window(aux, win)

            main_low = main_proj(main_pad)
            aux_low = aux_proj(aux_pad)
            candidate_low = fusion(main_low, aux_low)
            delta = out_proj(candidate_low - main_low)

            if self.use_similarity_gate:
                aux_for_similarity = aux_pad
                if self.reverse_aux_windows and self.reverse_similarity_gate:
                    aux_for_similarity, _ = self._pad_to_window(aux[..., :h, :w].flip(-2, -1), win)
                main_norm = F.normalize(main_pad.float(), dim=1)
                aux_norm = F.normalize(aux_for_similarity.float(), dim=1)
                similarity = (main_norm * aux_norm).sum(dim=1, keepdim=True).add(1.).mul(0.5)
                similarity = similarity.to(dtype=main_pad.dtype)
            else:
                similarity = 1.

            fused = main_pad + torch.sigmoid(gate_logit).to(dtype=main_pad.dtype) * similarity * delta
            fused_feats.append(fused[..., :h, :w])

        return fused_feats


@register()
class DFINE(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        use_rot180_aux=False,
        fusion_channels=None,
        fusion_num_heads=8,
        fusion_window_size=8,
        fusion_shift_size=0,
        fusion_bottleneck_dim=128,
        fusion_drop=0.,
        fusion_attn_drop=0.,
        fusion_drop_path=0.,
        fusion_type='cwam',
        fusion_reverse_aux_windows=False,
        fusion_gate_init=-4.0,
        fusion_use_similarity_gate=True,
        fusion_reverse_similarity_gate=False,
        share_rot180_backbone=False,
        freeze_main_backbone=False,
        freeze_rot180_backbone=False,
        freeze_encoder=False,
        freeze_decoder=False,
        frozen_modules_eval=True,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.use_rot180_aux = use_rot180_aux
        self.share_rot180_backbone = share_rot180_backbone
        self.fusion_type = fusion_type
        self.freeze_main_backbone = freeze_main_backbone
        self.freeze_rot180_backbone = freeze_rot180_backbone
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.frozen_modules_eval = frozen_modules_eval

        if self.use_rot180_aux:
            if not self.share_rot180_backbone:
                # Start from the same initialization, then train the two views
                # with independent backbone parameters and normalization states.
                self.rot180_backbone = copy.deepcopy(backbone)
            fusion_channels = fusion_channels if fusion_channels is not None else _infer_backbone_channels(backbone)
            if fusion_type == 'cwam':
                self.rot180_fusion = Rot180CrossViewFusion(
                    channels=fusion_channels,
                    num_heads=fusion_num_heads,
                    window_size=fusion_window_size,
                    shift_size=fusion_shift_size,
                    drop=fusion_drop,
                    attn_drop=fusion_attn_drop,
                    drop_path=fusion_drop_path,
                    reverse_aux_windows=fusion_reverse_aux_windows,
                    gate_init=fusion_gate_init,
                    use_similarity_gate=fusion_use_similarity_gate,
                    reverse_similarity_gate=fusion_reverse_similarity_gate,
                )
            elif fusion_type == 'light':
                self.rot180_fusion = LightweightRot180Fusion(
                    channels=fusion_channels,
                    reverse_aux_windows=fusion_reverse_aux_windows,
                    gate_init=fusion_gate_init,
                    use_similarity_gate=fusion_use_similarity_gate,
                    reverse_similarity_gate=fusion_reverse_similarity_gate,
                )
            elif fusion_type == 'bottleneck_cwam':
                self.rot180_fusion = BottleneckRot180CrossViewFusion(
                    channels=fusion_channels,
                    bottleneck_dim=fusion_bottleneck_dim,
                    num_heads=fusion_num_heads,
                    window_size=fusion_window_size,
                    shift_size=fusion_shift_size,
                    drop=fusion_drop,
                    attn_drop=fusion_attn_drop,
                    drop_path=fusion_drop_path,
                    reverse_aux_windows=fusion_reverse_aux_windows,
                    gate_init=fusion_gate_init,
                    use_similarity_gate=fusion_use_similarity_gate,
                    reverse_similarity_gate=fusion_reverse_similarity_gate,
                )
            else:
                raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self._apply_freeze_config()

    @staticmethod
    def _set_requires_grad(module, requires_grad):
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def _apply_freeze_config(self):
        if self.freeze_main_backbone:
            self._set_requires_grad(self.backbone, False)
        if self.use_rot180_aux and not self.share_rot180_backbone and self.freeze_rot180_backbone:
            self._set_requires_grad(self.rot180_backbone, False)
        if self.freeze_encoder:
            self._set_requires_grad(self.encoder, False)
        if self.freeze_decoder:
            self._set_requires_grad(self.decoder, False)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.frozen_modules_eval:
            if self.freeze_main_backbone:
                self.backbone.eval()
            if self.use_rot180_aux and not self.share_rot180_backbone and self.freeze_rot180_backbone:
                self.rot180_backbone.eval()
            if self.freeze_encoder:
                self.encoder.eval()
        return self

    def forward(self, x, targets=None):
        if self.use_rot180_aux:
            x_rot = torch.rot90(x, 2, dims=(-2, -1))
            main_feats = self.backbone(x)
            rot180_feats = self.backbone(x_rot) if self.share_rot180_backbone else self.rot180_backbone(x_rot)
            x = self.rot180_fusion(main_feats, rot180_feats)
        else:
            x = self.backbone(x)

        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
