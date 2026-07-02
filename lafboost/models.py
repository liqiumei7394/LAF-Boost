from __future__ import annotations

import math
from typing import List, Sequence

import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.down(x)


class TCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, levels: int = 3, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        ch = in_dim
        for i in range(levels):
            layers.append(TemporalBlock(ch, hidden, kernel, 2**i, dropout))
            ch = hidden
        self.net = nn.Sequential(*layers)
        self.attn = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        # x: B, L, F
        h = self.net(x.transpose(1, 2)).transpose(1, 2)
        w = torch.softmax(self.attn(h), dim=1)
        return (h * w).sum(dim=1)


class TCNForecast(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, input_len: int, n_floors: int, n_zones: int, hidden: int = 64, dropout: float = 0.1, **_):
        super().__init__()
        self.encoder = TCNEncoder(input_dim, hidden, dropout=dropout)
        self.floor_emb = nn.Embedding(n_floors, 8)
        self.zone_emb = nn.Embedding(n_zones, 8)
        self.head = nn.Sequential(nn.Linear(hidden + 16, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, output_dim))

    def forward(self, x, floor, zone, seasonal_bases=None):
        h = self.encoder(x)
        e = torch.cat([self.floor_emb(floor), self.zone_emb(zone)], dim=-1)
        return self.head(torch.cat([h, e], dim=-1))


class PatchTSTLite(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, input_len: int, n_floors: int, n_zones: int, hidden: int = 64, patch_len: int = 16, dropout: float = 0.1, **_):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = math.ceil(input_len / patch_len)
        self.proj = nn.Linear(input_dim * patch_len, hidden)
        enc_layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.floor_emb = nn.Embedding(n_floors, 8)
        self.zone_emb = nn.Embedding(n_zones, 8)
        self.head = nn.Sequential(nn.Linear(hidden + 16, hidden), nn.GELU(), nn.Linear(hidden, output_dim))

    def forward(self, x, floor, zone, seasonal_bases=None):
        b, l, f = x.shape
        pad = self.n_patches * self.patch_len - l
        if pad:
            x = torch.cat([x, x[:, -1:, :].repeat(1, pad, 1)], dim=1)
        x = x.reshape(b, self.n_patches, self.patch_len * f)
        h = self.encoder(self.proj(x)).mean(dim=1)
        e = torch.cat([self.floor_emb(floor), self.zone_emb(zone)], dim=-1)
        return self.head(torch.cat([h, e], dim=-1))


class ITransformerLite(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, input_len: int, n_floors: int, n_zones: int, hidden: int = 64, dropout: float = 0.1, **_):
        super().__init__()
        self.var_proj = nn.Linear(input_len, hidden)
        self.var_emb = nn.Parameter(torch.randn(1, input_dim, hidden) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.floor_emb = nn.Embedding(n_floors, 8)
        self.zone_emb = nn.Embedding(n_zones, 8)
        self.head = nn.Sequential(nn.Linear(hidden + 16, hidden), nn.GELU(), nn.Linear(hidden, output_dim))

    def forward(self, x, floor, zone, seasonal_bases=None):
        h = self.var_proj(x.transpose(1, 2)) + self.var_emb[:, : x.shape[-1]]
        h = self.encoder(h).mean(dim=1)
        e = torch.cat([self.floor_emb(floor), self.zone_emb(zone)], dim=-1)
        return self.head(torch.cat([h, e], dim=-1))


class TimeMixerLite(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, input_len: int, n_floors: int, n_zones: int, hidden: int = 64, dropout: float = 0.1, **_):
        super().__init__()
        self.scales = [1, 4, 12]
        self.mixers = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)) for _ in self.scales])
        self.floor_emb = nn.Embedding(n_floors, 8)
        self.zone_emb = nn.Embedding(n_zones, 8)
        self.head = nn.Sequential(nn.Linear(hidden * len(self.scales) + 16, hidden), nn.GELU(), nn.Linear(hidden, output_dim))

    def forward(self, x, floor, zone, seasonal_bases=None):
        reps = []
        for scale, mixer in zip(self.scales, self.mixers):
            if scale > 1:
                pooled = nn.functional.avg_pool1d(x.transpose(1, 2), kernel_size=scale, stride=scale).transpose(1, 2)
            else:
                pooled = x
            reps.append(mixer(pooled).mean(dim=1))
        e = torch.cat([self.floor_emb(floor), self.zone_emb(zone)], dim=-1)
        return self.head(torch.cat(reps + [e], dim=-1))


class TimesNetLite(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, input_len: int, n_floors: int, n_zones: int, hidden: int = 64, dropout: float = 0.1, **_):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden)
        self.convs = nn.ModuleList([nn.Conv1d(hidden, hidden, k, padding=k // 2) for k in [3, 5, 9, 15]])
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.floor_emb = nn.Embedding(n_floors, 8)
        self.zone_emb = nn.Embedding(n_zones, 8)
        self.head = nn.Sequential(nn.Linear(hidden + 16, hidden), nn.GELU(), nn.Linear(hidden, output_dim))

    def forward(self, x, floor, zone, seasonal_bases=None):
        h = self.inp(x).transpose(1, 2)
        hs = [torch.relu(conv(h)) for conv in self.convs]
        h = torch.stack(hs, dim=0).mean(dim=0).transpose(1, 2)
        h = self.dropout(self.norm(h)).mean(dim=1)
        e = torch.cat([self.floor_emb(floor), self.zone_emb(zone)], dim=-1)
        return self.head(torch.cat([h, e], dim=-1))


class LAFNet(nn.Module):
    """Lighting-aware Adaptive Fusion Network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        input_len: int,
        n_floors: int,
        n_zones: int,
        feature_names: Sequence[str],
        hidden: int = 64,
        dropout: float = 0.1,
        use_env: bool = True,
        use_related: bool = True,
        use_gate: bool = True,
        use_decomp: bool = True,
        **_,
    ):
        super().__init__()
        self.feature_names = list(feature_names)
        self.use_env = use_env
        self.use_related = use_related
        self.use_gate = use_gate
        self.use_decomp = use_decomp
        self.light_idx = [self.feature_names.index("light")]
        self.env_idx = [self.feature_names.index(c) for c in ["lux", "temp", "rh"] if c in self.feature_names]
        self.related_idx = [self.feature_names.index(c) for c in ["plug", "ac"] if c in self.feature_names]
        self.time_idx = [i for i, c in enumerate(self.feature_names) if c not in ["light", "plug", "ac", "lux", "temp", "rh"]]

        light_in = 3 if use_decomp else 1
        self.light_enc = TCNEncoder(light_in, hidden, dropout=dropout)
        self.env_enc = TCNEncoder(max(1, len(self.env_idx)), hidden, dropout=dropout)
        self.rel_enc = TCNEncoder(max(1, len(self.related_idx)), hidden, dropout=dropout)
        self.time_proj = nn.Sequential(nn.Linear(max(1, len(self.time_idx)), hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.stat_proj = nn.Sequential(nn.Linear(input_dim * 3, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.floor_emb = nn.Embedding(n_floors, 8)
        self.zone_emb = nn.Embedding(n_zones, 8)
        self.gate = nn.Sequential(nn.Linear(hidden * 5 + 16, hidden), nn.GELU(), nn.Linear(hidden, 5))
        self.base_gate = nn.Sequential(nn.Linear(hidden * 5 + 16, hidden), nn.GELU(), nn.Linear(hidden, output_dim * 3))
        self.head = nn.Sequential(nn.Linear(hidden * 5 + 16, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, output_dim))

    def moving_average(self, y: torch.Tensor, kernel: int = 31) -> torch.Tensor:
        pad = kernel // 2
        yp = nn.functional.pad(y.transpose(1, 2), (pad, pad), mode="replicate")
        return nn.functional.avg_pool1d(yp, kernel_size=kernel, stride=1).transpose(1, 2)

    def pick(self, x: torch.Tensor, idx: List[int]) -> torch.Tensor:
        if not idx:
            return torch.zeros(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
        return x[:, :, idx]

    def forward(self, x, floor, zone, seasonal_bases=None):
        light = self.pick(x, self.light_idx)
        last_light = light[:, -1, :]
        if self.use_decomp:
            trend = self.moving_average(light)
            resid = light - trend
            light_in = torch.cat([light, trend, resid], dim=-1)
        else:
            light_in = light
        fl = self.light_enc(light_in)

        if self.use_env:
            fe = self.env_enc(self.pick(x, self.env_idx))
        else:
            fe = torch.zeros_like(fl)

        if self.use_related:
            fr = self.rel_enc(self.pick(x, self.related_idx))
        else:
            fr = torch.zeros_like(fl)

        ft = self.time_proj(self.pick(x, self.time_idx).mean(dim=1))
        stat = torch.cat([x.mean(dim=1), x.std(dim=1), x.max(dim=1).values - x.min(dim=1).values], dim=-1)
        fs = self.stat_proj(stat)
        emb = torch.cat([self.floor_emb(floor), self.zone_emb(zone)], dim=-1)
        parts = torch.stack([fl, fe, fr, ft, fs], dim=1)
        if self.use_gate:
            gate_in = torch.cat([fl, fe, fr, ft, fs, emb], dim=-1)
            weights = torch.softmax(self.gate(gate_in), dim=-1).unsqueeze(-1)
            fused_parts = (parts * weights).reshape(x.shape[0], -1)
        else:
            fused_parts = parts.reshape(x.shape[0], -1)
        fused = torch.cat([fused_parts, emb], dim=-1)
        delta = self.head(fused)
        if seasonal_bases is None:
            anchor = last_light.repeat(1, delta.shape[-1])
        else:
            base_weights = self.base_gate(fused).view(x.shape[0], delta.shape[-1], 3)
            base_weights = torch.softmax(base_weights, dim=-1)
            anchor = (seasonal_bases * base_weights).sum(dim=-1)
        # Forecast residuals around adaptive periodic anchors. The anchors
        # encode short-term persistence, yesterday's same prediction time, and
        # last week's same prediction time without lengthening the input window.
        return anchor + delta


MODEL_REGISTRY = {
    "tcn": TCNForecast,
    "patchtst": PatchTSTLite,
    "timesnet": TimesNetLite,
    "itransformer": ITransformerLite,
    "timemixer": TimeMixerLite,
    "lafnet": LAFNet,
}
