#!/usr/bin/env python3
"""AV word recognizer with a visual gate on audio features."""

from __future__ import annotations

import torch
import torch.nn as nn

from train import ResBlock


class _Conv3DResBlock(nn.Module):
    """3D residual block for lip video."""

    def __init__(self, in_ch: int, out_ch: int, stride=(1, 1, 1)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        if stride != (1, 1, 1) or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class VisualEncoder(nn.Module):
    """Map (B, 1, T, 88, 88) lip clips to (B, 64, 40, 50) features."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(5, 7, 7),
                      stride=(1, 1, 1), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.res1 = _Conv3DResBlock(8, 16, stride=(1, 1, 1))
        self.res2 = _Conv3DResBlock(16, 32, stride=(1, 2, 2))
        self.res3 = _Conv3DResBlock(32, 64, stride=(1, 1, 1))
        # Pool variable-length clips to 50 time steps and collapse width.
        self.pool = nn.AdaptiveAvgPool3d((50, 40, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)               # (B, 64, 50, 40, 1)
        x = x.squeeze(-1)              # (B, 64, 50, 40)
        x = x.permute(0, 1, 3, 2).contiguous()  # (B, 64, 40, 50)
        return x


class CrossModalGate(nn.Module):
    """Scale audio features: a_out = a * (1 + α · σ(W_a·a + W_v·v)).

    α is learned; its default initial value is 0.2.
    """

    def __init__(self, channels: int = 64, alpha_init: float = 0.2):
        super().__init__()
        self.Wa = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.Wv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, a_mid: torch.Tensor, v_mid: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.Wa(a_mid) + self.Wv(v_mid))
        return a_mid * (1.0 + self.alpha * g)


class AVWordResNet(nn.Module):
    """Audio ResNet with visual gating between its two residual blocks."""

    def __init__(self, num_classes: int, alpha_init: float = 0.2):
        super().__init__()
        # Audio path
        self.audio_block1 = ResBlock(1, 64, stride=2)
        self.audio_block2 = ResBlock(64, 128, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

        # Visual path
        self.visual = VisualEncoder()

        # Mid-level fusion
        self.gate = CrossModalGate(channels=64, alpha_init=alpha_init)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor | None = None,
        audio_dead: torch.Tensor | None = None,
        video_dead: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Classify audio, video, or paired input using the same AV weights.

        Boolean masks of shape (B,) mark missing inputs. Missing audio zeroes
        the mel input; missing video zeroes pixels and encoded features.
        `video=None` also sets visual features to zero.
        """
        # audio: (B, 1, 80, 99); video: (B, 1, T, 88, 88) or None.
        if audio_dead is not None:
            audio = audio * (~audio_dead).view(-1, 1, 1, 1).to(audio.dtype)
        a_mid = self.audio_block1(audio)        # (B, 64, 40, 50)
        if video is None:
            v_mid = torch.zeros_like(a_mid)
        else:
            if video_dead is not None:
                video = video * (~video_dead).view(-1, 1, 1, 1, 1).to(video.dtype)
            v_mid = self.visual(video)          # (B, 64, 40, 50)
            if video_dead is not None:
                v_mid = v_mid * (~video_dead).view(-1, 1, 1, 1).to(v_mid.dtype)
        a_fused = self.gate(a_mid, v_mid)       # (B, 64, 40, 50)
        x = self.audio_block2(a_fused)          # (B, 128, 20, 25)
        x = self.gap(x).flatten(1)              # (B, 128)
        x = self.dropout(x)
        return self.fc(x)                       # (B, num_classes)

    def load_audio_pretrained(self, checkpoint_state_dict: dict) -> None:
        """Load an audio checkpoint, renaming block1 and block2 keys."""
        renames = {
            "block1": "audio_block1",
            "block2": "audio_block2",
        }
        new_sd = {}
        for k, v in checkpoint_state_dict.items():
            for old, new in renames.items():
                if k.startswith(old + "."):
                    new_sd[new + k[len(old):]] = v
                    break
            else:
                new_sd[k] = v
        # Keep the visual encoder and gate at their current weights.
        self.load_state_dict(new_sd, strict=False)
