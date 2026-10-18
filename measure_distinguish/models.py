"""Image-to-direction task models. No ground-truth masks enter these models."""
from __future__ import annotations

import os

# This project uses PyTorch exclusively, including the optional HF adapter.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from torch import nn


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, image):
        return (image-self.mean)/self.std


class SmallViT(nn.Module):
    def __init__(self, size=128, patch=8, width=192, depth=6, heads=6):
        super().__init__()
        if size % patch or width % heads:
            raise ValueError("Size/patch and width/heads must divide evenly")
        self.size = size
        self.normalize = Normalize()
        self.patch = nn.Conv2d(3, width, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.empty(1, (size//patch)**2+1, width))
        nn.init.trunc_normal_(self.position, std=.02)
        layer = nn.TransformerEncoderLayer(width, heads, width*4, dropout=0.,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, image):
        if image.shape[-2:] != (self.size, self.size):
            raise ValueError("Input resolution differs from configured SmallViT size")
        tokens = self.patch(self.normalize(image)).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls.expand(len(image), -1, -1), tokens), dim=1)
        return self.head(self.norm(self.blocks(tokens+self.position))[:, 0]).squeeze(-1)


class ResNetReporter(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        self.normalize = Normalize()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, image):
        return self.backbone(self.normalize(image)).squeeze(-1)


class DinoReporter(nn.Module):
    def __init__(self, model_id="facebook/dinov2-small", revision=None, cache_dir=None,
                 pretrained=True, offline=True, freeze_backbone=False):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model
        self.normalize = Normalize()
        if pretrained:
            self.backbone = Dinov2Model.from_pretrained(
                model_id, revision=revision, cache_dir=cache_dir,
                local_files_only=offline, attn_implementation="eager")
        else:
            # Exact DINOv2-S architecture; not labeled as a pretrained model.
            self.backbone = Dinov2Model(Dinov2Config(
                hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
                image_size=518, patch_size=14, layerscale_value=1.,
                hidden_act="gelu", mlp_ratio=4, layer_norm_eps=1e-6,
                _attn_implementation="eager"))
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, image):
        if image.shape[-1] % 14 or image.shape[-2] % 14:
            raise ValueError("Use dimensions divisible by DINOv2 patch size 14")
        hidden = self.backbone(pixel_values=self.normalize(image)).last_hidden_state[:, 0]
        return self.head(hidden).squeeze(-1)


def make_model(config: dict):
    model = config["model"]
    name = model["name"]
    if name == "small_vit":
        return SmallViT(size=config["data"]["size"], **model.get("kwargs", {}))
    if name == "resnet18":
        return ResNetReporter()
    if name == "dinov2_small":
        return DinoReporter(**model.get("kwargs", {}))
    raise ValueError(f"Unsupported model: {name}")
