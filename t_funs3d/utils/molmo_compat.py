"""Compatibility for Molmo's vision backbone on the pinned PyTorch 2.1 stack."""
from types import MethodType

import torch


def _encode_image(self, images):
    # Mirrors OLMoPretrainedVisionBackbone.encode_image, replacing only the
    # tuple-dimension reduction unsupported by PyTorch 2.1.
    batch, crops, patches, pixels = images.shape
    images = images.view(batch * crops, patches, pixels)
    mask = ~(images == -1).all(dim=2, keepdim=True).all(dim=1, keepdim=True)
    hidden_states = self.image_vit(images)
    if self.config.vit_layers is not None:
        features = torch.cat([hidden_states[i] for i in self.config.vit_layers], dim=-1)
    else:
        features = hidden_states[-1]
    cls_embed = None
    if self.num_prefix_tokens > 0:
        cls_embed = features[:, 0]
        features = features[:, 1:]
    features = (features * mask).view(batch, crops, patches, -1)
    if cls_embed is not None:
        cls_embed = cls_embed.view(batch, crops, -1)
    return features, cls_embed


def apply_molmo_torch_compat(model):
    """Patch only this Molmo instance when torch.all lacks tuple dimensions."""
    try:
        torch.all(torch.ones(1, 1, 1, dtype=torch.bool), dim=(1, 2), keepdim=True)
    except TypeError:
        backbone = model.model.vision_backbone
        backbone.encode_image = MethodType(_encode_image, backbone)
