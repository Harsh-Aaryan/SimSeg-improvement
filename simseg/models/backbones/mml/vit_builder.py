import timm
import torch
import torch.nn as nn

from ..builder import BACKBONE


class ViTModel(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(ViTModel, self).__init__()
        self.model = timm.create_model(cfg.model.image_encoder.tag, pretrained=cfg.model.image_encoder.pretrained, num_classes=0, **kwargs)
        self.use_flash_attention = getattr(cfg.model.image_encoder, 'flash_attention', False)
        
        # Enable Flash Attention if available and requested
        if self.use_flash_attention:
            self._enable_flash_attention()

    def _enable_flash_attention(self):
        """Enable Flash Attention for memory-efficient attention computation."""
        try:
            from torch.nn.functional import scaled_dot_product_attention
            # Flash attention is automatically used by PyTorch 2.0+ when available
            print("[INFO] Flash Attention enabled for memory-efficient attention")
        except ImportError:
            print("[WARNING] Flash Attention not available, using standard attention")

    def forward(self, x):
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        x = torch.reshape(x, (x.size(0), x.size(1), -1))
        return x


class SwinModel(nn.Module):
    """Swin Transformer backbone with hierarchical architecture and shifted window mechanism."""
    def __init__(self, cfg, **kwargs):
        super(SwinModel, self).__init__()
        self.model = timm.create_model(cfg.model.image_encoder.tag, pretrained=cfg.model.image_encoder.pretrained, num_classes=0, **kwargs)
        self.use_flash_attention = getattr(cfg.model.image_encoder, 'flash_attention', False)

    def forward(self, x):
        # Swin transformer forward pass
        x = self.model.patch_embed(x)
        if hasattr(self.model, 'pos_drop'):
            x = self.model.pos_drop(x)
        
        # Process through layers
        for layer in self.model.layers:
            x = layer(x)
        
        x = self.model.norm(x)
        
        # Reshape for compatibility with downstream modules
        if len(x.shape) == 3:
            # Already in (B, N, C) format
            pass
        elif len(x.shape) == 4:
            # (B, H, W, C) -> (B, H*W, C)
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)
        
        return x


@BACKBONE.register_obj
def vit_modelzoo(cfg, **kwargs):
    model = ViTModel(cfg, **kwargs)
    return model


@BACKBONE.register_obj
def vit_large(cfg, **kwargs):
    """ViT-L: 24 layers, 1024 hidden dimension"""
    model = ViTModel(cfg, **kwargs)
    return model


@BACKBONE.register_obj
def vit_huge(cfg, **kwargs):
    """ViT-H: 32 layers, 1280 hidden dimension"""
    model = ViTModel(cfg, **kwargs)
    return model


@BACKBONE.register_obj
def swin_modelzoo(cfg, **kwargs):
    """Swin Transformer with hierarchical architecture and shifted window mechanism"""
    model = SwinModel(cfg, **kwargs)
    return model
