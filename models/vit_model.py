"""
Pretrained Vision Transformer using timm library
Uses DeiT-Small pretrained on ImageNet → fine-tuned for brain age
"""

import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("timm not installed. Run: pip install timm")


class BrainAgeViT(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_channels=3,
        dim=384,
        depth=6,
        heads=6,
        mlp_dim=768,
        dropout=0.1
    ):
        """
        Pretrained ViT for brain age regression
        Uses timm's pretrained deit_small_patch16_224
        All other args kept for compatibility but not used
        """
        super(BrainAgeViT, self).__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("Run: pip install timm")

        # Load pretrained DeiT-Small (ImageNet pretrained)
        # Much better than training from scratch with 322 samples!
        self.backbone = timm.create_model(
            'deit_small_patch16_224',
            pretrained=True,
            num_classes=0,      # Remove classification head
            drop_rate=dropout,
            drop_path_rate=0.1
        )

        # Get feature dimension from backbone
        feature_dim = self.backbone.num_features  # 384 for deit_small

        # Custom regression head for brain age
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1)       # Predict normalized age
        )

        print(f"✅ Loaded pretrained DeiT-Small (ImageNet)")
        print(f"   Feature dim: {feature_dim}")
        total = sum(p.numel() for p in self.parameters())
        print(f"   Total params: {total:,}")

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, 224, 224)
        Returns:
            Predicted normalized age (B,)
        """
        features = self.backbone(x)   # (B, 384)
        age      = self.head(features) # (B, 1)
        return age.squeeze(-1)