"""
Vision Transformer for brain age prediction
"""

import torch
import torch.nn as nn

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
        Vision Transformer for brain age regression
        
        Args:
            image_size: Input image size
            patch_size: Size of each patch
            num_channels: Number of input channels
            dim: Embedding dimension
            depth: Number of transformer layers
            heads: Number of attention heads
            mlp_dim: MLP hidden dimension
            dropout: Dropout rate
        """
        super(BrainAgeViT, self).__init__()
        
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = num_channels * patch_size * patch_size
        
        self.patch_size = patch_size
        self.dim = dim
        
        # Patch embedding
        self.patch_embedding = nn.Linear(patch_dim, dim)
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Regression head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Predicted normalized age (B,)
        """
        B = x.shape[0]
        
        # Create patches: (B, num_patches, patch_dim)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, -1, self.patch_size * self.patch_size * x.shape[1])
        
        # Embed patches: (B, num_patches, dim)
        x = self.patch_embedding(patches)
        
        # Add CLS token: (B, num_patches+1, dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Transformer
        x = self.transformer(x)
        
        # Use CLS token for prediction
        cls_output = x[:, 0]
        age = self.mlp_head(cls_output)
        
        return age.squeeze(-1)