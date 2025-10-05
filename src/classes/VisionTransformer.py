import torch
import torch.nn as nn
from classes.PatchEmbedding import PatchEmbedding
from classes.TransformerBlock import TransformerBlock

class VisionTransformer(nn.Module):
    """Vision Transformer for Image Classification"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 num_classes=4, embed_dim=768, depth=12, n_heads=12, 
                 mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        
        # Class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x)
            attention_weights.append(attn)
        
        x = self.norm(x)
        
        # Classification head (use class token)
        x = self.head(x[:, 0])
        
        return x
