import math
import torch
import torch.nn as nn
from einops import rearrange, einsum
from mamba_ssm import Mamba


# ==================== mHC CORE ====================

def sinkhorn_projection(logits, iters=10, tau=0.05):
    """Sinkhorn projection for doubly stochastic matrices."""
    Z = logits / tau
    u = v = torch.zeros_like(Z[..., 0])
    for _ in range(iters):
        u = -torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = -torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)
    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))


class SpectralStreamHyperConnections(nn.Module):
    """
    mHC where spectral tokens are treated as streams.
    Spatial Mamba serves as the branch function F(.).
    """
    
    def __init__(
        self,
        channels,           # Input channels (spectral bands)
        token_num=8,        # Number of spectral tokens = mHC streams
        use_residual=True,
        group_num=4,
        mhc_iters=10,
        mhc_tau=0.05,
        layer_index=0       # For cycling H_pre focus
    ):
        super().__init__()
        
        self.token_num = token_num  # This is now mHC's num_streams
        self.use_residual = use_residual
        
        # Calculate spectral token dimensions
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num
        
        # ===== mHC PARAMETERS =====
        # H_res: Spectral correlation matrix (token-to-token mixing)
        self.H_res_logits = nn.Parameter(
            torch.full((token_num, token_num), -10.0)
        )
        with torch.no_grad():
            self.H_res_logits.fill_diagonal_(0.0)  # Prefer identity
        
        # H_pre: Aggregates all tokens into one for spatial processing
        self.H_pre_logits = nn.Parameter(torch.zeros(1, token_num))
        with torch.no_grad():
            # Cycle focus through tokens
            selected_token = layer_index % token_num
            self.H_pre_logits[:, selected_token] = 5.0
        
        # H_post: Distributes spatial features back to tokens
        self.H_post_logits = nn.Parameter(torch.zeros(1, token_num))
        
        # ===== BRANCH FUNCTION: Spatial Mamba =====
        self.spatial_branch = nn.Sequential(
            # Spatial Mamba processes the aggregated spectral features
            Mamba(
                d_model=self.group_channel_num,  # Process aggregated spectral dimension
                d_state=16,
                d_conv=4,
                expand=2,
            ),
            nn.GroupNorm(group_num, self.group_channel_num),
            nn.SiLU()
        )
        
        # Learnable scales (γ, β from mHC paper)
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.residual_scale = nn.Parameter(torch.ones(1))
        
        # Final projection
        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )
    
    def padding_feature(self, x):
        """Pad spectral dimension if needed."""
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            return torch.cat([x, pad_features], dim=1)
        return x
    
    def forward(self, x):
        """
        x: (B, C, H, W) - HSI cube
        
        Process:
        1. Group spectral bands into tokens (streams)
        2. H_res mixes spectral tokens (spectral correlation)
        3. H_pre aggregates to single stream
        4. Spatial Mamba processes spatial dimensions  
        5. H_post distributes back to tokens
        6. Residual connection
        """
        # 1. Pad if necessary
        x_pad = self.padding_feature(x)  # (B, C', H, W)
        
        # 2. Rearrange: (B, C', H, W) -> (B, H, W, C') -> (B*H*W, token_num, group_dim)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape
        
        x_tokens = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        # Now: (n, token_num, d) where n = B*H*W, token_num = streams
        
        # 3. Reshape for mHC: (n, streams, d)
        R = x_tokens  # R shape: (n, streams, d)
        
        # 4. H_res: Spectral token mixing (doubly stochastic)
        H_res = sinkhorn_projection(self.H_res_logits, iters=10)
        # H_res shape: (streams, streams)
        
        Rw = einsum(H_res, R, "s t, n s d -> n t d")
        # Rw: Mixed spectral tokens
        
        # 5. H_pre: Aggregate tokens to single stream for spatial processing
        H_pre = self.H_pre_logits.softmax(dim=-1)  # (1, streams)
        x_aggregated = einsum(H_pre, R, "1 s, n s d -> n d")
        # x_aggregated: (n, d) - All spectral info aggregated
        
        # 6. BRANCH: Spatial Mamba processing
        # Need to restore spatial structure for spatial Mamba
        x_spatial = x_aggregated.view(B, H, W, self.group_channel_num)
        x_spatial = x_spatial.permute(0, 3, 1, 2).contiguous()  # (B, d, H, W)
        
        # Apply spatial Mamba (processes spatial dimensions)
        # We need to flatten spatial dimensions for Mamba
        x_spatial_flat = x_spatial.permute(0, 2, 3, 1).contiguous()
        x_spatial_flat = x_spatial_flat.view(1, -1, self.group_channel_num)
        y_spatial = self.spatial_branch(x_spatial_flat)  # (1, B*H*W, d)
        
        # Reshape back
        y_spatial = y_spatial.view(B, H, W, self.group_channel_num)
        y_spatial = y_spatial.permute(0, 3, 1, 2).contiguous()  # (B, d, H, W)
        
        # Flatten for H_post distribution
        y = y_spatial.permute(0, 2, 3, 1).contiguous()
        y = y.view(B * H * W, self.group_channel_num)  # (n, d)
        
        # 7. H_post: Distribute spatial features back to spectral tokens
        H_post = self.H_post_logits.softmax(dim=-1)  # (1, streams)
        delta = einsum(y, H_post, "n d, 1 s -> n s d")
        # delta: (n, streams, d) - Spatial features distributed to tokens
        
        # 8. Combine: mHC formulation with learned scales
        out = self.gamma * (self.residual_scale * Rw + delta) + self.beta
        # out: (n, streams, d)
        
        # 9. Reshape back to HSI cube format
        x_recon = out.view(B, H, W, C_pad)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # (B, C', H, W)
        
        # 10. Projection and residual
        x_proj = self.proj(x_recon)
        
        if self.use_residual:
            # Need to match dimensions
            if x.shape != x_proj.shape:
                # Crop if padded
                x_proj = x_proj[:, :x.shape[1], :, :]
            return x + x_proj
        else:
            return x_proj


# ==================== UPDATED ARCHITECTURE ====================

class mHC_MambaHSI_Integrated(nn.Module):
    """
    Integrated architecture where:
    - Spectral tokens = mHC streams
    - H_res learns spectral correlations  
    - Spatial Mamba serves as branch function
    """
    
    def __init__(
        self,
        in_channels=128,
        hidden_dim=64,
        num_classes=10,
        token_num=8,           # Number of spectral tokens/mHC streams
        use_residual=True,
        group_num=4,
        use_dual_branch=False  # Option to keep original dual branch
    ):
        super().__init__()
        
        # Patch embedding (spectral reduction)
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim,
                     kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )
        
        # Multi-scale mHC-Spatial processing blocks
        self.blocks = nn.Sequential(
            # Block 1: Full resolution
            SpectralStreamHyperConnections(
                channels=hidden_dim,
                token_num=token_num,
                use_residual=use_residual,
                group_num=group_num,
                layer_index=0
            ),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            
            # Block 2: Half resolution
            SpectralStreamHyperConnections(
                channels=hidden_dim,
                token_num=token_num,
                use_residual=use_residual,
                group_num=group_num,
                layer_index=1
            ),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            
            # Block 3: Quarter resolution
            SpectralStreamHyperConnections(
                channels=hidden_dim,
                token_num=token_num,
                use_residual=use_residual,
                group_num=group_num,
                layer_index=2
            ),
        )
        
        # Upsampling to recover spatial resolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
            
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
        )
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(group_num, hidden_dim * 2),
            nn.SiLU(),
            
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
            
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1, stride=1, padding=0),
        )
    
    def forward(self, x):
        # 1. Spectral reduction
        x = self.patch_embedding(x)
        
        # 2. Multi-scale mHC-spatial processing
        x = self.blocks(x)
        
        # 3. Upsample to original spatial size
        x = self.upsample(x)
        
        # 4. Classification
        logits = self.cls_head(x)
        
        return logits


# ==================== DUAL-BRANCH VARIANT ====================

class DualBranch_mHC_MambaHSI(nn.Module):
    """
    Optional: Keep original dual-branch but replace SpeMamba with mHC version.
    This maintains both independent spatial and spectral pathways.
    """
    
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, token_num=8):
        super().__init__()
        
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(4, hidden_dim),
            nn.SiLU()
        )
        
        # Independent branches
        self.spectral_branch = SpectralStreamHyperConnections(
            channels=hidden_dim,
            token_num=token_num,
            use_residual=True
        )
        
        self.spatial_branch = nn.Sequential(
            SpaMamba(hidden_dim, use_residual=True),  # Original SpaMamba
            SpaMamba(hidden_dim, use_residual=True)
        )
        
        # Learnable fusion
        self.fusion_weights = nn.Parameter(torch.ones(2) / 2)
        self.softmax = nn.Softmax(dim=0)
        
        # Classification
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )
    
    def forward(self, x):
        x = self.patch_embedding(x)
        
        # Process independently
        x_spec = self.spectral_branch(x)
        x_spat = self.spatial_branch(x)
        
        # Adaptive fusion
        weights = self.softmax(self.fusion_weights)
        x_fused = x_spec * weights[0] + x_spat * weights[1]
        
        # Classification
        logits = self.cls_head(x_fused)
        
        return logits


# ==================== ANALYSIS & BENEFITS ====================

def analyze_integration():
    """
    Compare the integrated approach vs original.
    """
    print("=" * 60)
    print("ANALYSIS: mHC-Spectral Integration")
    print("=" * 60)
    
    # Original parameters
    original_params = sum(p.numel() for p in MambaHSI().parameters())
    
    # Integrated parameters  
    integrated_params = sum(p.numel() for p in mHC_MambaHSI_Integrated().parameters())
    
    print(f"\nParameter Comparison:")
    print(f"Original MambaHSI: {original_params:,}")
    print(f"Integrated mHC-MambaHSI: {integrated_params:,}")
    print(f"Difference: {integrated_params - original_params:,}")
    
    print(f"\nKey Integration Points:")
    print("1. Spectral tokens → mHC streams (natural alignment)")
    print("2. H_res matrix learns spectral correlations")
    print("3. H_pre aggregates spectral info for spatial processing")
    print("4. Spatial Mamba as branch function (preserves 2D structure)")
    print("5. H_post redistributes spatial features to spectral tokens")
    
    print(f"\nTheoretical Benefits:")
    print("✓ Explicit spectral correlation learning via H_res")
    print("✓ Doubly-stochastic constraints ensure stability")
    print("✓ Clear separation: spectral mixing → spatial processing")
    print("✓ Adaptive spectral aggregation via learnable H_pre/H_post")
    print("✓ Maintains spatial Mamba's efficiency for 2D processing")


# ==================== TEST ====================

if __name__ == "__main__":
    # Test the integrated architecture
    print("Testing Integrated mHC-MambaHSI Architecture...")
    
    # Create model
    model = mHC_MambaHSI_Integrated(
        in_channels=128,
        hidden_dim=64,
        num_classes=10,
        token_num=8
    )
    
    # Test input
    x = torch.randn(2, 128, 64, 64)  # HSI cube
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Run analysis
    analyze_integration()
    
    print("\n" + "=" * 60)
    print("✅ Integration Complete!")
    print("Architecture successfully replaces spectral Mamba with mHC")
    print("while using spatial Mamba as the branch function.")
    print("=" * 60)
