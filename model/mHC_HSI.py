import math
import torch
import torch.nn as nn
from einops import rearrange, einsum
from mamba_ssm import Mamba


# ==================== mHC CORE COMPONENTS ====================

def sinkhorn_log(logits, iters=10, tau=0.05):
    """Sinkhorn projection for doubly stochastic matrices."""
    Z = logits / tau
    u = v = torch.zeros_like(Z[..., 0])
    for _ in range(iters):
        u = -torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = -torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)
    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))


class HyperConnections(nn.Module):
    """Manifold-Constrained Hyper-Connections (mHC) layer."""
    
    def __init__(
        self,
        num_streams,
        dim,
        branch: nn.Module,
        layer_index=0,
        dropout=0.0,
        mhc_iters=10,
        mhc_tau=0.05,
    ):
        super().__init__()
        
        self.S = num_streams
        self.dim = dim
        self.branch = branch
        self.dropout = nn.Dropout(dropout)
        self.iters = mhc_iters
        self.tau = mhc_tau
        
        # mHC parameters - aligned with paper Section 4.2
        # H_res: residual mapping (doubly stochastic via Sinkhorn)
        self.H_res_logits = nn.Parameter(
            torch.full((num_streams, num_streams), -10.0)
        )
        with torch.no_grad():
            self.H_res_logits.fill_diagonal_(0.0)  # Diagonal preference
        
        # H_pre: pre-mixing weights (selects streams for branch)
        self.H_pre_logits = nn.Parameter(torch.zeros(1, num_streams))
        with torch.no_grad():
            # Cycle through streams (paper Section 4.1)
            selected_stream = layer_index % num_streams
            self.H_pre_logits[:, selected_stream] = 5.0
        
        # H_post: post-mixing weights
        self.H_post_logits = nn.Parameter(torch.zeros(1, num_streams))
        
        # Learnable scales (γ and β from paper)
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.residual_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, residuals):
        """
        residuals: (batch * streams, dim)
        returns: (batch * streams, dim)
        """
        # Reshape to work with streams
        R = rearrange(residuals, "(b s) d -> b s d", s=self.S)
        
        # 1. Width mixing with Sinkhorn projection (doubly stochastic)
        H_res = sinkhorn_log(self.H_res_logits, self.iters, self.tau)
        Rw = einsum(H_res, R, "s t, b s d -> b t d")
        
        # 2. Pre-mix: select streams for branch computation
        H_pre = self.H_pre_logits.softmax(dim=-1)
        x_selected = einsum(H_pre, R, "1 s, b s d -> b d")
        
        # 3. Branch computation (Mamba in our case)
        y = self.branch(x_selected)
        
        # 4. Post-mix: distribute branch output to streams
        H_post = self.H_post_logits.softmax(dim=-1)
        delta = einsum(y, H_post, "b d, 1 s -> b s d")
        
        # 5. Combine with residual and apply scales (Eq. 8 in paper)
        out = self.gamma * (self.residual_scale * Rw + delta) + self.beta
        
        # 6. Flatten and apply dropout
        out = rearrange(out, "b s d -> (b s) d")
        return self.dropout(out)


# ==================== mHC-ENHANCED MAMBA COMPONENTS ====================

class mHC_SpeMamba(nn.Module):
    """Spectral Mamba with mHC-enhanced residual connections."""
    
    def __init__(self, channels, token_num=8, num_streams=4, 
                 use_residual=True, group_num=4):
        super().__init__()
        self.token_num = token_num
        self.use_residual = use_residual
        self.num_streams = num_streams
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num
        
        # Determine streams per token for reshaping
        assert num_streams % token_num == 0, \
            "num_streams must be divisible by token_num for clean reshaping"
        self.streams_per_token = num_streams // token_num
        
        # 1. mHC layer for spectral feature mixing
        self.mhc_layer = HyperConnections(
            num_streams=num_streams,
            dim=self.group_channel_num,  # Mamba's d_model
            branch=self._create_mamba_branch(),
            dropout=0.1,
            mhc_iters=10,
            mhc_tau=0.05
        )
        
        # 2. Projection after processing
        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU(),
            nn.Dropout2d(0.1)
        )
    
    def _create_mamba_branch(self):
        """Create the core Mamba block as mHC's branch function."""
        return nn.Sequential(
            Mamba(
                d_model=self.group_channel_num,
                d_state=16,
                d_conv=4,
                expand=2,
            ),
            nn.GroupNorm(4, self.group_channel_num),
            nn.SiLU()
        )
    
    def padding_feature(self, x):
        """Pad features to match expected channel dimensions."""
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            return torch.cat([x, pad_features], dim=1)
        return x
    
    def forward(self, x):
        # 1. Pad if necessary
        x_pad = self.padding_feature(x)
        
        # 2. Rearrange: (B, C, H, W) -> (B, H, W, C)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape
        
        # 3. Reshape for token processing
        # (B, H, W, C) -> (B*H*W, token_num, group_channel_num)
        x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        
        # 4. Prepare for mHC: Expand to streams
        # (n, token_num, d) -> (n, num_streams, d)
        x_expanded = rearrange(x_flat, 'n t d -> n (t s) d', 
                             s=self.streams_per_token)
        
        # 5. Apply mHC layer
        residuals_flat = rearrange(x_expanded, 'n s d -> (n s) d')
        out_mhc = self.mhc_layer(residuals_flat)
        out_mhc = rearrange(out_mhc, '(n s) d -> n s d', 
                          s=self.num_streams)
        
        # 6. Recover token structure
        x_mamba_ready = rearrange(out_mhc, 'n (t s) d -> n t d', 
                                t=self.token_num)
        
        # 7. Reshape back to spatial format
        x_recon = x_mamba_ready.view(B, H, W, C_pad)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        
        # 8. Final projection
        x_proj = self.proj(x_recon)
        
        if self.use_residual:
            return x + x_proj
        else:
            return x_proj


class mHC_SpaMamba(nn.Module):
    """Spatial Mamba with mHC-enhanced residual connections."""
    
    def __init__(self, channels, num_streams=4, use_residual=True, 
                 group_num=4, use_proj=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.num_streams = num_streams
        
        # mHC layer for spatial feature mixing
        self.mhc_layer = HyperConnections(
            num_streams=num_streams,
            dim=channels,
            branch=self._create_spatial_mamba_branch(channels),
            dropout=0.1,
            mhc_iters=10,
            mhc_tau=0.05
        )
        
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU(),
                nn.Dropout2d(0.1)
            )
    
    def _create_spatial_mamba_branch(self, channels):
        """Spatial Mamba processes flattened spatial dimensions."""
        return nn.Sequential(
            Mamba(
                d_model=channels,
                d_state=16,
                d_conv=4,
                expand=2,
            ),
            nn.GroupNorm(4, channels),
            nn.SiLU()
        )
    
    def forward(self, x):
        # 1. Rearrange: (B, C, H, W) -> (B, H, W, C)
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_re.shape
        
        # 2. Flatten spatial dimensions
        x_flat = x_re.view(B * H * W, C)  # (B*H*W, C)
        
        # 3. Prepare for mHC: Expand to streams
        residuals = rearrange(x_flat, 'n d -> (n s) d', s=self.num_streams)
        
        # 4. Apply mHC
        out_mhc = self.mhc_layer(residuals)
        out_mhc = rearrange(out_mhc, '(n s) d -> n d', s=self.num_streams)
        
        # 5. Reshape back to spatial
        x_recon = out_mhc.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        
        if self.use_proj:
            x_recon = self.proj(x_recon)
        
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon


class mHC_BothMamba(nn.Module):
    """Dual-branch Mamba with mHC in both spatial and spectral paths."""
    
    def __init__(self, channels, token_num=4, num_streams=4, 
                 use_residual=True, group_num=4, use_att=True):
        super().__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        
        if self.use_att:
            # Learnable fusion weights
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)
        
        # mHC-enhanced spatial and spectral Mamba
        self.spa_mamba = mHC_SpaMamba(
            channels, 
            num_streams=num_streams,
            use_residual=use_residual,
            group_num=group_num
        )
        self.spe_mamba = mHC_SpeMamba(
            channels,
            token_num=token_num,
            num_streams=num_streams,
            use_residual=use_residual,
            group_num=group_num
        )
        
        # Optional channel attention for fusion
        if use_att:
            self.channel_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels * 2, channels // 8, 1),
                nn.ReLU(),
                nn.Conv2d(channels // 8, channels * 2, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)
        
        if self.use_att:
            # Concatenate for channel attention
            concat = torch.cat([spa_x, spe_x], dim=1)
            att_weights = self.channel_att(concat)
            
            # Split attention weights
            spa_att, spe_att = att_weights.chunk(2, dim=1)
            
            # Apply attention-weighted fusion
            fusion_x = spa_x * spa_att + spe_x * spe_att
        else:
            # Simple weighted fusion
            weights = self.softmax(self.weights) if self.use_att else torch.tensor([0.5, 0.5])
            fusion_x = spa_x * weights[0] + spe_x * weights[1]
        
        if self.use_residual:
            return fusion_x + x
        else:
            return fusion_x


# ==================== COMPLETE mHC-MambaHSI MODEL ====================

class mHC_MambaHSI(nn.Module):
    """
    Complete mHC-enhanced MambaHSI model for hyperspectral image segmentation.
    Integrates Manifold-Constrained Hyper-Connections with MambaSSM.
    """
    
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, 
                 num_streams=4, use_residual=True, mamba_type='both',
                 token_num=4, group_num=4, use_att=True):
        super().__init__()
        self.mamba_type = mamba_type
        
        # Patch embedding (spectral projection)
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim,
                     kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
            nn.Dropout2d(0.1)
        )
        
        # Multi-scale mHC-Mamba processing blocks
        if mamba_type == 'spa':
            self.mamba = nn.Sequential(
                mHC_SpaMamba(hidden_dim, num_streams=num_streams, 
                           use_residual=use_residual, group_num=group_num),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Dropout2d(0.1),
                
                mHC_SpaMamba(hidden_dim, num_streams=num_streams,
                           use_residual=use_residual, group_num=group_num),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Dropout2d(0.1),
                
                mHC_SpaMamba(hidden_dim, num_streams=num_streams,
                           use_residual=use_residual, group_num=group_num),
                nn.Dropout2d(0.1),
            )
        elif mamba_type == 'spe':
            self.mamba = nn.Sequential(
                mHC_SpeMamba(hidden_dim, token_num=token_num,
                           num_streams=num_streams, use_residual=use_residual,
                           group_num=group_num),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Dropout2d(0.1),
                
                mHC_SpeMamba(hidden_dim, token_num=token_num,
                           num_streams=num_streams, use_residual=use_residual,
                           group_num=group_num),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Dropout2d(0.1),
                
                mHC_SpeMamba(hidden_dim, token_num=token_num,
                           num_streams=num_streams, use_residual=use_residual,
                           group_num=group_num),
                nn.Dropout2d(0.1),
            )
        else:  # 'both' - default and recommended
            self.mamba = nn.Sequential(
                mHC_BothMamba(channels=hidden_dim, token_num=token_num,
                            num_streams=num_streams, use_residual=use_residual,
                            group_num=group_num, use_att=use_att),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Dropout2d(0.1),
                
                mHC_BothMamba(channels=hidden_dim, token_num=token_num,
                            num_streams=num_streams, use_residual=use_residual,
                            group_num=group_num, use_att=use_att),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Dropout2d(0.1),
                
                mHC_BothMamba(channels=hidden_dim, token_num=token_num,
                            num_streams=num_streams, use_residual=use_residual,
                            group_num=group_num, use_att=use_att),
                nn.Dropout2d(0.1),
            )
        
        # Upsampling to recover spatial resolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
            nn.Dropout2d(0.1),
        )
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(group_num, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1, stride=1, padding=0),
        )
        
        # Optional deep supervision
        self.deep_supervision = False
        
    def forward(self, x):
        # Store original size for upsampling
        orig_size = x.shape[2:]
        
        # 1. Patch embedding
        x = self.patch_embedding(x)
        
        # 2. Multi-scale mHC-Mamba processing
        features = []
        for i, layer in enumerate(self.mamba):
            x = layer(x)
            if i % 2 == 0:  # Save features before pooling
                features.append(x)
        
        # 3. Upsample to original size
        x = self.upsample(x)
        
        # 4. Optional: Add deep supervision features
        if self.deep_supervision and len(features) > 0:
            # Upsample and add features from different scales
            for feat in features[::-1]:  # From deep to shallow
                feat_up = nn.functional.interpolate(
                    feat, size=orig_size, mode='bilinear', align_corners=False
                )
                x = x + feat_up * 0.3  # Weighted addition
        
        # 5. Final classification
        logits = self.cls_head(x)
        
        return logits


# ==================== MODEL CONFIGURATIONS ====================

def mHC_MambaHSI_small(in_channels=128, num_classes=10):
    """Small configuration for quick experimentation."""
    return mHC_MambaHSI(
        in_channels=in_channels,
        hidden_dim=48,
        num_classes=num_classes,
        num_streams=2,
        token_num=4,
        use_att=False
    )

def mHC_MambaHSI_base(in_channels=128, num_classes=10):
    """Base configuration recommended for most HSI datasets."""
    return mHC_MambaHSI(
        in_channels=in_channels,
        hidden_dim=64,
        num_classes=num_classes,
        num_streams=4,
        token_num=4,
        use_att=True
    )

def mHC_MambaHSI_large(in_channels=128, num_classes=10):
    """Large configuration for complex HSI segmentation tasks."""
    return mHC_MambaHSI(
        in_channels=in_channels,
        hidden_dim=96,
        num_classes=num_classes,
        num_streams=8,
        token_num=8,
        use_att=True
    )


# ==================== TRAINING UTILITIES ====================

class HSI_Segmentation_Trainer:
    """Training utilities for mHC-MambaHSI model."""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions for HSI segmentation
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.dice_loss = self.dice_coefficient_loss
        self.focal_loss = self.focal_loss_fn
        
        # Optimizer (AdamW from paper)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.05,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-5
        )
    
    def dice_coefficient_loss(self, pred, target, smooth=1e-6):
        """Dice loss for imbalanced HSI classes."""
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(
            target, num_classes=pred.shape[1]
        ).permute(0, 3, 1, 2).float()
        
        intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def focal_loss_fn(self, pred, target, alpha=0.25, gamma=2.0):
        """Focal loss for hard-to-classify pixels."""
        ce_loss = torch.nn.functional.cross_entropy(
            pred, target, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def compute_loss(self, pred, target, loss_weights=(0.4, 0.3, 0.3)):
        """Combined loss for HSI segmentation."""
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss_fn(pred, target)
        
        return (
            loss_weights[0] * ce +
            loss_weights[1] * dice +
            loss_weights[2] * focal
        )
    
    def train_step(self, batch):
        """Single training step."""
        images, masks = batch
        images = images.to(self.device)
        masks = masks.to(self.device)
        
        self.optimizer.zero_grad()
        pred = self.model(images)
        loss = self.compute_loss(pred, masks)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()


# ==================== TEST AND USAGE ====================

if __name__ == "__main__":
    # Test the complete model
    print("Testing mHC-MambaHSI model...")
    
    # Create model
    model = mHC_MambaHSI_base(in_channels=128, num_classes=10)
    
    # Test with random HSI data
    batch_size = 2
    height, width = 64, 64
    channels = 128
    num_classes = 10
    
    # Simulate HSI cube
    x = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test different configurations
    print("\nTesting different configurations:")
    
    small_model = mHC_MambaHSI_small(in_channels=128, num_classes=10)
    large_model = mHC_MambaHSI_large(in_channels=128, num_classes=10)
    
    print(f"Small model params: {sum(p.numel() for p in small_model.parameters()):,}")
    print(f"Large model params: {sum(p.numel() for p in large_model.parameters()):,}")
    
    # Test spectral-only and spatial-only variants
    model_spe = mHC_MambaHSI(
        in_channels=128,
        hidden_dim=64,
        num_classes=10,
        num_streams=4,
        mamba_type='spe'
    )
    model_spa = mHC_MambaHSI(
        in_channels=128,
        hidden_dim=64,
        num_classes=10,
        num_streams=4,
        mamba_type='spa'
    )
    
    print(f"\nSpectral-only params: {sum(p.numel() for p in model_spe.parameters()):,}")
    print(f"Spatial-only params: {sum(p.numel() for p in model_spa.parameters()):,}")
    
    print("\n✅ Model implementation complete and tested!")
