import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConditionalEmbedding(nn.Module):
    """Эмбеддинг для условий (scale_type + distortion_type)"""
    def __init__(self, num_scale_types: int = 2, num_distortion_types: int = 7, max_offset: int = 200, embedding_dim: int = 128):
        super().__init__()
        # Делим embedding_dim на 3 части
        part = embedding_dim // 3
        self.scale_embedding = nn.Embedding(num_scale_types, part)
        self.distortion_embedding = nn.Embedding(num_distortion_types, part)
        self.offset_embedding = nn.Embedding(max_offset, embedding_dim - 2 * part)
        
    def forward(self, scale_conditions: torch.Tensor, distortion_conditions: torch.Tensor, offset_conditions: torch.Tensor) -> torch.Tensor:
        scale_emb = self.scale_embedding(scale_conditions)
        distortion_emb = self.distortion_embedding(distortion_conditions)
        offset_emb = self.offset_embedding(offset_conditions)
        return torch.cat([scale_emb, distortion_emb, offset_emb], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, cond_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.cond_mlp = nn.Linear(cond_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        
        # Добавляем эмбеддинги времени и условий
        time_emb = self.time_mlp(F.silu(time_emb))
        cond_emb = self.cond_mlp(F.silu(cond_emb))
        emb = time_emb + cond_emb
        
        while len(emb.shape) < len(h.shape):
            emb = emb.unsqueeze(-1)
            
        h = h + emb
        h = self.block2(h)
        
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.group_norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        h = self.group_norm(x)
        h = h.view(batch_size, channels, -1).transpose(1, 2)  # [B, H*W, C]
        
        h, _ = self.attention(h, h, h)
        
        h = h.transpose(1, 2).view(batch_size, channels, height, width)
        return x + h

class UNet(nn.Module):
    def __init__(self, 
                 image_channels: int = 1,
                 base_channels: int = 64,
                 channel_mults: list = [1, 2, 4, 8],
                 num_res_blocks: int = 2,
                 attention_resolutions: list = [],
                 dropout: float = 0.1,
                 time_emb_dim: int = 256,
                 cond_emb_dim: int = 128):
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        self.cond_emb_dim = cond_emb_dim
        
        # Эмбеддинги
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Входной слой
        self.input_conv = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        channels = [base_channels] + [base_channels * mult for mult in channel_mults]
        
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            
            # Residual blocks
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(in_ch, out_ch, time_emb_dim, cond_emb_dim, dropout))
                in_ch = out_ch
                
                # Добавляем attention если нужно
                if i in attention_resolutions:
                    blocks.append(AttentionBlock(out_ch))
            
            self.encoder.append(blocks)
            self.downsample.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(channels[-1], channels[-1], time_emb_dim, cond_emb_dim, dropout),
            AttentionBlock(channels[-1]),
            ResidualBlock(channels[-1], channels[-1], time_emb_dim, cond_emb_dim, dropout)
        )
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for i in reversed(range(len(channels) - 1)):
            in_ch = channels[i + 1]   # Каналы от bottleneck или предыдущего уровня
            out_ch = channels[i]      # Целевые каналы на этом уровне
            skip_ch = in_ch

            # Upsample
            self.upsample.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, 3, padding=1)
            ))

            blocks = nn.ModuleList()

            # Первый блок
            blocks.append(ResidualBlock(out_ch + skip_ch, out_ch, time_emb_dim, cond_emb_dim, dropout))
            if i in attention_resolutions:
                blocks.append(AttentionBlock(out_ch))

            # Остальные блоки
            for _ in range(num_res_blocks - 1):
                blocks.append(ResidualBlock(out_ch, out_ch, time_emb_dim, cond_emb_dim, dropout))
                if i in attention_resolutions:
                    blocks.append(AttentionBlock(out_ch))

            self.decoder.append(blocks)
        
        # Output
        self.output = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, image_channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, time: torch.Tensor, 
            scale_conditions: torch.Tensor, 
            distortion_conditions: torch.Tensor,
            offset_conditions: torch.Tensor,
            cond_embedding: nn.Module) -> torch.Tensor:
        
        # Эмбеддинги
        time_emb = self.time_embed(time)
        cond_emb = cond_embedding(scale_conditions, distortion_conditions, offset_conditions)
        
        # Encoder
        h = self.input_conv(x)
        skips = [h]
        
        for blocks, downsample in zip(self.encoder, self.downsample):
            for block in blocks:
                if isinstance(block, ResidualBlock):
                    h = block(h, time_emb, cond_emb)
                else:
                    h = block(h)
            skips.append(h)
            h = downsample(h)
        
        # Bottleneck
        h = self.bottleneck[0](h, time_emb, cond_emb)
        h = self.bottleneck[1](h)
        h = self.bottleneck[2](h, time_emb, cond_emb)
        
        # Decoder
        for upsample, blocks in zip(self.upsample, self.decoder):
            h = upsample(h)
            skip = skips.pop()
            # Выравниваем пространственные размеры
            if h.shape[2] != skip.shape[2] or h.shape[3] != skip.shape[3]:
                h = F.interpolate(h, size=(skip.shape[2], skip.shape[3]), mode="nearest")
            h = torch.cat([h, skip], dim=1)
            
            for block in blocks:
                if isinstance(block, ResidualBlock):
                    h = block(h, time_emb, cond_emb)
                else:
                    h = block(h)
        
        return self.output(h)

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.unet = UNet(
            image_channels=config["data"]["num_channels"],
            base_channels=config["model"]["base_channels"],
            channel_mults=config["model"]["channel_mults"],
            num_res_blocks=config["model"]["num_res_blocks"],
            attention_resolutions=config["model"]["attention_resolutions"],
            dropout=config["model"]["dropout"]
        )
        
        self.cond_embedding = ConditionalEmbedding(
            num_scale_types=2,  
            num_distortion_types=7, 
            max_offset=config["data"]["image_width"]
        )
        
    def forward(self, x, t, scale_cond, distortion_cond, offset_cond) -> torch.Tensor:
        return self.unet(x, t, scale_cond, distortion_cond, offset_cond, self.cond_embedding)
    