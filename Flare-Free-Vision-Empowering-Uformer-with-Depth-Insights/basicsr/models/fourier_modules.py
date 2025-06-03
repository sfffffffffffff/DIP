"""
傅里叶增强模块
包含所有与傅里叶变换相关的网络组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class FourierGlobalAttention(nn.Module):
    """
    基于傅里叶变换的全局注意力机制
    解决传统注意力机制局部感受野限制问题
    """
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, 
                 dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 传统空域注意力分支
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # 傅里叶频域分支
        self.fourier_conv = nn.Conv2d(dim, dim, 1, groups=dim)
        self.fourier_norm = nn.LayerNorm(dim)
        
        # 频域权重学习
        self.freq_weight = nn.Parameter(torch.ones(dim, 1, 1))
        
        # 分支融合权重
        self.mix_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
        
    def fourier_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        傅里叶域全局注意力计算
        Args:
            x: 输入特征 [B, H, W, C]
        Returns:
            处理后的特征 [B, H, W, C]
        """
        B, H, W, C = x.shape
        
        # 重排为卷积格式 [B, C, H, W]
        x_conv = x.permute(0, 3, 1, 2)
        
        # FFT变换到频域
        x_fft = torch.fft.rfft2(x_conv, dim=(-2, -1), norm='ortho')
        
        # 频域特征增强
        # 学习频域权重
        freq_enhanced = x_fft * self.freq_weight.unsqueeze(0)
        
        # 频域卷积 (等效全局操作)
        freq_conv = self.fourier_conv(x_fft.real) + 1j * self.fourier_conv(x_fft.imag)
        
        # 结合原始频域特征和卷积结果
        freq_combined = freq_enhanced + 0.1 * freq_conv
        
        # 逆FFT回到空域
        x_ifft = torch.fft.irfft2(freq_combined, s=(H, W), dim=(-2, -1), norm='ortho')
        
        # 重排回注意力格式
        x_fourier = x_ifft.permute(0, 2, 3, 1)
        
        return self.fourier_norm(x_fourier)
    
    def spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        传统空域多头注意力
        Args:
            x: 输入特征 [B, H, W, C]
        Returns:
            注意力处理后的特征 [B, H, W, C]
        """
        B, H, W, C = x.shape
        N = H * W
        
        # 展平空间维度
        x_flat = x.reshape(B, N, C)
        
        # QKV计算
        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0)
        
        # 注意力权重计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 注意力应用
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # 投影和dropout
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        
        return x_attn.reshape(B, H, W, C)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：融合空域和频域注意力
        """
        # 空域注意力
        spatial_out = self.spatial_attention(x)
        
        # 频域注意力
        fourier_out = self.fourier_attention(x)
        
        # 自适 = F.softmax(self.mix_weight, dim=0)
        output = weights[0] * spatial_out + weights[1] * fourier_out
        
        return output

class FrequencyDepthFusion(nn.Module):
    """
    频域感知的RGB-深度信息融合模块
    """
    
    def __init__(self, img_channels: int = 3, depth_channels: int = 1, 
                 out_channels: int = 64, num_freq_bands: int = 3):
        super().__init__()
        
        self.num_freq_bands = num_freq_bands
        
        # 图像和深度特征提取
        self.img_conv = nn.Sequential(
            nn.Conv2d(img_channels, out_channels//2, 3, 1, 1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True)
        )
        
        self.depth_conv = nn.Sequential(
            nn.Conv2d(depth_channels, out_channels//2, 3, 1, 1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # 频域分析器
        self.freq_analyzer = FrequencyAnalyzer(out_channels, num_freq_bands)
        
        # 跨域融合
        self.cross_fusion = CrossDomainFusion(out_channels)
        
        # 输出投影
        self.output_proj = nn.Conv2d(out_channels, out_channels, 1)
        
    def forward(self, image: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        # 特征提取
        img_feat = self.img_conv(image)
        depth_feat = self.depth_conv(depth)
        
        # 特征连接
        combined_feat = torch.cat([img_feat, depth_feat], dim=1)
        
        # 频域分析
        freq_features = self.freq_analyzer(combined_feat)
        
        # 跨域融合
        fused_feat = self.cross_fusion(combined_feat, freq_features)
        
        # 输出投影
        output = self.output_proj(fused_feat)
        
        return output

class FrequencyAnalyzer(nn.Module):
    """
    多频段特征分析器
    """
    
    def __init__(self, channels: int, num_bands: int = 3):
        super().__init__()
        self.channels = channels
        self.num_bands = num_bands
        
        # 每个频段的权重参数
        self.band_weights = nn.Parameter(torch.ones(num_bands))
        
        # 频域特征处理网络
        self.freq_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_bands)
        ])
        
        # 频段融合
        self.band_fusion = nn.Conv2d(channels * num_bands, channels, 1)
        
    def create_frequency_masks(self, shape: Tuple[int, ...], device: torch.device) -> list:
        """
        创建不同频段的掩码
        """
        B, C, H, W = shape
        masks = []
        
        # 创建频率坐标
        freq_h = torch.fft.fftfreq(H, device=device).view(-1, 1)
        freq_w = torch.fft.fftfreq(W//2 + 1, device=device).view(1, -1)
        freq_magnitude = torch.sqrt(freq_h**2 + freq_w**2)
        
        # 定义频段边界
        max_freq = freq_magnitude.max()
        band_boundaries = torch.linspace(0, max_freq, self.num_bands + 1, device=device)
        
        for i in range(self.num_bands):
            mask = torch.zeros_like(freq_magnitude)
            mask[(freq_magnitude >= band_boundaries[i]) & 
                 (freq_magnitude < band_boundaries[i+1])] = 1
            masks.append(mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1))
        
        return masks
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device
        
        # FFT变换
        x_fft = torch.fft.rfft2(x, dim=(-2, -1))
        
        # 创建频段掩码
        freq_masks = self.create_frequency_masks(x_fft.shape, device)
        
        # 分频段处理
        band_features = []
        weights = F.softmax(self.band_weights, dim=0)
        
        for i, (mask, processor) in enumerate(zip(freq_masks, self.freq_processors)):
            # 频段滤波
            x_band_fft = x_fft * mask
            
            # 逆FFT到空域
            x_band = torch.fft.irfft2(x_band_fft, s=(H, W), dim=(-2, -1))
            
            # 特征处理
            x_band_processed = processor(x_band) * weights[i]
            band_features.append(x_band_processed)
        
        # 频段融合
        combined_bands = torch.cat(band_features, dim=1)
        fused_output = self.band_fusion(combined_bands)
        
        return fused_output

class CrossDomainFusion(nn.Module):
    """
    空域-频域跨域特征融合
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        # 空域特征处理分支
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )
        
        # 频域特征处理分支
        self.frequency_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
        # 注意力权重网络
        self.attention_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2, 1),
            nn.Sigmoid()
        )
        
        # 残差连接
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, spatial_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        # 分支处理
        spatial_out = self.spatial_branch(spatial_feat)
        freq_out = self.frequency_branch(freq_feat)
        
        # 注意力权重计算
        concat_feat = torch.cat([spatial_out, freq_out], dim=1)
        attention_weights = self.attention_net(concat_feat)
        
        # 加权融合
        w_spatial = attention_weights[:, 0:1, :, :]
        w_freq = attention_weights[:, 1:2, :, :]
        
        fused = w_spatial * spatial_out + w_freq * freq_out
        
        # 残差连接
        output = fused + self.residual_weight * spatial_feat
        
        return F.relu(output)

class GlobalFourierBottleneck(nn.Module):
    """
    全局傅里叶瓶颈层 - 用于网络的最深层
    """
    
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        
        # 全局上下文提取
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )
        
        # 傅里叶全局注意力
        self.fourier_attn = FourierGlobalAttention(dim, num_heads=8)
        
        # 特征增强
        self.enhancement = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim, 1)
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 全局上下文权重
        global_weight = self.global_context(x)
        x_weighted = x * global_weight
        
        # 傅里叶全局注意力
        x_reshaped = x_weighted.permute(0, 2, 3, 1)  # [B, H, W, C]
        attn_out = self.fourier_attn(self.norm(x_reshaped))
        attn_out = attn_out.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 特征增强
        enhanced = self.enhancement(attn_out)
        
        # 残差连接
        output = x + enhanced
        
        return output
