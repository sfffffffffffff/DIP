import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torch import abs_, optim
from PIL import Image
from typing import Mapping, Sequence, Tuple, Union
from torchvision.models import vgg19
import torchvision.models.vgg as vgg
from basicsr.utils.registry import LOSS_REGISTRY

# 修复原有的损失函数
class L_Abs_sideout(nn.Module):
    def __init__(self):
        super(L_Abs_sideout, self).__init__()
        self.resolution_weight = [1., 1., 1., 1.]

    def forward(self, x, flare_gt):
        # [256,256],[128,128],[64,64],[32,32]
        Abs_loss = torch.tensor(0.0, device=x[0].device, dtype=x[0].dtype)
        for i in range(4):
            flare_loss = torch.abs(x[i] - flare_gt[i])
            Abs_loss += torch.mean(flare_loss) * self.resolution_weight[i]
        return Abs_loss

class L_Abs(nn.Module):
    def __init__(self):
        super(L_Abs, self).__init__()

    def forward(self, x, flare_gt, base_gt, mask_gt, merge_gt):
        base_predicted = base_gt * mask_gt + (1 - mask_gt) * x
        flare_predicted = merge_gt - (1 - mask_gt) * x
        base_loss = torch.abs(base_predicted - base_gt)
        flare_loss = torch.abs(flare_predicted - flare_gt)
        Abs_loss = torch.mean(base_loss + flare_loss)
        return Abs_loss




# 傅里叶相关损失函数
class LightSourceDetector(nn.Module):
    """光源检测和定位模块"""
    
    def __init__(self, input_channels=3):
        super().__init__()
        
        self.detector = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((32, 32)),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
        self.param_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)  # [x, y, intensity, radius]
        )
    
    def forward(self, x):
        features = self.detector[:-2](x)
        heatmap = self.detector[-2:](features)
        params = self.param_predictor(features)
        return heatmap, params

class FlarePatternGenerator(nn.Module):
    """基于光源参数生成光晕模式"""
    
    def __init__(self):
        super().__init__()
        
    def generate_flare_pattern(self, light_params, image_size):
        B = light_params.shape[0]
        H, W = image_size
        device = light_params.device
        
        y_coords = torch.arange(H, device=device).float().view(-1, 1)
        x_coords = torch.arange(W, device=device).float().view(1, -1)
        
        flare_patterns = []
        
        for i in range(B):
            x_center, y_center, intensity, radius = light_params[i]
            
            x_center = torch.sigmoid(x_center) * W
            y_center = torch.sigmoid(y_center) * H
            intensity = torch.sigmoid(intensity)
            radius = torch.sigmoid(radius) * min(H, W) / 4
            
            dist = torch.sqrt((x_coords - x_center)**2 + (y_coords - y_center)**2)
            
            # 主光晕
            main_flare = intensity * torch.exp(-dist**2 / (2 * (radius + 1e-8)**2))
            
            # 光线条纹
            angles = torch.atan2(y_coords - y_center, x_coords - x_center)
            streaks = intensity * 0.3 * torch.exp(-dist / (radius + 1e-8)) * \
                     torch.abs(torch.sin(8 * angles))
            
            # 光环
            ring_dist = torch.abs(dist - radius * 1.5)
            rings = intensity * 0.2 * torch.exp(-ring_dist**2 / ((radius * 0.2 + 1e-8)**2))
            
            total_flare = main_flare + streaks + rings
            flare_patterns.append(total_flare)
        
        return torch.stack(flare_patterns, dim=0).unsqueeze(1)

@LOSS_REGISTRY.register()
class FourierAwareLoss(nn.Module):
    """傅里叶感知损失"""
    
    def __init__(self, loss_weight=1.0, num_freq_bands=3, phase_weight=0.1, 
                 high_freq_weight=2.0, low_freq_weight=1.0, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.num_freq_bands = num_freq_bands
        self.phase_weight = phase_weight
        self.high_freq_weight = high_freq_weight
        self.low_freq_weight = low_freq_weight
        
        self.freq_weights = nn.Parameter(torch.ones(num_freq_bands))
        
    def fft_shift(self, x):
        """FFT shift for centering"""
        return torch.fft.fftshift(x, dim=(-2, -1))
    
    def get_frequency_mask(self, shape, device, freq_type='high'):
        """创建频率掩码"""
        B, C, H, W = shape
        
        freq_h = torch.arange(H, device=device) - H // 2
        freq_w = torch.arange(W, device=device) - W // 2
        freq_h = freq_h.view(-1, 1).float()
        freq_w = freq_w.view(1, -1).float()
        
        freq_magnitude = torch.sqrt(freq_h**2 + freq_w**2)
        max_freq = min(H, W) // 2
        
        if freq_type == 'high':
            mask = (freq_magnitude > max_freq * 0.3).float()
        elif freq_type == 'low':
            mask = (freq_magnitude <= max_freq * 0.3).float()
        else:
            mask = ((freq_magnitude > max_freq * 0.1) & 
                   (freq_magnitude <= max_freq * 0.3)).float()
        
        return mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    
    def magnitude_loss(self, pred, target):
        """幅度谱损失"""
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target, dim=(-2, -1))
        
        pred_mag = pred_fft.abs()
        target_mag = target_fft.abs()
        
        return F.l1_loss(pred_mag, target_mag)
    
    def phase_loss(self, pred, target):
        """相位损失"""
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target, dim=(-2, -1))
        
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        phase_diff = pred_phase - target_phase
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        
        return torch.mean(torch.abs(phase_diff))
    
    def frequency_band_loss(self, pred, target):
        """分频段损失"""
        try:
            pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
            target_fft = torch.fft.fft2(target, dim=(-2, -1))
            
            pred_fft_shifted = self.fft_shift(pred_fft)
            target_fft_shifted = self.fft_shift(target_fft)
            
            total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            
            # 低频损失
            low_mask = self.get_frequency_mask(pred.shape, pred.device, 'low')
            pred_low = pred_fft_shifted * low_mask
            target_low = target_fft_shifted * low_mask
            low_loss = F.l1_loss(pred_low.abs(), target_low.abs())
            total_loss += self.low_freq_weight * low_loss
            
            # 高频损失
            high_mask = self.get_frequency_mask(pred.shape, pred.device, 'high')
            pred_high = pred_fft_shifted * high_mask
            target_high = target_fft_shifted * high_mask
            high_loss = F.l1_loss(pred_high.abs(), target_high.abs())
            total_loss += self.high_freq_weight * high_loss
            
            return total_loss
            
        except Exception as e:
            print(f"frequency_band_loss error: {e}")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    def spectral_loss(self, pred, target):
        """光谱损失"""
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        
        pred_power = (pred_fft.abs() ** 2).mean(dim=(2, 3))
        target_power = (target_fft.abs() ** 2).mean(dim=(2, 3))
        
        return F.l1_loss(pred_power, target_power)
    
    def forward(self, pred, target, **kwargs):
        """前向传播"""
        device = pred.device
        dtype = pred.dtype
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        try:
            # 1. 基础幅度损失
            mag_loss = self.magnitude_loss(pred, target)
            total_loss += mag_loss
            
            # 2. 相位损失
            if self.phase_weight > 0:
                ph_loss = self.phase_loss(pred, target)
                total_loss += self.phase_weight * ph_loss
            
            # 3. 分频段损失
            band_loss = self.frequency_band_loss(pred, target)
            total_loss += band_loss
            
            # 4. 光谱损失
            spec_loss = self.spectral_loss(pred, target)
            total_loss += 0.1 * spec_loss
            
        except Exception as e:
            print(f"FourierAwareLoss error: {e}")
            total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        result = total_loss * self.loss_weight
        
        # 确保返回标量张量
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(float(result), device=device, dtype=dtype)
        elif result.dim() > 0:
            result = result.mean()
        
        return result

@LOSS_REGISTRY.register()
class MultiscaleFourierLoss(nn.Module):
    """多尺度傅里叶损失"""
    
    def __init__(self, loss_weight=1.0, scales=[1.0, 0.5, 0.25], 
                 weights=[1.0, 0.8, 0.6], use_phase=True, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.scales = scales
        self.weights = weights
        self.use_phase = use_phase
        
        self.fourier_losses = nn.ModuleList([
            FourierAwareLoss(loss_weight=1.0, phase_weight=0.1 if use_phase else 0.0)
            for _ in scales
        ])
    
    def forward(self, pred, target, **kwargs):
        device = pred.device
        dtype = pred.dtype
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        try:
            for scale, weight, fourier_loss in zip(self.scales, self.weights, self.fourier_losses):
                if scale != 1.0:
                    size = (int(pred.shape[2] * scale), int(pred.shape[3] * scale))
                    pred_scaled = F.interpolate(pred, size=size, mode='bilinear', align_corners=False)
                    target_scaled = F.interpolate(target, size=size, mode='bilinear', align_corners=False)
                else:
                    pred_scaled = pred
                    target_scaled = target
                
                scale_loss = fourier_loss(pred_scaled, target_scaled)
                
                # 确保 scale_loss 是标量张量
                if not isinstance(scale_loss, torch.Tensor):
                    scale_loss = torch.tensor(float(scale_loss), device=device, dtype=dtype)
                elif scale_loss.dim() > 0:
                    scale_loss = scale_loss.mean()
                
                total_loss += weight * scale_loss
                
        except Exception as e:
            print(f"MultiscaleFourierLoss error: {e}")
            total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        result = total_loss * self.loss_weight
        
        # 确保返回标量张量
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(float(result), device=device, dtype=dtype)
        elif result.dim() > 0:
            result = result.mean()
        
        return result

@LOSS_REGISTRY.register()
class GlobalConsistencyLoss(nn.Module):
    """全局一致性损失"""
    
    def __init__(self, loss_weight=1.0, consistency_weight=0.1, num_regions=4, 
                 patch_size=32, stride=16, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.consistency_weight = consistency_weight
        self.num_regions = num_regions
        self.patch_size = patch_size
        self.stride = stride
    
    def forward(self, pred, target, **kwargs):
        device = pred.device
        dtype = pred.dtype
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        try:
            # 基础L1损失
            l1_loss = F.l1_loss(pred, target)
            total_loss += l1_loss
            
            # 梯度一致性损失
            if pred.shape[-1] > 1 and pred.shape[-2] > 1:
                pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
                pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
                target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
                target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
                
                grad_loss = (F.l1_loss(pred_grad_x, target_grad_x) + 
                            F.l1_loss(pred_grad_y, target_grad_y)) / 2
                total_loss += self.consistency_weight * grad_loss
                
        except Exception as e:
            print(f"GlobalConsistencyLoss error: {e}")
            total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        result = total_loss * self.loss_weight
        
        # 确保返回标量张量
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(float(result), device=device, dtype=dtype)
        elif result.dim() > 0:
            result = result.mean()
        
        return result

@LOSS_REGISTRY.register()
class DepthAwareLoss(nn.Module):
    """深度感知损失"""
    
    def __init__(self, loss_weight=1.0, depth_weight=0.5, gradient_weight=0.3, 
                 edge_weight=0.2, use_depth_gradient=True, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.depth_weight = depth_weight
        self.gradient_weight = gradient_weight
        self.edge_weight = edge_weight
        self.use_depth_gradient = use_depth_gradient
        
        # Sobel算子
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                                   dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                                   dtype=torch.float32).view(1, 1, 3, 3))
    
    def forward(self, pred, target, depth=None, **kwargs):
        device = pred.device
        dtype = pred.dtype
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        try:
            # 基础L1损失
            base_loss = F.l1_loss(pred, target)
            total_loss += base_loss
            
            if depth is not None:
                # 确保深度图尺寸匹配
                if depth.shape[-2:] != pred.shape[-2:]:
                    depth = F.interpolate(depth, size=pred.shape[-2:], mode='bilinear', align_corners=False)
                
                # 简化的深度感知损失
                depth_mask = (depth > depth.mean()).float()
                if depth_mask.sum() > 0:
                    masked_pred = pred * depth_mask
                    masked_target = target * depth_mask
                    depth_loss = F.l1_loss(masked_pred, masked_target)
                    total_loss += self.depth_weight * depth_loss
                    
        except Exception as e:
            print(f"DepthAwareLoss error: {e}")
        
        result = total_loss * self.loss_weight
        
        # 确保返回标量张量
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(float(result), device=device, dtype=dtype)
        elif result.dim() > 0:
            result = result.mean()
        
        return result

@LOSS_REGISTRY.register()
class UnsupervisedFlareRemovalLoss(nn.Module):
    """无监督去光晕损失"""
    
    def __init__(self, loss_weight=1.0, light_weight=0.3, physics_weight=0.4, 
                 smooth_weight=0.2, content_weight=0.1, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.light_weight = light_weight
        self.physics_weight = physics_weight
        self.smooth_weight = smooth_weight
        self.content_weight = content_weight
    
    def _light_suppression_loss(self, pred, target):
        """光源抑制损失"""
        try:
            pred_brightness = torch.max(pred, dim=1, keepdim=True)[0]
            target_brightness = torch.max(target, dim=1, keepdim=True)[0]
            
            bright_mask = (target_brightness > 0.8).float()
            if bright_mask.sum() > 0:
                bright_suppression = (pred_brightness * bright_mask).mean()
            else:
                bright_suppression = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            
            brightness_diff = F.relu(pred_brightness.mean() - target_brightness.mean() * 0.9)
            
            return bright_suppression + brightness_diff
            
        except Exception as e:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    def _simple_physics_loss(self, pred, target):
        """物理约束损失"""
        try:
            additive_constraint = F.relu(pred - target).mean()
            range_loss = F.relu(pred - 1.0).mean() + F.relu(-pred).mean()
            return additive_constraint + 0.1 * range_loss
        except Exception as e:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    def _smoothness_loss(self, pred):
        """平滑性损失"""
        try:
            if pred.shape[-1] > 1 and pred.shape[-2] > 1:
                grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
                grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
                return grad_x.mean() + grad_y.mean()
            else:
                return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        except Exception as e:
            return torch.tensor(0.0, device=pred.device,dtype=pred.dtype)
    def _smoothness_loss(self, pred):
        """平滑性损失"""
        try:
            if pred.shape[-1] > 1 and pred.shape[-2] > 1:
                grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
                grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
                return grad_x.mean() + grad_y.mean()
            else:
                return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        except Exception as e:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    def _content_preservation_loss(self, pred, target):
        """内容保持损失"""
        try:
            brightness = torch.max(target, dim=1, keepdim=True)[0]
            non_bright_mask = (brightness < 0.5).float()
            
            if non_bright_mask.sum() > 0:
                masked_pred = pred * non_bright_mask
                masked_target = target * non_bright_mask
                content_loss = F.l1_loss(masked_pred, masked_target)
            else:
                content_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            
            return content_loss
            
        except Exception as e:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    def forward(self, pred, target, **kwargs):
        device = pred.device
        dtype = pred.dtype
        
        # 确保输入匹配
        if pred.shape != target.shape:
            if pred.shape[-2:] != target.shape[-2:]:
                target = F.interpolate(target, size=pred.shape[-2:], mode='bilinear', align_corners=False)
            if pred.shape[1] != target.shape[1]:
                min_channels = min(pred.shape[1], target.shape[1])
                pred = pred[:, :min_channels, :, :]
                target = target[:, :min_channels, :, :]
        
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        try:
            # 1. 光源抑制损失
            light_loss = self._light_suppression_loss(pred, target)
            if isinstance(light_loss, torch.Tensor) and light_loss.dim() > 0:
                light_loss = light_loss.mean()
            total_loss += self.light_weight * light_loss
            
            # 2. 物理约束
            physics_loss = self._simple_physics_loss(pred, target)
            if isinstance(physics_loss, torch.Tensor) and physics_loss.dim() > 0:
                physics_loss = physics_loss.mean()
            total_loss += self.physics_weight * physics_loss
            
            # 3. 平滑性
            smooth_loss = self._smoothness_loss(pred)
            if isinstance(smooth_loss, torch.Tensor) and smooth_loss.dim() > 0:
                smooth_loss = smooth_loss.mean()
            total_loss += self.smooth_weight * smooth_loss
            
            # 4. 内容保持
            content_loss = self._content_preservation_loss(pred, target)
            if isinstance(content_loss, torch.Tensor) and content_loss.dim() > 0:
                content_loss = content_loss.mean()
            total_loss += self.content_weight * content_loss
            
        except Exception as e:
            print(f"UnsupervisedFlareRemovalLoss error: {e}")
            total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        result = total_loss * self.loss_weight
        
        # 最终确保是标量张量
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(float(result), device=device, dtype=dtype)
        elif result.dim() > 0:
            result = result.mean()
            
        return result

@LOSS_REGISTRY.register()
class LightSourceGuidedLoss(nn.Module):
    """光源引导的无监督去光晕损失（完整版）"""
    
    def __init__(self, loss_weight=1.0, use_vgg=True, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.use_vgg = use_vgg
        
        # 光源检测器
        self.light_detector = LightSourceDetector()
        self.flare_generator = FlarePatternGenerator()
        
        # VGG特征提取器
        if use_vgg:
            try:
                vgg_model = vgg19(pretrained=True)
                self.vgg = nn.Sequential(*list(vgg_model.features)[:16])
                for param in self.vgg.parameters():
                    param.requires_grad = False
            except:
                print("Warning: Could not load VGG, using simple features")
                self.vgg = None
                self.use_vgg = False
        else:
            self.vgg = None
    
    def _compute_light_consistency_loss(self, pred, flare_image):
        """光源一致性损失"""
        try:
            pred_heatmap, pred_params = self.light_detector(pred)
            flare_heatmap, flare_params = self.light_detector(flare_image)
            
            # 去光晕图像中的光源应该更弱
            light_reduction_loss = F.relu(pred_heatmap.mean() - 0.1).mean()
            
            # 光源位置一致性
            position_loss = F.mse_loss(pred_params[:, :2], flare_params[:, :2])
            
            # 强度降低
            intensity_loss = F.relu(pred_params[:, 2] - flare_params[:, 2] * 0.5).mean()
            
            return light_reduction_loss + 0.1 * position_loss + 0.1 * intensity_loss
            
        except Exception as e:
            print(f"Light consistency loss error: {e}")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    def _compute_physics_reconstruction_loss(self, pred, flare_image):
        """物理重建损失"""
        try:
            # 检测光源参数
            _, flare_params = self.light_detector(flare_image)
            
            # 生成光晕模式
            flare_pattern = self.flare_generator.generate_flare_pattern(
                flare_params, flare_image.shape[-2:]
            )
            
            # 物理模型：flare_image ≈ pred + flare_pattern
            flare_pattern_rgb = flare_pattern.expand_as(pred)
            reconstructed_flare = torch.clamp(pred + flare_pattern_rgb, 0, 1)
            
            # 重建损失
            reconstruction_loss = F.l1_loss(reconstructed_flare, flare_image)
            
            # 范围约束
            range_loss = F.relu(pred - 1.0).mean() + F.relu(-pred).mean()
            
            return reconstruction_loss + 0.1 * range_loss
            
        except Exception as e:
            print(f"Physics reconstruction loss error: {e}")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    def _compute_perceptual_loss(self, pred, flare_image):
        """感知损失"""
        if not self.use_vgg or self.vgg is None:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        try:
            pred_features = self.vgg(pred)
            flare_features = self.vgg(flare_image)
            
            # 特征清晰度
            feature_clarity = -pred_features.var(dim=(2, 3)).mean()
            
            # 内容保持（排除光晕区域）
            light_heatmap, _ = self.light_detector(flare_image)
            mask = (light_heatmap < 0.3).float()
            
            if mask.sum() > 0:
                mask_expanded = F.interpolate(mask, size=pred.shape[-2:], mode='bilinear')
                masked_pred = pred * mask_expanded
                masked_flare = flare_image * mask_expanded
                content_loss = F.mse_loss(masked_pred, masked_flare)
            else:
                content_loss = torch.tensor(0.0, device=pred.device)
            
            return 0.1 * feature_clarity + content_loss
            
        except Exception as e:
            print(f"Perceptual loss error: {e}")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    def _compute_naturalness_loss(self, pred):
        """自然性约束"""
        try:
            # 梯度平滑
            grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
            grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
            smoothness_loss = grad_x.mean() + grad_y.mean()
            
            # 饱和度约束
            saturation_loss = F.relu(pred.max(dim=1)[0] - 0.95).mean()
            
            # 对比度保持
            mean_pool = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
            contrast_loss = -torch.abs(pred - mean_pool).mean()
            
            return 0.1 * smoothness_loss + 0.1 * saturation_loss + 0.1 * contrast_loss
            
        except Exception as e:
            print(f"Naturalness loss error: {e}")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred: 去光晕后的图像 [B, C, H, W]
            target: 原始带光晕图像 [B, C, H, W]
        """
        device = pred.device
        dtype = pred.dtype
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        try:
            # 1. 光源一致性损失
            light_loss = self._compute_light_consistency_loss(pred, target)
            total_loss += 0.3 * light_loss
            
            # 2. 物理重建损失
            physics_loss = self._compute_physics_reconstruction_loss(pred, target)
            total_loss += 0.4 * physics_loss
            
            # 3. 感知损失
            if self.use_vgg and self.vgg is not None:
                perceptual_loss = self._compute_perceptual_loss(pred, target)
                total_loss += 0.2 * perceptual_loss
            
            # 4. 自然性损失
            naturalness_loss = self._compute_naturalness_loss(pred)
            total_loss += 0.1 * naturalness_loss
            
        except Exception as e:
            print(f"LightSourceGuidedLoss error: {e}")
            total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        result = total_loss * self.loss_weight
        
        # 确保返回标量张量
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(float(result), device=device, dtype=dtype)
        elif result.dim() > 0:
            result = result.mean()
        
        return result

@LOSS_REGISTRY.register()
class FrequencyDomainPerceptualLoss(nn.Module):
    """频域感知损失"""
    
    def __init__(self, loss_weight=1.0, vgg_layers=[2, 7, 12, 21, 30], freq_weight=0.3, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.freq_weight = freq_weight
        
        # 简化实现：直接使用频域L1损失
        self.fourier_loss = FourierAwareLoss(loss_weight=1.0)
    
    def forward(self, pred, target, **kwargs):
        device = pred.device
        dtype = pred.dtype
        
        try:
            # 频域损失
            freq_loss = self.fourier_loss(pred, target)
            
            # 空域L1损失
            spatial_loss = F.l1_loss(pred, target)
            
            # 组合损失
            total_loss = spatial_loss + self.freq_weight * freq_loss
            
        except Exception as e:
            print(f"FrequencyDomainPerceptualLoss error: {e}")
            total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        result = total_loss * self.loss_weight
        
        # 确保返回标量张量
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(float(result), device=device, dtype=dtype)
        elif result.dim() > 0:
            result = result.mean()
        
        return result

@LOSS_REGISTRY.register()
class EdgePreservingLoss(nn.Module):
    """边缘保持损失"""
    
    def __init__(self, loss_weight=1.0, sobel_weight=1.0, laplacian_weight=0.5, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.sobel_weight = sobel_weight
        self.laplacian_weight = laplacian_weight
        
        # Sobel算子
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                                   dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                                   dtype=torch.float32).view(1, 1, 3, 3))
        
        # Laplacian算子
        self.register_buffer('laplacian', torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                                      dtype=torch.float32).view(1, 1, 3, 3))
    
    def get_edges(self, x):
        """提取边缘信息"""
        B, C, H, W = x.shape
        x_gray = x.mean(dim=1, keepdim=True)  # 转为灰度
        
        # Sobel边缘检测
        edges_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        edges_y = F.conv2d(x_gray, self.sobel_y, padding=1)
        edges_sobel = torch.sqrt(edges_x**2 + edges_y**2)
        
        # Laplacian边缘检测
        edges_lap = torch.abs(F.conv2d(x_gray, self.laplacian, padding=1))
        
        return edges_sobel, edges_lap
    
    def forward(self, pred, target, **kwargs):
        device = pred.device
        dtype = pred.dtype
        
        try:
            # 提取边缘
            pred_sobel, pred_lap = self.get_edges(pred)
            target_sobel, target_lap = self.get_edges(target)
            
            # 边缘损失
            sobel_loss = F.l1_loss(pred_sobel, target_sobel)
            lap_loss = F.l1_loss(pred_lap, target_lap)
            
            total_loss = (self.sobel_weight * sobel_loss + 
                         self.laplacian_weight * lap_loss)
            
        except Exception as e:
            print(f"EdgePreservingLoss error: {e}")
            total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        result = total_loss * self.loss_weight
        
        # 确保返回标量张量
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(float(result), device=device, dtype=dtype)
        elif result.dim() > 0:
            result = result.mean()
        
        return result