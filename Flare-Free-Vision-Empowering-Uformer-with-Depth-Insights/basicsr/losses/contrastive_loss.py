import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY
@LOSS_REGISTRY.register()
class ContrastiveLoss(nn.Module):
    def __init__(self, loss_weight=1.0, temperature=0.07, margin=1.0, feature_dim=128, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.margin = margin
        self.feature_dim = feature_dim
    
    def forward(self, pred, target, *args, **kwargs):
        """
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
            *args: 额外的位置参数（被忽略）
            **kwargs: 额外的关键字参数（被忽略）
        """
        device = pred.device
        dtype = pred.dtype
        
        try:
            # 简单的对比损失实现
            pred_flat = pred.view(pred.shape[0], -1)
            target_flat = target.view(target.shape[0], -1)
            
            # 计算余弦相似度
            pred_norm = F.normalize(pred_flat, p=2, dim=1)
            target_norm = F.normalize(target_flat, p=2, dim=1)
            
            # 正样本相似度
            pos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1)
            
            # 创建负样本（随机打乱batch）
            batch_size = pred.shape[0]
            if batch_size > 1:
                indices = torch.randperm(batch_size, device=device)
                neg_target_norm = target_norm[indices]
                neg_sim = F.cosine_similarity(pred_norm, neg_target_norm, dim=1)
            else:
                neg_sim = torch.zeros_like(pos_sim)
            
            # 对比损失
            loss = -pos_sim.mean() + F.relu(neg_sim + self.margin).mean()
            
            # 确保返回有效张量
            if torch.isnan(loss) or torch.isinf(loss):
                loss = torch.tensor(0.0, device=device, dtype=dtype)
            
            result = loss * self.loss_weight
            
        except Exception as e:
            print(f"ContrastiveLoss error: {e}")
            result = torch.tensor(0.0, device=device, dtype=dtype)
        
        # 最终确保是标量张量
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(float(result), device=device, dtype=dtype)
        elif result.dim() > 0:
            result = result.mean()
        
        return result