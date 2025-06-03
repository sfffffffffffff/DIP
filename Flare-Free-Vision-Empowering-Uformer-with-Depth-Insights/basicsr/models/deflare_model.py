#deflare_model
from collections import OrderedDict
from os import path as osp
import numpy as np
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.flare_util import predict_flare_from_6_channel,predict_flare_from_3_channel
from basicsr.metrics import calculate_metric
import torch
from tqdm import tqdm
from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
import cv2
from torchvision.transforms import Compose
from torch import Tensor
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

class L2Norm(nn.Module):
    """L2 标准化层"""
    def __init__(self, dim=1):
        super(L2Norm, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)
@MODEL_REGISTRY.register()
class DeflareModel(SRModel):

    def init_training_settings(self):
        self.accumulated_iter = 2 # simulate desired batch size
        self.currently_validating = False
        self.net_g.train()
        train_opt = self.opt['train']
        self.output_ch=self.opt['network_g']['output_ch']
        if 'multi_stage' in self.opt['network_g']:
            self.multi_stage=self.opt['network_g']['multi_stage']
        else:
            self.multi_stage=1
        print("Output channel is:", self.output_ch)
        print("Network contains",self.multi_stage,"stages.")

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses - 支持新旧两种配置方式
        if 'losses' in train_opt:
            # 新的配置方式（支持对比学习）
            losses_config = train_opt['losses']
            
            # L1损失
            if 'l1_opt' in losses_config:
                self.l1_pix = build_loss(losses_config['l1_opt']).to(self.device)
            elif 'l1_opt' in train_opt:  # 向后兼容
                self.l1_pix = build_loss(train_opt['l1_opt']).to(self.device)
            
            # 感知损失
            if 'perceptual' in losses_config:
                self.l_perceptual = build_loss(losses_config['perceptual']).to(self.device)
            elif 'perceptual' in train_opt:  # 向后兼容
                self.l_perceptual = build_loss(train_opt['perceptual']).to(self.device)
            
            # 对比学习损失
            self.use_contrastive = 'contrastive_loss' in losses_config
            if self.use_contrastive:
                self.l_contrastive = build_loss(losses_config['contrastive_loss']).to(self.device)
                logger = get_root_logger()
                logger.info('Using contrastive learning loss')
                
            # 傅里叶感知损失
            if 'fourier_loss' in losses_config:
                self.l_fourier = build_loss(losses_config['fourier_loss']).to(self.device)
                self.use_fourier_loss = True
            else:
                self.use_fourier_loss = False
            if 'light_source_guided' in losses_config:
                self.l_light_source = build_loss(losses_config['light_source_guided']).to(self.device)
                print("Using light source guided loss for unsupervised training")

            # 全局一致性损失
            if 'global_consistency' in losses_config:
                self.l_global = build_loss(losses_config['global_consistency']).to(self.device)
                self.use_global_loss = True
            else:
                self.use_global_loss = False
            
            # 深度感知损失
            if 'depth_aware' in losses_config:
                self.l_depth_aware = build_loss(losses_config['depth_aware']).to(self.device)
                self.use_depth_aware = True
            else:
                self.use_depth_aware = False
            
            # 多尺度傅里叶损失
            if 'multiscale_fourier' in losses_config:
                self.l_multiscale = build_loss(losses_config['multiscale_fourier']).to(self.device)
                self.use_multiscale = True
            else:
                self.use_multiscale = False
        else:
            # 旧的配置方式（向后兼容）
            self.l1_pix = build_loss(train_opt['l1_opt']).to(self.device)
            self.l_perceptual = build_loss(train_opt['perceptual']).to(self.device)
            self.use_contrastive = False

        # Define depth model
        net_w, net_h = 384, 384
        self.depth_model = DPTDepthModel(
            path="DPT/dpt_hybrid-midas-501f0c75.pt",
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        self.depth_normalize = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.depth_transform = Compose(
                [
                    Resize(
                        net_w,
                        net_h,
                        resize_target=None,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="minimal",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    self.depth_normalize,
                    PrepareForNet(),
                ]
            )

        self.depth_model.eval()
        self.depth_model.to(self.device)

        # 初始化特征提取器（用于对比学习）
        if self.use_contrastive:
            self._init_feature_extractor()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def _init_feature_extractor(self):
        """初始化特征提取器用于对比学习"""
        # 根据输出通道数确定特征维度
        if self.output_ch == 6:
            feature_input_dim = 6
        else:
            feature_input_dim = 3
        
        # 创建特征提取器
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 4 * feature_input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            L2Norm(dim=1)  # L2标准化
        ).to(self.device)

    def _extract_features(self, x):
        """从网络输出中提取特征用于对比学习"""
        if hasattr(self, 'feature_extractor'):
            return self.feature_extractor(x)
        else:
            # 简单的全局平均池化作为后备方案
            batch_size = x.shape[0]
            features = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            features = features.view(batch_size, -1)
            return torch.nn.functional.normalize(features, dim=1)

    def feed_data(self, data):
        self.gt = data['gt'].to(self.device)
        self.input_for_loss = data['lq'].to(self.device)
        data['lq'] = Tensor(data['lq']).cpu()

        if 'flare' in data:
            self.flare = data['flare'].to(self.device)
            self.gamma = data['gamma'].to(self.device)
        if 'mask' in data:
            self.mask = data['mask'].to(self.device)

        img = np.uint8(data['lq'].permute(0, 2, 3, 1).numpy() * 255)
        depth_batch = self.predict_depth_dpt(img)  # [B, 512, 512]
        depth_batch = depth_batch[:, np.newaxis, :, :]  # [B, 1, 512, 512]

        lq_np = data['lq'].cpu().numpy()
        concat_input = np.concatenate((lq_np, depth_batch), axis=1)  # [B, C+1, H, W]

        self.lq = torch.from_numpy(concat_input).float().to(self.device)

    def predict_depth_dpt(self, img):
        img_input = np.zeros((img.shape[0], 3, 384, 384), dtype=np.float32)
        for i in range(img.shape[0]):
            img_input[i] = self.depth_transform({"image": img[i]})["image"]

        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device)
            prediction = self.depth_model.forward(sample)  # [B, H, W]
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),  # [B, 1, H, W]
                size=(512, 512),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)  # [B, 512, 512]

        prediction = prediction.cpu().numpy()

        # Normalize and clip
        prediction = np.clip(prediction, 0, 3000) / 3000.0  # normalize to [0, 1]
        return prediction  # [B, 512, 512]

    def optimize_parameters(self, current_iter):
        
        # 前向传播
        self.output = self.net_g(self.lq)
        
        # 预测去光晕结果
        if self.output_ch==6:
            self.deflare,self.flare_hat,self.merge_hat=predict_flare_from_6_channel(self.output,self.gamma)
        elif self.output_ch==3:
            self.mask=torch.zeros_like(self.lq).cuda() # Comment this line if you want to use the mask
            self.deflare,self.flare_hat=predict_flare_from_3_channel(self.output,self.mask,self.lq,self.flare,self.lq,self.gamma)        
        else:
            assert False, "Error! Output channel should be defined as 3 or 6."
        
        # 如果使用对比学习，提取特征
        if self.use_contrastive:
            # 提取当前预测的特征
            self.pred_features = self._extract_features(self.output)
            
            # 获取干净图像和光晕图像的特征
            with torch.no_grad():
                # 构造干净图像的输入（GT + depth）
                gt_depth = self.lq[:, 3:, :, :]  # 提取深度信息
                gt_input = torch.cat([self.gt, gt_depth], dim=1)
                
                # 获取干净图像的网络输出和特征
                gt_output = self.net_g(gt_input)
                self.clean_features = self._extract_features(gt_output)
                
                # 获取光晕图像的特征
                self.flare_features = self._extract_features(self.output.detach())

        # 在初始化损失字典时添加新项
        if not hasattr(self, 'loss_dict') or (current_iter) % self.accumulated_iter == 0:
            self.loss_dict = OrderedDict()
            # 现有损失项
            self.loss_dict['l1_recons'] = 0
            self.loss_dict['l1_flare'] = 0 
            self.loss_dict['l1_base'] = 0
            self.loss_dict['l1'] = 0
            self.loss_dict['l_vgg'] = 0
            self.loss_dict['l_vgg_flare'] = 0
            self.loss_dict['l_vgg_base'] = 0
            self.loss_dict['l_total'] = 0
            
            # 新增损失项
            self.loss_dict['l_fourier'] = 0
            self.loss_dict['l_global'] = 0
            self.loss_dict['l_depth'] = 0
            self.loss_dict['l_multiscale'] = 0
            
            if self.use_contrastive:
                self.loss_dict['l_contrastive'] = 0

        # =============== 重要：先初始化总损失 ===============
        l_total = 0
        
        # 提取深度信息（从输入中）
        depth_map = self.lq[:, 3:4, :, :] if self.lq.shape[1] > 3 else None
        
        # =============== 基础损失计算 ===============
        
        # L1 loss
        l1_flare = self.l1_pix(self.flare_hat, self.flare) / self.accumulated_iter
        l1_base = self.l1_pix(self.deflare, self.gt) / self.accumulated_iter
        l1 = l1_flare + l1_base
        
        if self.output_ch == 6:
            l1_recons = 2 * self.l1_pix(self.merge_hat, self.input_for_loss) / self.accumulated_iter
            self.loss_dict['l1_recons'] += l1_recons
            l1 += l1_recons
        
        l_total += l1
        self.loss_dict['l1_flare'] += l1_flare
        self.loss_dict['l1_base'] += l1_base
        self.loss_dict['l1'] += l1

        # Perceptual loss
        l_vgg_flare = self.l_perceptual(self.flare_hat, self.flare) / self.accumulated_iter
        l_vgg_base = self.l_perceptual(self.deflare, self.gt) / self.accumulated_iter
        l_vgg = l_vgg_base + l_vgg_flare
        l_total += l_vgg
        self.loss_dict['l_vgg'] += l_vgg
        self.loss_dict['l_vgg_base'] += l_vgg_base
        self.loss_dict['l_vgg_flare'] += l_vgg_flare
        
        # =============== 高级损失函数计算 ===============
        
        # 傅里叶感知损失
        if hasattr(self, 'use_fourier_loss') and self.use_fourier_loss:
            l_fourier_base = self.l_fourier(self.deflare, self.gt) / self.accumulated_iter
            l_fourier_flare = self.l_fourier(self.flare_hat, self.flare) / self.accumulated_iter
            l_fourier = l_fourier_base + l_fourier_flare
            l_total += l_fourier
            self.loss_dict['l_fourier'] += l_fourier
        
        # 全局一致性损失
        if hasattr(self, 'use_global_loss') and self.use_global_loss:
            l_global = self.l_global(self.deflare, self.gt) / self.accumulated_iter
            l_total += l_global
            self.loss_dict['l_global'] += l_global
        
        # 深度感知损失
        if hasattr(self, 'use_depth_aware') and self.use_depth_aware and depth_map is not None:
            l_depth_base = self.l_depth_aware(self.deflare, self.gt, depth_map) / self.accumulated_iter
            l_depth_flare = self.l_depth_aware(self.flare_hat, self.flare, depth_map) / self.accumulated_iter
            l_depth = l_depth_base + l_depth_flare
            l_total += l_depth
            self.loss_dict['l_depth'] += l_depth
        if hasattr(self, 'l_light_source'):
        # 注意：这里 target 是带光晕的原图，pred 是去光晕后的图
            self.loss_dict['l_light_source'] = self.l_light_source(self.output, self.lq)
            
            # 类型检查（你之前添加的修复）
            for key, value in self.loss_dict.items():
                if not isinstance(value, torch.Tensor):
                    print(f"Converting {key} from {type(value)} to tensor")
                    self.loss_dict[key] = torch.tensor(float(value), device=self.device)
        # 多尺度傅里叶损失
        if hasattr(self, 'use_multiscale') and self.use_multiscale:
            l_multiscale = self.l_multiscale(self.deflare, self.gt) / self.accumulated_iter
            l_total += l_multiscale
            self.loss_dict['l_multiscale'] += l_multiscale
        
        # 对比学习损失
        if self.use_contrastive:
            try:
                l_contrastive = self.l_contrastive(
                    self.clean_features, 
                    self.flare_features, 
                    self.pred_features
                ) / self.accumulated_iter
                l_total += l_contrastive
                self.loss_dict['l_contrastive'] += l_contrastive
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Contrastive loss calculation failed: {e}')
                # 如果对比学习损失计算失败，继续训练但不添加该损失
                pass
        
        # =============== 反向传播和优化器更新 ===============
        
        self.loss_dict['l_total'] += l_total
        l_total.backward()

        if (current_iter) % self.accumulated_iter == 0:
            self.optimizer_g.step()
            self.optimizer_g.zero_grad()
            self.log_dict = self.reduce_loss_dict(self.loss_dict)
            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
        
        if self.output_ch==6:
            self.deflare,self.flare_hat,self.merge_hat=predict_flare_from_6_channel(self.output,self.gamma)
        elif self.output_ch==3:
            self.mask=torch.zeros_like(self.lq).cuda() # Comment this line if you want to use the mask
            self.deflare,self.flare_hat=predict_flare_from_3_channel(self.output,self.mask,self.gt,self.flare,self.lq,self.gamma)        
        else:
            assert False, "Error! Output channel should be defined as 3 or 6."
        
        if not hasattr(self, 'net_g_ema'):
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.currently_validating = True
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.currently_validating = False
        

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.currently_validating = True
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            img_name='deflare_'+str(idx).zfill(5)+'_'
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        self.currently_validating = False

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.input_for_loss.detach().cpu()
        #self.blend= blend_light_source(self.lq[:,:-1,:,:], self.deflare, 0.97)
        out_dict['result']= self.deflare.detach().cpu() # self.blend.detach().cpu()
        out_dict['flare']=self.flare_hat.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict