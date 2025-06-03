import importlib
from copy import deepcopy
from basicsr.utils import get_root_logger
from basicsr.utils.registry import LOSS_REGISTRY

# 直接导入所有损失模块
importlib.import_module('basicsr.losses.basic_loss')
importlib.import_module('basicsr.losses.flare_loss')
importlib.import_module('basicsr.losses.gan_loss')
importlib.import_module('basicsr.losses.contrastive_loss')
importlib.import_module('basicsr.losses.fourier_loss')

print("Registered losses:", list(LOSS_REGISTRY.keys()))

__all__ = ['build_loss']

def build_loss(opt):
    """Build loss from options."""
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    
    print(f"Trying to build loss: {loss_type}")
    print(f"Available losses: {list(LOSS_REGISTRY.keys())}")
    
    if loss_type not in LOSS_REGISTRY:
        available_losses = list(LOSS_REGISTRY.keys())
        raise KeyError(f"Loss type '{loss_type}' not found. Available: {available_losses}")
    
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss