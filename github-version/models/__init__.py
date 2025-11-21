from .AICNet import SSC_RGBD_AICNet
from .DDRNet import SSC_RGBD_DDRNet


def make_model(modelname, num_classes):
    if modelname == 'AICNet':
        return SSC_RGBD_AICNet()
    if modelname == 'DDRNet':
        return SSC_RGBD_DDRNet

__all__ = ["make_model"]
