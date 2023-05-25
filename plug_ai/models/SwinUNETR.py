import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNETR_wrapped(nn.Module):
    """Wrapper for SwinUNETR
    Args:
        img_size:
        in_channels:
        out_channels:
        depths:
        num_heads:
        feature_size:
        norm_name:
        drop_rate:
        attn_drop_rate:
        dropout_path_rate:
        normalize:
        use_checkpoint:
        spatial_dims:
        downsample:
    """
    def __init__(self, 
                img_size,
                 in_channels,
                 out_channels,
                 depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24),
                 feature_size=24,
                 norm_name='instance',
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 dropout_path_rate=0.0,
                 normalize=True,
                 use_checkpoint=False,
                 spatial_dims=3,
                 downsample='merging'):
        
        self.base = SwinUNETR(img_size,
                              in_channels,
                              out_channels,
                              depths=(2, 2, 2, 2),
                              num_heads=(3, 6, 12, 24),
                              feature_size=24,
                              norm_name='instance',
                              drop_rate=0.0,
                              attn_drop_rate=0.0,
                              dropout_path_rate=0.0,
                              normalize=True,
                              use_checkpoint=False,
                              spatial_dims=3,
                              downsample='merging')


    def forward(self, x):
        return self.base(x)