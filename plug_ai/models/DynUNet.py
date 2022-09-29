from monai.networks.nets import DynUNet
import torch.nn as nn
import torch


class PlugDynUNet(nn.Module):
    def __init__(self, in_channels, n_class, kernels, strides):
        super(PlugDynUNet, self).__init__()
        self.base = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=n_class,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",  # you can used fused kernel for normal layer when set to `INSTANCE_NVFUSER`
            deep_supervision=True,
            deep_supr_num=3,
        )

    def forward(self, inp):
        out = self.base(inp)
        out = torch.unbind(out, dim=1)

        return out
