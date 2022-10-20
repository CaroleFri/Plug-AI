from monai.networks.nets import DynUNet
import torch.nn as nn
import torch

dyn_kwargs = {
            "spatial_dims": 3,
            "in_channels": 4,
            "out_channels": 5,
            "kernel_size": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            "upsample_kernel_size": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            "norm_name": "instance",  # you can use fused kernel for normal layer when set to `INSTANCE_NVFUSER`
            "deep_supervision": True,
            "deep_supr_num": 3
}


class PlugDynUNet(nn.Module):
    """
    Plug-AI version of DynUnet
    """
    def __init__(self, model_kwargs=None, res_out=False):
        """

        :param model_kwargs:
        :param res_out:
        """
        super(PlugDynUNet, self).__init__()
        self.res_out = res_out

        if type(model_kwargs) is dict:
            dyn_kwargs.update(self.model_kwargs)

        self.base = DynUNet(
            **dyn_kwargs
        )

    def forward(self, inp):
        """

        :param inp:
        :return:
        """
        out = self.base(inp)

        # for now res_out is always false
        if not self.res_out:
            out = torch.unbind(out, dim=1)
            out = out[0]

        return out
