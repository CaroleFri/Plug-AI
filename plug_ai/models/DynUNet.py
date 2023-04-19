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
    def __init__(self,
                 spatial_dims = 3,
                 in_channels = 4,
                 out_channels = 5,
                 kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                 strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                 upsample_kernel_size = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                 norm_name = "instance",  # you can use fused kernel for normal layer when set to `INSTANCE_NVFUSER`
                 deep_supervision =  True,
                 deep_supr_num = 3,
                 res_out=False):
        """
        spatial_dims:
        in_channels:
        out_channels:
        kernel_size:
        strides:
        upsample_kernel_size:
        norm_name:
        deep_supervision:
        deep_supr_num:
        res_out:
        """
        super(PlugDynUNet, self).__init__()
        self.res_out = res_out
        # use_signature: A boolean that should indicate if to use or not the signature of the dataset for adaptation
        # res_out: A model specific parameter
        #checkpoints_path: a path to checkpoints for the model

        self.base = DynUNet(spatial_dims = spatial_dims,
                            in_channels = in_channels ,
                            out_channels = out_channels ,
                            kernel_size = kernel_size ,
                            strides = strides,
                            upsample_kernel_size = upsample_kernel_size,
                            norm_name = norm_name,
                            deep_supervision = deep_supervision,
                            deep_supr_num = deep_supr_num)
        
        '''
        if self.use_signature is not None:
            # WIP
            available_signatures = {
                "nnUnet" : None
            }
            if self.use_signature in available_signatures:
                print("using signature to configure model")
            #
        '''    

    def forward(self, inp):
        """
        :param inp:
        :return:
        """
        out = self.base(inp)

        # for now res_out is always false
        if not self.res_out and self.training:
            out = torch.unbind(out, dim=1)
            out = out[0]

        return out
