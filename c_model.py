from monai.networks.nets import DynUNet


def get_model(in_channels, n_class, kernels, strides):

    model = DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=n_class,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance", # you can used fused kernel for normal layer when set to `INSTANCE_NVFUSER`
        deep_supervision=True,
        deep_supr_num=3,
    )

    return model
