from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, SpatialCropd, ConcatItemsd, AsDiscreted


class transforms_base:
    def __init__(self, keys, nb_class=5):

        self.train = Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            ConcatItemsd(keys[:-1], "input"),
            SpatialCropd(keys=['input', 'label'], # crop it to make easily usable for etape 1
                         roi_size=[128, 128, 128],
                         roi_center=[0, 0, 0]
                         ),
            AsDiscreted(keys=['label'], to_onehot=nb_class)
        ])

        self.infer = Compose([
                        LoadImaged(keys=keys),  # To change when we have a real eval dataset
                        EnsureChannelFirstd(keys=keys),  # To change when we have a real eval dataset
                        ConcatItemsd(keys, "input"),  # To change when we have a real eval dataset
                        SpatialCropd(keys=['input'],  # crop it to make easily usable for etape 1
                                     roi_size=[128, 128, 128],
                                     roi_center=[0, 0, 0]
                                     )
                    ])


available_transforms = {
    'BraTS_transform': transforms_base,
    'DecathlonT1_transform': transforms_base
}
