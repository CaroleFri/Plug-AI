from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, SpatialCropd, ConcatItemsd, AsDiscreted, SplitChanneld, SqueezeDimd, ScaleIntensity
from monai.transforms import (
    AddChannel,
    AddChanneld,
    Compose,
    EnsureChannelFirst,
    LoadImage,
    ScaleIntensityd,
    ToTensord,
)
from monai.transforms import Transform, MapTransform
import numpy as np

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

        
        
class transforms_mednist:
    def __init__(self, keys=("input", "label"), nb_class=6):
        self.keys = keys

        self.train = Compose(
            [
                LoadImaged(keys=self.keys[0], image_only=True),
                EnsureChannelFirstd(keys=self.keys[0]),
                ScaleIntensityd(keys=self.keys[0]),
                ToTensord(keys=self.keys),
                AsDiscreted(keys=self.keys[1], to_onehot=nb_class),
            ]
        )

        self.infer = Compose(
            [
                LoadImaged(keys=self.keys[0], image_only=True),
                EnsureChannelFirstd(keys=self.keys[0]),
                ScaleIntensityd(keys=self.keys[0]),
                ToTensord(keys=[self.keys[0]]),
            ]
        )
        
class remove_depth:
    def __init__(self, keys, nb_class=5):

        self.train = Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            ConcatItemsd(keys[:-1], "input"),
            SpatialCropd(keys=['input', 'label'], # crop it to make easily usable for etape 1
                         roi_size=[128, 128, 1],
                         roi_center=[0, 0, 64]
                         ),
            SqueezeDimd(keys=['input', 'label'], dim=-1),
            AsDiscreted(keys=['label'], to_onehot=nb_class)
        ])

        self.infer = Compose([
                        LoadImaged(keys=keys),  # To change when we have a real eval dataset
                        EnsureChannelFirstd(keys=keys),  # To change when we have a real eval dataset
                        ConcatItemsd(keys, "input"),  # To change when we have a real eval dataset
                        SpatialCropd(keys=['input'],  # crop it to make easily usable for etape 1
                                     roi_size=[128, 128, 1],
                                     roi_center=[0, 0, 64]
                                     ),
                        SqueezeDimd(keys=['input'], dim=-1)
                    ])

        
available_transforms = {
    'BraTS_transform': transforms_base,
    'DecathlonT1_transform': transforms_base,
    'MedNIST_transform': transforms_mednist, 
    'remove_depth': remove_depth 
}
