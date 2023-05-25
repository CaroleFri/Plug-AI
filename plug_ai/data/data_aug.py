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
from monai import transforms


from monai.transforms import(
    Compose,
    LoadImaged,
    EnsureTyped,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CropForegroundd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Orientationd,
    Spacingd
)



class transforms_Brats:
    def __init__(self, keys):
        roi = (224, 224, 224)
        self.train = Compose([
            LoadImaged(keys=['channel_0', 'channel_1', 'channel_2', 'channel_3', 'label']),
            EnsureChannelFirstd(keys = ['channel_0', 'channel_1', 'channel_2', 'channel_3']),
            ConcatItemsd(keys=['channel_0', 'channel_1', 'channel_2', 'channel_3'], name="input", dim=0),
            EnsureTyped(keys=["input", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["input", "label"], axcodes="RAS"),
            Spacingd(
                keys=["input", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=["input", "label"], roi_size=[224, 224, 144], random_size=False),            
            RandFlipd(keys=["input", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["input", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["input", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="input", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="input", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="input", offsets=0.1, prob=1.0),
        ])

        self.infer = Compose([
            LoadImaged(keys=['channel_0', 'channel_1', 'channel_2', 'channel_3']),
            EnsureChannelFirstd(keys=['channel_0', 'channel_1', 'channel_2', 'channel_3']),
            ConcatItemsd(keys=['channel_0', 'channel_1', 'channel_2', 'channel_3'], name="input", dim=0),
            EnsureTyped(keys=["input"]),
            Orientationd(keys=["input"], axcodes="RAS"),
            Spacingd(
                keys=["input"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear"),
            ),            
            NormalizeIntensityd(keys="input", nonzero=True, channel_wise=True),
        ])



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
    'BraTS_transform': transforms_Brats,
    'DecathlonT1_transform': transforms_base,
    'MedNIST_transform': transforms_mednist, 
    'remove_depth': remove_depth 
}
