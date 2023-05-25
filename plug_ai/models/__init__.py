from .DynUNet import PlugDynUNet
from monai.networks.nets import UNETR
from .nnUNet import nnUNet_model
from .ModSegNet import ModSegNet
from monai.networks.nets import DenseNet
from .SwinUNETR import SwinUNETR_wrapped
from monai.networks.nets import SwinUNETR
from monai.networks.nets import SegResNet



# Dict of available model on PlugAI
supported_models = {
    "DynUnet" : PlugDynUNet,
    "unetr" : UNETR,
    "nnU-Net" : nnUNet_model,
    "ModSegNet" : ModSegNet,
    "DenseNet": DenseNet,
    "SwinUNETRw": SwinUNETR_wrapped,
    "SwinUNETR" : SwinUNETR,
    "SegResNet" : SegResNet
}
