from .DynUNet import PlugDynUNet
from monai.networks.nets import UNETR
from .nnUNet import nnUNet_model
from .ModSegNet import ModSegNet

# Dict of available model on PlugAI
supported_models = {
    "DynUnet" : PlugDynUNet,
    "unetr" : UNETR,
    "nnU-Net" : nnUNet_model,
    "ModSegNet" : ModSegNet,
}
