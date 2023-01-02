from .DynUNet import PlugDynUNet
from monai.networks.nets import UNETR
from .nnUNet import nnUNet_model

# Dict of available model on PlugAI
supported_models = {
    "DynUnet" : PlugDynUNet,
    "unetr" : UNETR,
    "nnU-Net" : nnUNet_model,
}
