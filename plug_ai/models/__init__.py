from .DynUNet import PlugDynUNet
from monai.networks.nets import UNETR
    
# Dict of available model on PlugAI
supported_models = {
    "DynUnet" : PlugDynUNet,
    "unetr" : UNETR
}
