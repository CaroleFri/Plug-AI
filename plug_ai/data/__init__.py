from .data_aug import *
from .BraTS import BraTS
from .MedNIST import MedNIST
from .DecathlonT1 import DecathlonT1
from .nnUNet import nnUNet_dataset
#from .MICCAI import MICCAI2012Dataset


supported_datasets = {
    'BraTS' :  BraTS,
    'MedNIST' : MedNIST,
    'DecathlonT1' : DecathlonT1,
    'TciaDataset' : None,
    'nnU-Net' : nnUNet_dataset,
}
#    'MICCAI' : MICCAI2012Dataset


#from .MICCAI import MICCAI_preprocessing

supported_preprocessing = {
    None : None,
    'None' : None,
}
#    'MICCAI_preprocessing' : MICCAI_preprocessing,
# Removed MICCAI because of OPENCV issues.
