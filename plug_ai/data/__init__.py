from .data_aug import *
from .datasets import *
from .BraTS import BraTS
from .MedNIST import MedNIST
from .DecathlonT1 import DecathlonT1
from .nnUNet import nnUNet_dataset


supported_datasets = {
    'BraTS' :  BraTS,
    'MedNIST' : MedNIST,
    'DecathlonT1' : DecathlonT1,
    'TciaDataset' : None,
    'nnU-Net' : nnUNet_dataset,
}
