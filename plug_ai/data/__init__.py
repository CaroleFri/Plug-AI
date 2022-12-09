from .data_aug import *
from .datasets import *
from .BraTS import BraTS
from .MedNIST import MedNIST


supported_datasets = {
    'BraTS' :  BraTS,
    'MedNIST' : MedNIST,
    'Decathlon' : None,
    'TciaDataset' : None,
}

