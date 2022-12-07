from .data_aug import *
from .datasets import *
from .BraTS import BraTS
from .MedNIST import MedNIST


supported_datasets = {
    'BraTS' :  BraTS,
    'MedNIST' : MedNIST,
    'Decathlon' : None,
    'TciaDataset' : None,
    'Specific_Dataset_A' : get_dataset_a,
    'Specific_Dataset_B' : get_dataset_b
}
#    'Dataset_Type_A' :  Dataset_Type_A,

"""
transforms = {
    'BraTS_default' : transforms_BraTS
}
"""