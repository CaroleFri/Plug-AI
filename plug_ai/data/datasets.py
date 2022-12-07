import os
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from monai.data import Dataset
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, SpatialCropd, ConcatItemsd








def get_dataset_a(a=1, b=2):
    print("this is dataset a")
    dataset = [a,b]
    return dataset
    
get_dataset_b= "b_dataset"
get_dataset_c= "c_dataset"