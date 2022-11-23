import os
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from monai.data import Dataset
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, SpatialCropd, ConcatItemsd









class Dataset_Type_A:            
    def __init__(self, **kwargs):
        self.dataset = 1 # should return a pytorch dataset (init, len, get_item)
        self.dataset_dir = kwargs["dataset_dir]

    def download(self, **kwargs):
        #add download procedure
        return self.dataset_dir
        
    def process(self):
        if self.kwargs["verbose"] == "Full" :
            print("Dataset initialization ...")
        self.kwargs["dataset"] = self.check_dataset(**self.kwargs)
        self.preprocess = self.check_preprocess(self.kwargs["preprocess"], self.kwargs)
    
    def get_signature():
        self.nbr_classes = 0
        self.nbr_features = 0
        self.complexity = 0
        return
        