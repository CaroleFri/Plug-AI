import os
from monai.apps.datasets import MedNISTDataset
from monai.apps.utils import download_and_extract
from monai.transforms import Compose
from .data_aug import available_transforms
from monai.data import Dataset

import torch

class MedNIST:
    url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz" 
    md5 = "0bc7306e7427e00ad1c5526a6677552d"
        
    def __init__(self, dataset_dir, download_dataset=False, limit_sample=None, transformation=None, mode="TRAINING",nb_class=6, progress = True):
        self.dataset_dir = dataset_dir
        self.download_dataset = download_dataset
        self.limit_sample = limit_sample
        self.transformation = transformation
        self.mode = mode
        self.nb_class = nb_class

        if self.download_dataset:
            self.download(root_dir = os.path.dirname(self.dataset_dir), progress=progress)
        

        self.dataset = self.get_dataset(self.dataset_dir, self.limit_sample, self.transformation, self.mode)
    
    def download(self, root_dir, progress):
        compressed_file_name = "MedNIST.tar.gz"
        tarfile_name = root_dir / compressed_file_name
        
        download_and_extract(
            url=self.url,
            filepath=tarfile_name,
            output_dir=root_dir,
            hash_val=self.md5,
            hash_type="md5",
            progress=progress, #Either make it match to verbose, or extra dataset_kwargs parameter
        )
        os.remove(tarfile_name)
        #shutil.rmtree(os.path.join(data_dir,'raw',task))
        # Remove archive after
        

    def get_datalist(self,dataset_dir):        
        class_names = sorted(x for x in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, x)))
        num_class = len(class_names)

        datalist = []

        for i, class_name in enumerate(class_names):
            class_dir = os.path.join(dataset_dir, class_name)
            for image_file in os.listdir(class_dir):
                file_dict = {
                    "input": os.path.join(class_dir, image_file),
                    "label": i
                }
                datalist.append(file_dict)

        print("got datalist, extract: \n", datalist[0])
        
        return datalist



    def get_dataset(self, dataset_dir, limit_sample=None, transformation = "Default", mode="Training"):#, transforma = transforms_BraTS()        
        print("loading dataset...")
        datalist = self.get_datalist(dataset_dir)
        
        
        if mode in ["TRAINING","EVALUATION"]:
            keys = list(datalist[0].keys())
        else:
            keys = list(datalist[0].keys())[:-1]
        print("keys:", keys)
        
        if limit_sample: 
            image_files_list = image_files_list[:limit_sample]
            image_class = image_class[:limit_sample]
            
        if isinstance(transformation, Compose):
            transform = transformation
        elif transformation in available_transforms:
            if mode in ["TRAINING","EVALUATION"]:
                transform = available_transforms[transformation](keys, nb_class=self.nb_class).train
            else:
                transform = available_transforms[transformation](keys, nb_class=self.nb_class).infer
        else:
            transform = None
                
                
        dataset = Dataset( # A more optimized dataset can be used
            data=datalist,
            transform=transform
        )
        return dataset
    
