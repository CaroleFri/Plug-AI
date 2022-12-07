import os
from monai.apps.datasets import MedNISTDataset
from monai.apps.utils import download_and_extract

class MedNIST:
    url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz" 
    md5 = "0bc7306e7427e00ad1c5526a6677552d"
        
    def __init__(self, dataset_dir, download_dataset=False, limit_sample=None, transformation=None, mode="Training", progress = True):
        self.dataset_dir = dataset_dir
        self.download = download
        self.limit_sample = limit_sample
        self.transformation = transformation
        self.mode = mode
        
        if download_dataset:
            self.download(root_dir = os.path.dirname(self.dataset_dir))
        

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
        datalist = []
        with open(os.path.join(dataset_dir, "train.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                file_dic = {}
                files = line.split()
                for i, file in enumerate(files[:-1]):
                    file_dic[f"channel_{i}"] = os.path.join(dataset_dir, file)

                file_dic["label"] = os.path.join(dataset_dir, files[-1])
                datalist.append(file_dic)

        print("got datalist, extract: \n", datalist[0])
        return datalist


    def get_dataset(self, dataset_dir, limit_sample=None, transformation = "Default", mode="Training"):#, transforma = transforms_BraTS()
        print("loading dataset...")
        datalist = self.get_datalist(dataset_dir)
        # Modified transformation so that the loader just takes the keys. Up to the file generator to be format things correctly, not the transform. Best case, we sould not even have that fix below and just have a different "dataset_dir" for inference with no labels in it
        if mode in ["Training","Evaluation"]:
            keys = list(datalist[0].keys())
        else:
            keys = list(datalist[0].keys())[:-1]
        print("keys:", keys)
        
        if limit_sample: 
            datalist = datalist[:limit_sample]
        
        if isinstance(transformation, Compose):
            transform = transformation
        elif transformation in available_transforms:
            # Must correct .train/.infer to make it generic to any transformation/args, or accept not full compatibility between dataset/transform
            # I believe transform should be compatible if a pattern is respected, here keys could be well-defined...
            if mode in ["Training","Evaluation"]:
                transform = available_transforms[transformation](keys).train
            else:
                transform = available_transforms[transformation](keys).infer
        else:
            transform = None

                
                
        dataset = Dataset( # A more optimized dataset can be used
            data=datalist,
            transform=transform
        )
        
        return dataset
    