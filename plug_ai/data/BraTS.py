import os
from monai.data import Dataset
from monai.transforms import Compose
from .data_aug import available_transforms



class BraTS:
    # Idea : add a createParser that manager retrieve to complete his own parser
    #def createParser(cls):
    
    def __init__(self, dataset_dir, download_dataset=False, transformation=None, mode="TRAINING", nb_class=6):
        self.dataset_dir = dataset_dir
        self.download_dataset = download_dataset
        self.transformation = transformation
        self.mode = mode
        self.nb_class = nb_class
        
        if self.download_dataset:
            self.download()
        
        self.dataset = self.get_dataset(self.dataset_dir, self.transformation, self.mode) # self.limit_sample,
        
    def download(self):
        #WIP
        print("Dowloading dataset")
        #add download procedure (cf Mednist)
        return self.dataset_dir
        
    def process(self):
        #WIP
        if self.kwargs["verbose"] == "Full" :
            print("Dataset initialization ...")
        self.kwargs["dataset"] = self.check_dataset(**self.kwargs)
        self.preprocess = self.check_preprocess(self.kwargs["preprocess"], self.kwargs)
    
    def get_datalist(self,dataset_dir):
        datalist = []

        if self.mode in ["TRAINING", "EVALUATION"]:
            with open(os.path.join(dataset_dir, "train.txt"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    file_dic = {}
                    files = line.split()
                    file_dic["data_id"] = files[0].split('/')[0]
                    for i, file in enumerate(files[:-1]):
                        file_dic[f"channel_{i}"] = os.path.join(dataset_dir, file)

                    file_dic["label"] = os.path.join(dataset_dir, files[-1])
                    datalist.append(file_dic)

            #print("got datalist, extract: \n", datalist[0])
        elif self.mode == "INFERENCE":
            subfolders = [f.path for f in os.scandir(dataset_dir) if f.is_dir()]
            for subfolder in subfolders:
                file_dic = {}
                file_dic["data_id"] = os.path.basename(subfolder)
                # List all files in the subfolder and add them as separate channels
                files = [f.path for f in os.scandir(subfolder) if f.is_file()]
                for i, file in enumerate(sorted(files)):
                    file_dic[f"channel_{i}"] = file
                datalist.append(file_dic)

        print("Datalist extact with: ", len(datalist), " items")
        
        
        return datalist


    def get_dataset(self, dataset_dir, transformation = "Default", mode="TRAINING"):#, transforma = transforms_BraTS() # limit_sample=None, 
        print("loading dataset...")
        datalist = self.get_datalist(dataset_dir)
        # Modified transformation so that the loader just takes the keys. Up to the file generator to be format things correctly, not the transform. Best case, we sould not even have that fix below and just have a different "dataset_dir" for inference with no labels in it
        if mode in ["TRAINING","EVALUATION"]:
            keys = list(datalist[0].keys())
        elif mode == "INFERENCE":
            keys = list(datalist[0].keys()) #[:-1]
        print("Dataset keys:", keys)
                
        if isinstance(transformation, Compose):
            transform = transformation
        elif transformation in available_transforms:
            # Must correct .train/.infer to make it generic to any transformation/args, or accept not full compatibility between dataset/transform
            # I believe transform should be compatible if a pattern is respected, here keys could be well-defined...

            # Can't give paramters to transformation = not good HB
            if mode in ["TRAINING","EVALUATION"]:
                transform = available_transforms[transformation](keys).train #nb_class=self.nb_class
            else:
                transform = available_transforms[transformation](keys).infer
        else:
            transform = None

        dataset = Dataset( # A more optimized dataset can be used
            data=datalist,
            transform=transform
        )
        
        return dataset

    
    
