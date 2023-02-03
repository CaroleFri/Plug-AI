from monai.data import Dataset
import os
from .data_aug import available_transforms
from monai.transforms import Compose


class DecathlonT1:
    # Idea : add a createParser that manager retrieve to complete his own parser
    # def createParser(cls):

    def __init__(self, dataset_dir, download_dataset=False, transformation=None, mode="TRAINING", nb_class=4):
        self.dataset_dir = dataset_dir
        self.download_dataset = download_dataset
        self.transformation = transformation
        self.mode = mode
        self.nb_class = nb_class

        if self.download_dataset:
            self.download()

        self.dataset = self.get_dataset(self.dataset_dir, self.transformation, self.mode)  # self.limit_sample,

    def download(self):
        # WIP
        print("Dowloading dataset")
        # add download procedure (cf Mednist)
        return self.dataset_dir

    def process(self):
        # WIP
        if self.kwargs["verbose"] == "Full":
            print("Dataset initialization ...")
        self.kwargs["dataset"] = self.check_dataset(**self.kwargs)
        self.preprocess = self.check_preprocess(self.kwargs["preprocess"], self.kwargs)


    def get_datalist(self, dataset_dir):
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

    def get_dataset(self, dataset_dir, transformation="Default",
                    mode="TRAINING"):
        print("loading dataset...")
        datalist = self.get_datalist(dataset_dir)
        # Modified transformation so that the loader just takes the keys. Up to the file generator to be format things correctly, not the transform. Best case, we sould not even have that fix below and just have a different "dataset_dir" for inference with no labels in it
        if mode in ["TRAINING", "EVALUATION"]:
            keys = list(datalist[0].keys())
        else:
            keys = list(datalist[0].keys())[:-1]
        print("keys:", keys)

        if isinstance(transformation, Compose):
            transform = transformation
        elif transformation in available_transforms:
            # Must correct .train/.infer to make it generic to any transformation/args, or accept not full compatibility between dataset/transform
            # I believe transform should be compatible if a pattern is respected, here keys could be well-defined...
            if mode in ["TRAINING", "EVALUATION"]:
                transform = available_transforms[transformation](
                    keys, nb_class=self.nb_class).train
            else:
                transform = available_transforms[transformation](keys).infer
        else:
            transform = None

        dataset = Dataset(  # A more optimized dataset can be used
            data=datalist,
            transform=transform
        )

        return dataset
