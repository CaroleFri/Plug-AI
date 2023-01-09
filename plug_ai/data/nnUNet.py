import os
import shutil
import requests
#import gdown        
import subprocess
from monai.apps.utils import download_and_extract, download_url
from ..utils.script_utils import download_file_from_google_drive, gdrive_url2id
    
    
class nnUNet_dataset:
    def __init__(self, 
                 nnunet_dataset_rootdir = "/gpfswork/rech/ibu/commun/nnUNet_experiment_0/nnUNet_raw_data_base",
                 task_id = 4,
                 #task_name = 'Hippocampus',
                 raw_dataset_type = "MSD",
                 raw_dataset_dir = "/gpfswork/rech/ibu/commun/datasets/MSD/Task15_TestTask",
                 download_raw_dataset = True,
                 nbr_thread = 8,
                 progress = True):
        """
        Args : 
            dataset_type : A type between "MSD", "3D_TIFF", "2D_PNG", "2D_TIFF", "nnU-Net". Dataset in dataset_dir must correspond to the dataset_type.
            task_id : a value specific to your dataset between 1 and 999. For custom datasets, use a value >500 to avoid conflicts with pre-existing tasks.
            root_dir : 
            
        """
        self.nnunet_dataset_rootdir = nnunet_dataset_rootdir
        self.task_id = task_id 
        #self.task_name = task_name 
        self.raw_dataset_type = raw_dataset_type
        self.raw_dataset_dir = raw_dataset_dir 
        self.download_raw_dataset = download_raw_dataset
        self.nbr_thread = nbr_thread
        self.progress = progress
        
        print("nnunet_dict : ", self.__dict__)
        
    
        self.valid_dataset_types = {
            "nnU-Net" : self.prepare_nnUNet,
            "MSD" : self.setup_MSD,
            "3D_TIFF" : self.setup_3D_TIFF, 
            "2D_PNG" : self.setup_2D_PNG, 
            "2D_TIFF" : self.setup_2D_TIFF
        }

        self.check_dataset_type()

        self.setup_nnUNet_folders()

        self.valid_dataset_types[self.raw_dataset_type]()
        
        self.clean_user_env_var()

    def check_dataset_type(self):
        if isinstance(self.raw_dataset_type, str):
            if self.raw_dataset_type not in self.valid_dataset_types.keys():
                raise ValueError('''Expected dataset_type to be either "MSD", "3D_TIFF", "2D_PNG", "2D_TIFF"''')
            else:
                print("nnunet dataset type checked")
        else:
            raise TypeError('''Expected dataset_type to be a string either "MSD", "3D_TIFF", "2D_PNG", "2D_TIFF"''')  
            
    def setup_nnUNet_folders(self):
        # nnUNet environnement variables needed                
        self.nnunet_dataset_env_vars = {
            "nnUNet_raw_data_base" : self.nnunet_dataset_rootdir,
        }

        for env_var_name in self.nnunet_dataset_env_vars: # Why a loop ??
            os.environ[env_var_name] = self.nnunet_dataset_env_vars[env_var_name]         
            os.makedirs(self.nnunet_dataset_env_vars[env_var_name], exist_ok = True)
        
            
    def get_msd_task_dataset(self):
        tasks_infos = {
            1 : {"name":"Task01_BrainTumour", 
                 "urlg":"https://drive.google.com/file/d/1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU/view?usp=share_link",
                 "url":"https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
                 "md5":"240a19d752f0d9e9101544901065d872"},
            2 : {"name":"Task02_Heart", 
                 "urlg":"https://drive.google.com/file/d/1wEB2I6S6tQBVEPxir8cA5kFB8gTQadYY/view?usp=share_link",
                 "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar",
                 "md5": "06ee59366e1e5124267b774dbd654057"},
            3 : {"name":"Task03_Liver", 
                 "urlg":"https://drive.google.com/file/d/1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu/view?usp=share_link",
                 "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar",
                 "md5": "a90ec6c4aa7f6a3d087205e23d4e6397"},
            4 : {"name":"Task04_Hippocampus", 
                 "urlg":"https://drive.google.com/file/d/1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C/view?usp=share_link",
                 "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar",
                 "md5": "9d24dba78a72977dbd1d2e110310f31b"},
            5 : {"name":"Task05_Prostate", 
                 "urlg":"https://drive.google.com/file/d/1Ff7c21UksxyT4JfETjaarmuKEjdqe1-a/view?usp=share_link",
                 "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar",
                 "md5": "35138f08b1efaef89d7424d2bcc928db"},
            6 : {"name":"Task06_Lung", 
                 "urlg":"https://drive.google.com/file/d/1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi/view?usp=share_link",
                 "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar",
                 "md5": "8afd997733c7fc0432f71255ba4e52dc"},
            7 : {"name":"Task07_Pancreas", 
                 "urlg":"https://drive.google.com/file/d/1YZQFSonulXuagMIfbJkZeTFJ6qEUuUxL/view?usp=share_link",
                 "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar",
                 "md5": "4f7080cfca169fa8066d17ce6eb061e4"},
            8 : {"name":"Task08_HepaticVessel", 
                 "urlg":"https://drive.google.com/file/d/1qVrpV7vmhIsUxFiH189LmAn0ALbAPrgS/view?usp=share_link",
                 "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar",
                 "md5": "641d79e80ec66453921d997fbf12a29c"},
            9 : {"name":"Task09_Spleen", 
                 "urlg":"https://drive.google.com/file/d/1jzeNU1EKnK81PyTsrx0ujfNl-t0Jo8uE/view?usp=share_link",
                 "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar",
                 "md5": "410d4a301da4e5b2f6f86ec3ddba524e",},
            10 : {"name":"Task10_Colon", 
                  "urlg":"https://drive.google.com/file/d/1m7tMpE9qEcQGQjL_BdMD-Mvgmc44hG1Y/view?usp=share_link",
                  "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar",
                  "md5": "bad7a188931dc2f6acf72b08eb6202d0"},
        }
        
        
        if self.download_raw_dataset not in tasks_infos.keys():
            raise ValueError("Not a valid task number to download.")
        
        os.makedirs(self.raw_dataset_dir, exist_ok = True)
        
        # Retrieving the dataset task
        archive_filename = tasks_infos[self.download_raw_dataset]["name"] + '.tar'
        archive_path = os.path.join(self.raw_dataset_dir, archive_filename)
        #download_file_from_google_drive(gdrive_url2id(tasks_infos[self.task_id]["url"]), archive_path)
        #gdown.download(url=tasks_infos[self.task_id]["url"], output=archive_path, quiet=False, fuzzy=True)
        #shutil.unpack_archive(archive_path, self.dataset_dir)    
        download_and_extract(
            url=tasks_infos[self.download_raw_dataset]["url"],
            filepath=archive_path,
            output_dir=self.raw_dataset_dir,
            hash_val=tasks_infos[self.download_raw_dataset]["md5"],
            hash_type="md5",
            progress=True)
        
        os.remove(archive_path)
        #for filename in os.listdir(os.path.join(self.raw_dataset_dir, tasks_infos[self.task_id]["name"])):
        #    shutil.move(os.path.join(self.raw_dataset_dir, tasks_infos[self.task_id]["name"], filename), os.path.join(self.raw_dataset_dir, filename))
        shutil.copytree(os.path.join(self.raw_dataset_dir, tasks_infos[self.download_raw_dataset]["name"]), self.raw_dataset_dir, dirs_exist_ok=True)
        shutil.rmtree((os.path.join(self.raw_dataset_dir, tasks_infos[self.download_raw_dataset]["name"])))
        
        # Licensing of the dataset
        license_infos = {
            "name" : "license.txt",
            "url" : "https://drive.google.com/file/d/18dLVTJtkp052danMjzlirAgIsklT_Aem/view?usp=share_link"
        }
        license_path = os.path.join(self.raw_dataset_dir,license_infos["name"])
        # download_file_from_google_drive(gdrive_url2id(license_infos["url"]), license_path) # Pas défini ?

            
        
    def clean_user_env_var(self):
        for var_name in self.nnunet_dataset_env_vars.keys():
            os.environ.pop(var_name, None)
        
    def prepare_nnUNet(self):
        print("already nnunet, nothing to be done except setup variables and run plan and preprocess")
            
    def setup_MSD(self):
        #Get MSD
        if isinstance(self.download_raw_dataset, int) and (self.download_raw_dataset is not False):
            self.get_msd_task_dataset()
    
        # Convert MSD to nnunet
        print("Now converting MSD to nnUnet")
        i = str(self.raw_dataset_dir)
        p = str(self.nbr_thread)
        output_task_id = str(self.task_id)
        args = ["nnUNet_convert_decathlon_task", "-i", i, "-p", p, "-output_task_id", output_task_id]
        subprocess.run(args)

    def setup_3D_TIFF(self):
        print("Here is prepare_3dTIFF")
        # Convert 3D_TIFF to nnunet
    def setup_2D_PNG(self):
        print("Here is prepare_2DPNG")
        # Convert 2D_PNG to nnunet
    
    def setup_2D_TIFF(self):
        print("Here is prepare_2DTIFF")
        # Convert 2D_TIFF to nnunet
