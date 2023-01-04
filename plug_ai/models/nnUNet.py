import os
import shutil
import requests
#import gdown        
import subprocess
from monai.apps.utils import download_and_extract, download_url
  
    
    
class nnUNet_model:
    def __init__(self, 
                 nnunet_dataset_rootdir = "/gpfswork/rech/ibu/commun/nnUNet_experiment_0/nnUNet_raw_data_base",
                 nnunet_preprocessed_rootdir = "/gpfswork/rech/ibu/commun/nnUNet_experiment_0/nnUNet_preprocessed",
                 task_id = 4,
                 #task_name = 'Hippocampus',
                 verify_dataset_integrity = False,
                 no_preprocessing = False,
                 planner_2d = "ExperimentPlanner2D_v21",
                 planner_3d = "ExperimentPlanner3D_v21",
                 nbr_thread = 8,
                 overwrite_plans = None,
                 overwrite_plans_identifier = None,
                 progress = True):
        """
        Args : 
            dataset_type : A type between "MSD", "3D_TIFF", "2D_PNG", "2D_TIFF", "nnU-Net". Dataset in dataset_dir must correspond to the dataset_type.
            task_id : a value specific to your dataset between 1 and 999. For custom datasets, use a value >500 to avoid conflicts with pre-existing tasks.
            root_dir : 
            
        """
        self.nnunet_dataset_rootdir = nnunet_dataset_rootdir
        self.nnunet_preprocessed_rootdir = nnunet_preprocessed_rootdir
        self.task_id = task_id 
        #self.task_name = task_name 
        self.verify_dataset_integrity = verify_dataset_integrity
        self.no_preprocessing = no_preprocessing 
        self.planner_2d = planner_2d
        self.planner_3d = planner_3d 
        self.nbr_thread = nbr_thread
        self.overwrite_plans = overwrite_plans
        self.overwrite_plans_identifier = overwrite_plans_identifier
        self.progress = progress
                
    
        self.setup_nnUNet_folders()
        
        
        self.nnUNet_plan_and_preprocess()
        #WIP : To be corrected, it deletes the whole preprocessed_dataset_rootdir where multiple tasks could be stored
        # Would be better if it only deletes the preprocessed task 
        #if self.redo_preprocessing:
        #    shutil.rmtree(self.nnunet_preprocessed_dataset_rootdir)

        self.clean_user_env_var()

            
    def setup_nnUNet_folders(self):
        # nnUNet environnement variables needed                
        self.nnunet_dataset_env_vars = {
            "nnUNet_raw_data_base" : self.nnunet_dataset_rootdir,
            "nnUNet_preprocessed" : self.nnunet_preprocessed_rootdir,
        }
            #"RESULTS_FOLDER" : os.path.join(self.nnunet_experiment_dir, "RESULTS_FOLDER"), # This is for the model, not the dataset

        for env_var_name in self.nnunet_dataset_env_vars:
            os.environ[env_var_name] = self.nnunet_dataset_env_vars[env_var_name]         
            os.makedirs(self.nnunet_dataset_env_vars[env_var_name], exist_ok = True)


    def nnUNet_plan_and_preprocess(self):
        #nnUNet_plan_and_preprocess variable arguments
        nnUNet_plan_and_preprocess_args = {"-t" : str(self.task_id),
                                           "-pl3d" : str(self.planner_3d),
                                           "-pl2d" : str(self.planner_2d),
                                           "-tl" : str(self.nbr_thread),
                                           "-tf" : str(self.nbr_thread),                                                  
                                           "-overwrite_plans" : self.overwrite_plans,
                                           "-overwrite_plans_identifier" : str(self.overwrite_plans_identifier)}
        
        #nnUNet_plan_and_preprocess actions arguments
        nnUNet_plan_and_preprocess_action_args = {"-no_pp" : self.no_preprocessing,
                                                  "--verify_dataset_integrity" : self.verify_dataset_integrity,}
        args = ["nnUNet_plan_and_preprocess"]
        [args.extend([k, v]) for k,v in nnUNet_plan_and_preprocess_args.items() if (v is not None)]
        [args.extend([k]) for k,v in nnUNet_plan_and_preprocess_action_args.items() if v ]
        subprocess.run(args, stdout=subprocess.DEVNULL) # WIP : What should we do of nnunet outputs?
            
        
    def clean_user_env_var(self):
        for var_name in self.nnunet_dataset_env_vars.keys():
            os.environ.pop(var_name, None)
        