import os
import shutil
import subprocess
  
    
    
class nnUNet_Trainer:
    def __init__(self,
                 nnunet_dataset_rootdir = "/gpfswork/rech/ibu/commun/nnUNet_experiment_0/nnUNet_raw_data_base",
                 nnunet_preprocessed_rootdir = "/gpfswork/rech/ibu/commun/nnUNet_experiment_0/nnUNet_preprocessed",
                 nnunet_experiment_dir = "/gpfswork/rech/ibu/commun/nnUNet_experiment_0",
                 task_id = 4,
                 network = "2d",
                 network_trainer = "nnUNetTrainerV2", 
                 task = "TaskXXX_MYTASK",
                 fold = 'all',
                 validation_only = False, 
                 continue_training = False,
                 plans_identifier = "nnUNetPlansv2.1",
                 use_compressed_data = False,
                 deterministic = False,
                 npz = False,
                 find_lr = False,
                 valbest = False,
                 fp32 = False,
                 val_folder = "validation_raw",
                 disable_saving = False,
                 disable_postprocessing_on_folds = False, 
                 val_disable_overwrite = True,
                 disable_next_stage_pred = False,
                 pretrained_weights = None):
        """
        Args : 
             nnunet_dataset_rootdir:
             nnunet_preprocessed_rootdir:
             nnunet_experiment_dir:
             task_id:
             network:
             network_trainer:
             task:
             fold:
             validation_only: 
             continue_training:
             plans_identifier:
             use_compressed_data:
             deterministic:
             npz:
             find_lr:
             valbest:
             fp32:
             val_folder:
             disable_saving:
             disable_postprocessing_on_folds: 
             val_disable_overwrite:
             disable_next_stage_pred:
             pretrained_weights:
        """ 
        self.nnunet_dataset_rootdir = nnunet_dataset_rootdir
        self.nnunet_preprocessed_rootdir = nnunet_preprocessed_rootdir
        self.nnunet_experiment_dir = nnunet_experiment_dir
        self.task_id = task_id
        self.network = network
        self.network_trainer = network_trainer
        self.task = task
        self.fold = fold
        self.validation_only = validation_only  
        self.continue_training = continue_training
        self.plans_identifier = plans_identifier
        self.use_compressed_data = use_compressed_data
        self.deterministic = deterministic
        self.npz = npz
        self.find_lr = find_lr
        self.valbest = valbest
        self.fp32 = fp32
        self.val_folder = val_folder
        self.disable_saving = disable_saving
        self.disable_postprocessing_on_folds = disable_postprocessing_on_folds
        self.val_disable_overwrite = val_disable_overwrite
        self.disable_next_stage_pred = disable_next_stage_pred
        self.pretrained_weights = pretrained_weights

        self.setup_nnUNet_folders()
        
        
        #WIP : add a loop to go through all nnunet trainings (2D, 3D, 3D_cascade) if self.network=='all' and allow for automatic best configuration determination
        self.nnUNet_train()
        
        self.clean_user_env_var()

            
    def setup_nnUNet_folders(self):
        # nnUNet environnement variables needed                
        self.nnunet_dataset_env_vars = {
            "nnUNet_raw_data_base" : self.nnunet_dataset_rootdir,
            "nnUNet_preprocessed" : self.nnunet_preprocessed_rootdir,
            "RESULTS_FOLDER" : os.path.join(self.nnunet_experiment_dir),
        }

        for env_var_name in self.nnunet_dataset_env_vars:
            os.environ[env_var_name] = self.nnunet_dataset_env_vars[env_var_name]         
            os.makedirs(self.nnunet_dataset_env_vars[env_var_name], exist_ok = True)


    def nnUNet_train(self):
        #nnUNet_train_action_args variable arguments
        nnUNet_train_positional_args = {"network" : str(self.network),
                             "network_trainer" : str(self.network_trainer),
                             "task" : str(self.task),
                             "fold" : str(self.fold)}
        
        
        nnUNet_train_args = {"-p" : str(self.plans_identifier),
                             "--val_folder" : str(self.val_folder),
                             "--pretrained_weights" : self.pretrained_weights}
        
        #nnUNet_train_action_args actions arguments
        nnUNet_train_action_args = {"--validation_only" : self.validation_only,
                                    "--continue_training" : self.continue_training,
                                    "--use_compressed_data" : self.use_compressed_data,
                                    "--deterministic" : self.deterministic,
                                    "--npz" : self.npz,
                                    "--find_lr" : self.find_lr,
                                    "--valbest" : self.valbest,
                                    "--fp32" : self.fp32,
                                    "--disable_saving" : self.disable_saving,
                                    "--disable_postprocessing_on_folds" : self.disable_postprocessing_on_folds,
                                    "--val_disable_overwrite" : self.val_disable_overwrite,
                                    "--disable_next_stage_pred" : self.disable_next_stage_pred}
        
        args = ["nnUNet_train"]
        [args.extend([v]) for k,v in nnUNet_train_positional_args.items()]
        [args.extend([k, v]) for k,v in nnUNet_train_args.items() if (v is not None)]
        [args.extend([k]) for k,v in nnUNet_train_action_args.items() if v ]
        subprocess.run(args) # WIP : What should we do of nnunet outputs? stdout=subprocess.DEVNULL
    
    def nnUNet_find_best(self):
        args = ["nnUNet_find_best_configuration"]


        
    def clean_user_env_var(self):
        for var_name in self.nnunet_dataset_env_vars.keys():
            os.environ.pop(var_name, None)
        