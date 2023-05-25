import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ..models.DynUNet import PlugDynUNet
from ..runners import *
from ..utils.script_utils import *
#from ..utils.script_utils import print_verbose
from ..utils.parser import check_parse_config

from typing import Callable
from monai.data import Dataset

from ..data import supported_datasets, supported_preprocessing
from ..models import supported_models
#from ..runners import supported_modes
import json

verbose_decorator_0 = '{:=^100}'


class DatasetManager:    
    @classmethod
    def createParser(cls):
        parser = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS, conflict_handler='resolve')    
        # Data information
        class_args = parser.add_argument_group('Data arguments')
        # Dataset arguments
        class_args.add_argument("--dataset", type=str, help="A dataset name in the valid list of of datasets supported by Plug_ai")
        # OR a callable that returns a Pytorch Dataset, this is only for the function, not the parser
        class_args.add_argument("--dataset_kwargs", type=json.loads, help="The dictionnary of args to use that are necessary for dataset")
        # To allow giving dicts in cli, used json.loads instead of dict (default dict does not auto adapt strings to dict)
        # Preprocessing arguments
        class_args.add_argument("--preprocess", type=str, help="A valid preprocessing pipeline name provided by plug_ai")
        class_args.add_argument("--preprocess_kwargs", type=json.loads, help="A dictionnary of args that are given to the processing pipeline")
        
        # Data loader arguments
        class_args.add_argument("--mode", type=arg2mode, help="A mode between 'TRAINING', 'EVALUATION' and 'INFERENCE'")        
        class_args.add_argument("--batch_size", type=int, help="Number of samples to load per batch ")
        class_args.add_argument("--train_ratio", type=int, help="Float : The fraction of the dataset to use for training, the rest will be used for final evaluation")
        class_args.add_argument("--val_ratio", type=int, help="Float : The fraction of the train set to use for validation (hp tuning)")
        class_args.add_argument("--limit_sample", type=int, help="Index value at which to stop when considering the dataset")
        class_args.add_argument("--shuffle", type=arg2bool, help="Boolean that indicates if the dataset should be shuffled at each epoch")
        class_args.add_argument("--drop_last", type=arg2bool, help="Boolean that indicates if the last batch of an epoch should be left unused when incomplete.")
        
        # Generic arguments
        class_args.add_argument("--seed", type=int, help="An int to fix every plug_ai dataset random aspects.")
        class_args.add_argument("--verbose", type=arg2verbose, help="The level of verbose wanted. None, 'RESTRICTED' or 'FULL'")
        class_args.add_argument("--export_dir", type=arg2path, help="None or Path. If Path, models, checkpoints, predictions for inference will be saved in that directory")

        return parser

  
    def __init__(self, 
                 dataset = "Dataset_A", 
                 dataset_kwargs = {},
                 preprocess = None,
                 preprocess_kwargs = {},
                 mode = 'Training',
                 batch_size = 2,
                 train_ratio = 0.8,
                 val_ratio = 0.1,
                 limit_sample = True,
                 shuffle = True,
                 drop_last = True,
                 seed = 0,
                 verbose = "Full",
                 export_dir = None,
                 ):
        """
        Args
                 dataset: A dataset name in the valid list of of datasets supported by plug_ai OR a callable that returns a Pytorch Dataset 
                 dataset_kwargs: The dictionnary of args to use that are necessary for dataset
                 preprocess: A valid preprocessing pipeline name provided by plug_ai OR your own preprocessing pipeline given as a collable
                 preprocess_kwargs: A dictionnary of args that are given to the processing pipeline 
                 generate_signature: A boolean that indicates if nnUnet fingerprint should be determined.
                 mode: A mode between Training, Evaluation and Inference 
                 batch_size: Number of samples to load per batch 
                 limit_sample: Index value at which to stop when considering the dataset
                 shuffle: Boolean that indicates if the dataset should be shuffled at each epoch
                 drop_last: Boolean that indicates if the last batch of an epoch should be left unused when incomplete.
                 seed: An int to fix every plug_ai dataset random aspects.
                 verbose: The level of verbose wanted. None, "Minimal" or "Full"
                 export_dir: None or Path. If Path, models, checkpoints, predictions for inference will be saved in that directory
        """
        self.dataset = dataset
        self.dataset_kwargs = dataset_kwargs #dataset_kwargs = dict(self.dataset_kwargs)
        self.preprocess = preprocess
        self.preprocess_kwargs = preprocess_kwargs
        #self.generate_signature = generate_signature    
        self.mode = mode
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.limit_sample = limit_sample
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.verbose = verbose
        self.export_dir = export_dir
        
        # Inside Manager check and parse of arguments reusing the same parser
        # We can remove completely the parsing and only do the check in the config generation
        self.__dict__  = check_parse_config(self.__dict__, source="Dataset_Manager")
        
        
        print_verbose(verbose_decorator_0.format(" Dataset initialization ... "), 
                      print_lvl = "RESTRICTED", 
                      verbose_lvl = self.verbose)
        
        print_verbose("Running with interpreted config:\n", '\n'.join(f'\t{k}: {v}' for k, v in self.__dict__.items()), 
                      print_lvl = "FULL", 
                      verbose_lvl = self.verbose)

        
        # Here we can have various preprocessing solutions, one of them being nnUnet (crop...)
        # We might need a preprocess_kwargs
        
        if self.preprocess is not None:
            print_verbose("Preprocessing the dataset",
                          print_lvl = "RESTRICTED", 
                          verbose_lvl = self.verbose)
            preprocessing_output = self.run_preprocessing()
        
        
        # Add Necessary Execution kwargs to the dataset_kwargs
        self.dataset_kwargs["mode"] = self.mode
        # Retrieve dataset
        plug_ai_dataset = self.get_dataset_class()
        print(self.dataset)
        
        print(type(plug_ai_dataset.dataset))
        
        if self.dataset != "nnU-Net":
            print("Loaded the dataset")
            self.dataset = plug_ai_dataset.dataset 

            # Dataset splitting : maybe put everything in a function and allow for multiple strategies
            if self.limit_sample is not None:
                dataset_size = self.limit_sample
            else:
                dataset_size = len(self.dataset)            
            self.dataset = self.dataset[:dataset_size]
            print_verbose("Using ", len(self.dataset), "elements of the full Dataset.",
                          print_lvl = "FULL", 
                          verbose_lvl = self.verbose)


            if self.mode == "TRAINING" or self.mode == "EVALUATION" :
                # HOW TO MANAGE TRAIN VAL TEST SPLIT? Split dataset before? Do it in modes? 
                # First split : Dataset => Train+VAL | TEST I
                # Second split : Train+Val => Train | Val
                temp_size = int(self.train_ratio * dataset_size)
                self.test_size = dataset_size - temp_size
                self.val_size = int(temp_size * self.val_ratio)
                self.train_size = temp_size - self.val_size


                self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(self.dataset, 
                                                                                            [self.train_size, self.val_size, self.test_size]) 
                                                                                            # generator=torch.Generator().manual_seed(2022)

                print_verbose("Train, Val, Test sizes : ", len(self.train_set), len(self.val_set), len(self.test_set),
                              print_lvl = "FULL", 
                              verbose_lvl = self.verbose)        


                # Should we split dataset here from a full dataset? 
                # Should our datasetclass have methods to return train/val/test sets? I believe they should just return and full dataset and we allow here for various spliting strategies. Fow now simple :train, val, test

                self.train_loader = DataLoader(self.train_set,
                                              batch_size=self.batch_size,
                                              shuffle=self.shuffle,
                                              drop_last=self.drop_last)

                self.val_loader = DataLoader(self.val_set,
                                  batch_size=self.batch_size, # 1 for less memory (in particular if memory isn't freed) but slower, or back to self.batch_size
                                  shuffle=False, #Shuffle not important for val
                                  drop_last=False) #Validate on all data

                self.test_loader = DataLoader(self.test_set,
                                  batch_size=self.batch_size, # Eventually fix to 1 for less memory (in particular if memory isn't freed) but slower,
                                  shuffle=False, #Shuffle not important for test
                                  drop_last=False) #Test on all data

            elif self.mode  == "INFERENCE":
                self.infer_set = self.dataset
                self.infer_loader = DataLoader(self.infer_set,
                                  batch_size=self.batch_size,
                                  shuffle=False, #Shuffle not important for inference
                                  drop_last=False) #Infer on all data
        
        if self.verbose == "Full" :
            print("Dataset prepared !")
            
    
    def get_dataset_class(self):
        """
        Checks self.dataset to return a Pytorch Dataset
        """
        if isinstance( self.dataset, Dataset):
            print_verbose("Already a Pytorch dataset",
                          print_lvl = "RESTRICTED", 
                          verbose_lvl = self.verbose)
            dataset = self.dataset        
        if isinstance(self.dataset, str):
            #if os.path.isdir(self.dataset) or os.path.isfile(self.dataset):
            #   print("Found folder or file")
            #   dataset = None
                # For now we just check that the path is a file/folder, we must add extra verification of the file format and/or folder organisation
                # dataset = np.load(dataset) if npy/npz file, pd.load_csv, ... Must decide what type of dataset to handle.
            if self.dataset in supported_datasets:
                
                print("Dataset type is valid")
                #dataset_kwargs_filtered = call_func(supported_datasets[self.dataset], self.dataset_kwargs)
                # alternative, have a wrapper function that calls it with appropriate args, cons : non-explicit error if arg missing
                dataset_kwargs_filtered = filter_dict(supported_datasets[self.dataset], self.dataset_kwargs)
                dataset = supported_datasets[self.dataset](**dataset_kwargs_filtered) 
            else:
                raise ValueError('Expected a dataset in the valid list of datasets.')
        else :
            raise ValueError('Expected a dataset in the valid list of datasets.')
        return dataset
    
    def run_preprocessing(self):
        preprocess_kwargs_filtered = filter_dict(self.preprocess, self.preprocess_kwargs)
        output = self.preprocess(**preprocess_kwargs_filtered) 
        return output
    
    @staticmethod
    def call_class_method(class_method, kwargs):        
        kwargs_filtered = filter_dict(class_method, kwargs)
        #kwargs_filtered = dict()
        output = class_method(kwargs_filtered)
        return output  
    
    def check_preprocess(self, preprocess=None, *args, **kwargs):
        if preprocess is None:
            print("No preprocessing")
        elif preprocess in preprocess_pipelines: # Doesn't exist ??
            print("Preprocessing pipeline selected")
        elif isinstance(preprocess, callable):
            print("Preprocessing custom")
        else:
            raise ValueError('Expected a dataset in the valid list of datasets.')
        return preprocess 
        
        
    def preprocess_dataset(self, dataset, preprocessing):
        #WIP with nnUnet
        return None


class ModelManager:
    """
    Class to select and configure the model to Plug.
    """

    @classmethod
    def createParser(cls):
        parser = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS, conflict_handler='resolve')
        class_args = parser.add_argument_group('Model arguments')
        class_args.add_argument("--model", type=str, help = '''A model in the valid list of supported model or a
                                callable that instantiate a Pytorch/Monai model''')
        #class_args.add_argument("--checkpoints_path", type=str, help = "a path to checkpoints for the model")
        class_args.add_argument("--model_kwargs", type=list, help = "Every arguments which should be passed to the model callable")
        class_args.add_argument("--model_weights_path", type=str, help = "Path to trained model weights")
        class_args.add_argument("--mode", type=str, help = "A mode between Training, Evaluation and Inference ")
        #class_args.add_argument("--use_signature", type=arg2bool, help = '''A boolean that should indicate if to use or not the signature of the dataset for adaptation''')        
        class_args.add_argument("--verbose", type=str, help = "The level of verbose wanted. None, 'Minimal' or 'Full'")
        class_args.add_argument("--export_dir", type=arg2path, help="None or Path. If Path, models, checkpoints, predictions for inference will be saved in that directory")
        return parser
    
    def __init__(self, 
                 plug_dataset,
                 model,
                 device,
                 model_kwargs = {},
                 model_weights_path = None,
                 mode = "Training",
                 verbose = "FULL",
                 export_dir = None,
                 ):
        #use_signature = False,
        #res_out = False,        
        #checkpoints_path,
                 
        """
        Args
                 plug_dataset: A Plug_AI dataset class which has the necessary attributes such as the  dataloaders for the wanted mode or the signature...
                 model: A callable to instantiate a Pytorch or Monai model (WIP or the model right away)
                 model_kwargs: Every arguments which should be passed to the model callable
                 mode: A mode between Training, Evaluation and Inference 
                 verbose: The level of verbose wanted. None, "Minimal" or "Full"
                 export_dir: None or Path. If Path, models, checkpoints, predictions for inference will be saved in that directory
                 
        """

        #self.plug_dataset
        self.model = model
        self.model_kwargs = model_kwargs
        self.model_weights_path = model_weights_path
        self.device = device
        self.mode = mode
        self.verbose = verbose
        self.export_dir = export_dir

        if isinstance(model, str): self.model_name = model
        # Inside Manager check and parse of arguments reusing the same parser
        # We can remove completely the parsing and only do the check in the config generation
        self.__dict__  = check_parse_config(self.__dict__, source="Model_Manager")

        
        print_verbose(verbose_decorator_0.format(" Model initialization ... "), 
                      print_lvl = "RESTRICTED", 
                      verbose_lvl = self.verbose)
        print_verbose("Running with interpreted config:\n", '\n'.join(f'\t{k}: {v}' for k, v in self.__dict__.items()), 
                      print_lvl = "FULL", 
                      verbose_lvl = self.verbose)
        
        if self.model == "nnU-Net":
            self.model = self.get_model()
        else:
            self.model = self.get_model().to(self.device)
            #model_weights_path = "/gpfsdswork/projects/rech/ibu/ssos023/Plug-AI/examples/Training_Inference_demo/model_last.pt"
            if self.model_weights_path is not None:
                self.model.load_state_dict(torch.load(self.model_weights_path))
        print("Model preparation done!")

            
    def get_model(self):
        if isinstance( self.model, nn.Module):
            print("Already a Pytorch model")
            model = self.model        
        if isinstance(self.model, str):
            if self.model in supported_models:
                print("Model type is valid")
                model_kwargs_filtered = filter_dict(supported_models[self.model], self.model_kwargs)
                model = supported_models[self.model](**model_kwargs_filtered) 
            else:
                raise ValueError('Expected a model in the valid list of models.')
        else :
            raise ValueError('Expected a model in the valid list of models.')
        return model            
        
    def check_model_exists(self):
        """
        Check if the model exists in our database. Raise an error if not.
        :return:
        """

        print("checking model exists")
        if self.model_type not in self.list_model_type:
            raise ValueError(f"{self.model_type} is not in the list of PlugModel")


class ExecutionManager:

    @classmethod
    def createParser(cls):    
        parser = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS, conflict_handler='resolve')
        class_args = parser.add_argument_group('Execution arguments')
        # Mode information
        class_args.add_argument("--loop", type=str) # Training, Evaluation, Inference,
        class_args.add_argument("--loop_kwargs", type=dict) # Training, Evaluation, Inference,
        class_args.add_argument("--mode", type=arg2bool) # Training, Evaluation, Inference,
        class_args.add_argument("--nb_epoch", type=int)
        #class_args.add_argument("--learning_rate", type=float, help = "Learning rate")
        class_args.add_argument("--device", type=arg2device)
        class_args.add_argument("--seed", type=int) 
        class_args.add_argument("--report_log", type=arg2bool) # why str ?
        class_args.add_argument("--criterion", type=str) # why str ?
        class_args.add_argument("--criterion_kwargs", type=dict)
        class_args.add_argument("--metric", type=str)
        class_args.add_argument("--metric_kwargs", type=dict)
        class_args.add_argument("--optimizer", type=str)
        class_args.add_argument("--optimizer_kwargs", type=dict)
        class_args.add_argument("--lr_scheduler", type=str)
        class_args.add_argument("--lr_scheduler_kwargs", type=dict)
        class_args.add_argument("--verbose", type=str)
        class_args.add_argument("--export_dir", type=arg2path, help="None or Path. If Path, models, checkpoints, predictions for inference will be saved in that directory")
        #class_args.add_argument("--batch_size", type=int) #DatasetManager Parameter
        return parser
    
    
    def __init__(self, 
                 dataset_manager, 
                 model_manager,
                 loop,
                 loop_kwargs = {},
                 mode = "Training",
                 nb_epoch = 2,
                 device = "cuda",
                 seed = 2022,
                 report_log = False,
                 criterion = None,
                 metric = None,
                 criterion_kwargs = {},
                 metric_kwargs = {},
                 optimizer = None,
                 optimizer_kwargs = {},
                 lr_scheduler = None,
                 lr_scheduler_kwargs = {},
                 verbose = "FULL",                 
                 export_dir = None
                ):
        #learning_rate = 5e-05,

        self.loop = loop
        self.loop_kwargs = loop_kwargs
        self.mode = mode
        self.nb_epoch = nb_epoch
        #self.learning_rate = learning_rate
        self.device = device
        self.seed = seed
        self.report_log = report_log
        self.criterion = criterion
        self.metric = metric
        self.criterion_kwargs = criterion_kwargs
        self.metric_kwargs = metric_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.verbose = verbose
        self.export_dir = export_dir
        
        # Inside Manager check and parse of arguments reusing the same parser
        # We can remove completely the parsing and only do the check in the config generation
        self.__dict__  = check_parse_config(self.__dict__, source="Execution_Manager")
        #HOTFIX, dataset_manager and model_manager aren't in check_parse_config so they are filtered, thus the definition below
        
        self.dataset_manager = dataset_manager
        self.model_manager = model_manager
        
        print_verbose(verbose_decorator_0.format(" Execution initialization ... "), 
                      print_lvl = "RESTRICTED", 
                      verbose_lvl = self.verbose)
        print_verbose("Running with interpreted config:\n", '\n'.join(f'\t{k}: {v}' for k, v in self.__dict__.items()), 
                      print_lvl = "FULL", 
                      verbose_lvl = self.verbose)

        
        
        if self.mode == "TRAINING":
            #things to do around the train loop, will be different from eval or infer (ex: no need of optimizer...)
            print("TRAINING MODE : ")
            # WIP: homogenize what/how each loop should return and avoid the following if/else
            if self.model_manager.model_name == "nnU-Net":
                self.loop_kwargs["device"] = self.device
                self.loop_kwargs["verbose"] = self.verbose
                loop_kwargs_filtered = filter_dict(self.loop, self.loop_kwargs)
                print(loop_kwargs_filtered.keys())
                self.model = self.loop(**loop_kwargs_filtered)
                
            else:
                #Put everything in a dict then filter 
                # WIP: define what to put in
                # WARNING : Do not rewrite given loop_kwargs values (for example if a verbose is given already for the loop) 
                self.loop_kwargs["train_loader"] = self.dataset_manager.train_loader
                self.loop_kwargs["val_loader"] = self.dataset_manager.val_loader
                self.loop_kwargs["model"] = self.model_manager.model
                self.loop_kwargs["optimizer"] = self.optimizer
                self.loop_kwargs["lr_scheduler"] = self.lr_scheduler
                self.loop_kwargs["criterion"] = self.criterion
                self.loop_kwargs["metric"] = self.metric
                self.loop_kwargs["optimizer_kwargs"] = self.optimizer_kwargs
                self.loop_kwargs["criterion_kwargs"] = self.criterion_kwargs
                self.loop_kwargs["lr_scheduler_kwargs"] = self.lr_scheduler_kwargs
                self.loop_kwargs["metric_kwargs"] = self.metric_kwargs
                self.loop_kwargs["nb_epoch"] = self.nb_epoch
                self.loop_kwargs["device"] = self.device
                self.loop_kwargs["verbose"] = self.verbose
                self.loop_kwargs["export_dir"] = self.export_dir
                #self.loop_kwargs["train_step"] = self.train_step
                # train_step selector not here yet


                loop_kwargs_filtered = filter_dict(self.loop, self.loop_kwargs)
                print(loop_kwargs_filtered.keys())
                self.model = self.loop(**loop_kwargs_filtered).model
                
                if self.export_dir is not None:
                    # Create directory if it does not already exist
                    os.makedirs(os.path.join(self.export_dir, "models"), exist_ok=True)
                    # Save the model returned by the training
                    torch.save(self.model.state_dict(), os.path.join(self.export_dir, "models", "model_last.pt"))
        
        elif self.mode == "EVALUATION":
            #things to do around the eval loop
            print("EVAL LOOP")
        
        elif self.mode == "INFERENCE":
            #things to do around the infer loop
            print("INFERENCE MODE:")
            if self.model_manager.model_name == "nnU-Net":
                print("Nothing done, developpement in progress")
            else:
                self.loop_kwargs["infer_loader"] = self.dataset_manager.infer_loader
                self.loop_kwargs["model"] = self.model_manager.model
                self.loop_kwargs["metric"] = self.metric
                self.loop_kwargs["metric_kwargs"] = self.metric_kwargs
                self.loop_kwargs["device"] = self.device
                self.loop_kwargs["verbose"] = self.verbose
                self.loop_kwargs["export_dir"] = self.export_dir
                
                loop_kwargs_filtered = filter_dict(self.loop, self.loop_kwargs)
                print(loop_kwargs_filtered.keys())
                self.loop(**loop_kwargs_filtered)
                                
        print("Execution over")
