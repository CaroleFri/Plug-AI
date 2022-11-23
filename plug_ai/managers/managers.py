import os
import argparse
#import ..data.datasets
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ..models.DynUNet import PlugDynUNet
from ..data.datasets import *
from ..runners import *
from ..utils.script_utils import *
from typing import Callable
from monai.data import Dataset

from ..data import supported_datasets
from ..models import supported_models
from ..runners import supported_modes
from ..loss import supported_criterion
from ..optim import supported_optimizer

"""
from monai.data import Dataset
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, SpatialCropd, ConcatItemsd
"""

#test_var = "This is a text"

#pipeline_default = "default"


#preprocess_pipelines = {'default' : pipeline_default}



class DatasetManager:    
    @classmethod
    def createParser(cls):
        parser = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS, conflict_handler='resolve')    
        # Data information
        class_args = parser.add_argument_group('Data arguments')
        
        class_args.add_argument("--dataset", type=str)
        class_args.add_argument("--dataset_kwargs", type=dict)        
            #class_args.add_argument("--dataset_dir", type=str)
            #class_args.add_argument("--download_dataset", type=arg2bool)
            #class_args.add_argument("--task", type=str)
            #class_args.add_argument("--categories", type=list)
        
        
        class_args.add_argument("--preprocess", type=str)
        class_args.add_argument("--mode", type=str)

        class_args.add_argument("--generate_signature", type=arg2bool)
        
        class_args.add_argument("--batch_size", type=int)
        class_args.add_argument("--limit_sample", type=int)
        class_args.add_argument("--shuffle", type=arg2bool)
        class_args.add_argument("--drop_last", type=arg2bool)
        
        class_args.add_argument("--verbose", type=str)
        return parser

   
    def __init__(self, 
                 dataset = "Dataset_A", 
                 dataset_kwargs = {},
                 preprocess = None,
                 mode = 'Training',
                 generate_signature = True,
                 batch_size = 2,
                 limit_sample = True,
                 shuffle = True,
                 drop_last = True,
                 verbose = "Full", 
                 ):
        # global = dict(global)
         #dataset_dir = None,
         #download_dataset = False,
         #task = None,

         #categories = None,
        """
        Args
                 dataset: 
                 dataset_dir:
                 download:
                 preprocess:
                 task:
                 categories: 
                 mode:
                 generate_signature:
                 batch_size: 
                 limit_sample:
                 shuffle:
                 drop_last:
                 verbose:
                 dataset_kwargs: 
        Returns :
        
        """
        self.dataset = dataset  
        self.preprocess = preprocess
        self.mode = mode
        self.generate_signature = generate_signature
        self.batch_size = batch_size
        self.limit_sample = limit_sample
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.verbose = verbose

        #self.dataset_dir = dataset_dir
        #self.download_dataset = download_dataset
        #self.categories = categories
        #self.task = task

        
        self.dataset_kwargs = dataset_kwargs 
        #dataset_kwargs = dict(self.dataset_kwargs)
        #dataset_kwargs['dataset_dir'] = self.dataset_dir if not None else {}
                        
                  
        if self.verbose == "Full": 
            print("Dataset initialization ...") 
            print("Dataset_kwargs : ", self.dataset_kwargs)
        
        # Here we can have various preprocessing solutions, one of them being nnUnet (crop...)
        # We might need a preprocess_kwargs
        if preprocess:
            # self.dataset_dir = run_preprocessing() #save preprocessed data in a new dir
            print("Preprocessing the dataset")
        
        plug_ai_dataset = self.get_dataset_class()    
        self.dataset = plug_ai_dataset.dataset 
        # Should we have our
        # HOW TO MANAGE TRAIN VAL TEST SPLIT? Split dataset before? Do it in modes? 
                
        # call to class methods are done in the init of the class just like for torchvision datasets for example.
        # This allows user to use his class more easily. and this way not every dataset must implement a download method
        #if self.download_dataset == True:
        #    self.dataset_dir = dataset_class.download() # Should we export config at the end of run?
            #self.call_class_method(dataset_class.download,kwargs = dataset_kwargs)
        
        if self.generate_signature:
            print("Generating signature")
            self.signature = None #WIP signature on full dataset? on train_set? 

        # HOW TO ADD AN AUTO BATCH_SIZE? Not possible before loading/estimating model size...
        # IDEA, if arg="auto" => execution manager, first do a spinup round to find params and then run real execution
        if self.mode == "Training":
            # Should we split dataset here from a full dataset? 
            # Should our class have methods to return train/val/test sets?
            # self.datamode_set = self.call_class_method(class_method = dataset_class.get_trainset,kwargs = dataset_kwargs)
            # Fow now, went for spliting here => probably must add args to plug_ai
            train_size = int(0.8 * len(self.dataset))
            test_size = len(self.dataset) - train_size
            self.train_set, self.val_set = torch.utils.data.random_split(self.dataset, [train_size, test_size]) 
            
            self.train_loader = DataLoader(self.train_set,
                                          batch_size=self.batch_size,
                                          shuffle=self.shuffle,
                                          drop_last=self.drop_last)
            
            self.val_loader = DataLoader(self.val_set,
                              batch_size=1, # 1 for less memory (in particular if memory isn't freed) but slower, or back to self.batch_size
                              shuffle=False, #Shuffle not important for val
                              drop_last=False) #Validate on all data

            """
            # Removed this because of multiple sets/loaders possible for 1 mode
            self.datamode_loader = DataLoader(self.datamode_set,
                                              batch_size=self.batch_size,
                                              shuffle=self.shuffle,
                                              drop_last=self.drop_last)
            """
        elif self.mode == "Evaluation":
            self.val_set = self.dataset
            self.val_loader = DataLoader(self.val_set,
                              batch_size=self.batch_size, # Eventually fix to 1 for less memory (in particular if memory isn't freed) but slower,
                              shuffle=False, #Shuffle not important for val
                              drop_last=False) #Validate on all data

        elif self.mode  == "Inference":
            self.infer_set = self.dataset
            self.infer_loader = DataLoader(self.infer_set,
                              batch_size=self.batch_size,
                              shuffle=False, #Shuffle not important for inference
                              drop_last=False) #Infer on all data
        
        else:
            print("Wrong mode given")
        
        if self.generate_signature:
            print("Generating signature for dataset")
            #self.signature = self.nnUnetSignature(self.mode_load)
        #print("Dataset is", self.dataset)
        
        #self.preprocess = self.check_preprocess(self.kwargs["preprocess"], self.kwargs)
        
        
        #if self.preprocess is not None :    
        #    self.dataset = preprocess(self.kwargs["dataset"])
        #else:
        #    self.dataset = self.kwargs["dataset"]
        
        
        if self.verbose == "Full" :
            print("Dataset loaded !")
            
    
    def get_dataset_class(self):
        # Type
            # VALID
            
            # NOT VALID
        # Class
        # Else
        if isinstance( self.dataset, Dataset):
            print("Already a Pytorch dataset")
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
    
    @staticmethod
    def call_class_method(class_method, kwargs):        
        kwargs_filtered = filter_dict(class_method, kwargs)
        #kwargs_filtered = dict()
        output = class_method(kwargs_filtered)
        """
        if kwargs_filtered:
                output = class_method(kwargs_filtered)
        else:
                print("got here")
                output = class_method(kwargs_filtered)
        """
        return output  
    
    def check_preprocess(self, preprocess=None, *args, **kwargs):
        if preprocess is None:
            print("No preprocessing")
        elif preprocess in preprocess_pipelines:
            print("Preprocessing pipeline selected")
        elif isinstance(preprocess, callable):
            print("Preprocessing custom")
        else:
            raise ValueError('Expected a dataset in the valid list of datasets.')
        return preprocess 
        
        
    def preprocess_dataset(self, dataset, preprocessing):
        return None


class ModelManager:
    """
    Class to select and configure the model to Plug.
    """

    @classmethod
    def createParser(cls):
        parser = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS, conflict_handler='resolve')
        class_args = parser.add_argument_group('Model arguments')
        class_args.add_argument("--model", type=str)
        class_args.add_argument("--checkpoints_path", type=str)
        class_args.add_argument("--model_kwargs", type=list)
        #class_args.add_argument("--model_type", type=str)
        #class_args.add_argument("--config_file", type=str, default=None) # default="./test_config.yaml"
        #class_args.add_argument("--export_config", type=arg2path, help="test") # why str ?
        class_args.add_argument("--mode", type=str)
        
        return parser
    
    def __init__(self, 
                 plug_dataset,
                 model,
                 checkpoints_path,
                 model_kwargs = {},
                 mode = "Training",
                 use_signature = False,
                 res_out = False):
        # Removed parameters from function to avoid collision between a directly given parameter and the same one given with kwargs
        """
        :param model_type:
        :param model_kwargs:
        :param res_out:
        self, model_type="DynUnet", model_kwargs=None, res_out=False
        """
        """
        Args
                 plug_dataset: 
                 model:
                 download:
                 checkpoints_path:
                 model_kwargs:
                 mode: 
                 use_signature:
                 res_out: False
        Returns :
        
        """
        
        self.model = model
        self.checkpoints_path = checkpoints_path
        self.model_kwargs = model_kwargs
        self.mode = mode
        self.use_signature = use_signature
        self.res_out = res_out
        
        
        self.model = self.get_model_class()
        
        
        if self.use_signature is not None:
            available_signatures = {
                "nnUnet" : None
            }
            if self.use_signature in available_signatures:
                print("using signature to configure model")
            #
        print("Model preparation done!")

            
    def get_model_class(self):
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

    def get_model(self):
        """
        Configure and return a model to Plug.
        :return:
        """

        self.check_model_exists()
        print(f"loading {self.model_type} model")
        if self.model_type == "DynUnet":
            model = PlugDynUNet(model_kwargs=self.model_kwargs, res_out=self.res_out)

        return model    

    
class ExecutionManager:

    @classmethod
    def createParser(cls):    
        parser = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS, conflict_handler='resolve')
        class_args = parser.add_argument_group('Execution arguments')
        # Mode information
        class_args.add_argument("--mode", type=str) # Training, Evaluation, Inference,
        class_args.add_argument("--batch_size", type=int)
        class_args.add_argument("--nb_epoch", type=int)
        class_args.add_argument("--lr", type=float, help = "Learning rate")
        class_args.add_argument("--device", type=arg2device)
        class_args.add_argument("--report_log", type=arg2bool) # why str ?
        return parser
    
    
    def __init__(self, 
                 dataset_manager, 
                 model_manager,
                 mode = "Training",
                 nb_epoch = 2,
                 learning_rate = 5e-05,
                 device = "cuda",
                 random_seed = 2022,
                 report_log = False,
                 execution_kwargs = {}):
        
        self.dataset_manager = dataset_manager
        self.model_manager = model_manager
        self.mode = mode
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.device = device
        self.random_seed = random_seed
        self.report_log = report_log
        self.execution_kwargs = execution_kwargs
        
        # add an optimizer selector that allows for optimizer_kwargs (cf dataset_selector or model_selector)
        if optimizer in valid_optimizer:
            self.optimizer = valid_optimizer[optimizer]
        else:
            self.optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.99, weight_decay=3e-5, nesterov=True)

        # add a criterion selector that allows for criterion_kwargs (cf dataset_selector or model_selector)
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = DiceCELoss(to_onehot_y=True, softmax=True)

        #### same for train for loop and infer for loop
        #### same for inference step
        if isinstance(train_step, callable):
            self.train_step = train_step
        else:
            self.train_step = plugai.train_step
        
        self.train_stetp = 
        
        self.output = self.run() 
        #self.run_mode(self.kwargs["mode"], dataset, model, self.kwargs)
        #ADD HERE CALL TO MODE WITH EVERYTHING ELSE
        print("Execution over")
        
        
        
        
    def run_mode(self):
        # remove those checks by incorporating them in arg types/parser checker
        if self.mode in supported_modes :
            print("Running execution in mode : ", self.mode)
            if self.mode == "Training":
                output = supported_modes[self.mode](train_loader = dataset_manager.train_loader, 
                                                    model = model_manager.model,
                                                    optimizer = self.optimizer, 
                                                    criterion = self.criterion, 
                                                    nb_epoch = self.nb_epoch, 
                                                    device = self.device) 
            if self.mode == "Inference":
                output = supported_modes[self.mode](dataset_manager.train_loader, model, kwargs) 
            
                
        else:
            raise ValueError('Expected a mode in the valid list of modes.')
        return output

"""
# Unnecessary, if someone want to use part of plugai, he can just call data and model managers and then his own runner.
elif mode.lower() == "custom" :
    if callable(kwargs["custom_mode"]):
        kwargs["custom_mode"](dataset, model, kwargs)           
"""      



'''
   
    def __init__(self, **kwargs):
        """
        Args
        Returns
        """
        # dataset=None, preprocess=None, verbose=True,
        # Removed parameters from function to avoid collision between a directly given parameter and the same one given with kwargs
        self.kwargs = {
            'dataset' : 'Dataset_A',
            'preprocess' : None,
            'verbose' : 'Full'
            'mode' : 'Training'} # Default values for the func
        self.kwargs.update(kwargs) 
'''