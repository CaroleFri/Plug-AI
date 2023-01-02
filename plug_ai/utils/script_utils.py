import argparse
import ruamel.yaml
#import yaml
import torch
import inspect
import os
from ..loss import supported_criterion
from ..optim import supported_optimizer
import json
from typing import Callable

import requests
import gdown
  


def print_verbose(*args, print_lvl, verbose_lvl):
    lvls = ["RESTRICTED", "FULL"]
    if verbose_lvl is not None:
        if lvls.index(verbose_lvl) >= lvls.index(print_lvl):
            print(*args)

def filter_dict(func, kwarg_dict):
    if kwarg_dict:
        sign = inspect.signature(func).parameters.values()
        sign = set([val.name for val in sign])

        common_args = sign.intersection(kwarg_dict.keys())
        filtered_dict = {key: kwarg_dict[key] for key in common_args}
         #rejected_dict = {key: kwarg_dict[key] for key not in sign} WIP
    else:
        filtered_dict = dict()   
    # WIP : Add a warning indicating that some kwargs have been unused
    return filtered_dict

def call_func(class_method, kwargs):        
    kwargs_filtered = filter_dict(class_method, kwargs)
    output = class_method(kwargs_filtered)
    #if kwargs_filtered:
    #else:
    # output = class_method()
    return output






def read_yaml(file_path):
    yaml = ruamel.yaml.YAML(typ='safe')
    yaml.preserve_quotes = True
    with open(file_path) as cf:
        config_file  = yaml.load(cf)
        
    #with open(file_path, "r") as f:
    #    return yaml.safe_load(f) #load(f, Loader=yaml.FullLoader)
    return config_file

def arg2path(input):
    #print("checking path")
    if input is None:
        path = None
    elif isinstance(input,str):
        path = os.path.realpath(input)
    else : 
        raise argparse.ArgumentTypeError('Expected a path or filename')        
    return path    
   
def arg2mode(input):
    available_modes = ['TRAINING', 'EVALUATION', 'INFERENCE']
    if isinstance(input,str):
        if input.upper() in available_modes:
            mode = input.upper()
        else:
            raise ValueError('Expected a string with VALUE "TRAINING", "EVALUATION" or "INFERENCE"')
    else:
        raise TypeError('Expected a STRING with value "TRAINING", "EVALUATION" or "INFERENCE"')        
    return mode

def arg2verbose(input):
    available_verboses = ['FULL', 'RESTRICTED', None]
    if input is None:
        verbose = None
    if isinstance(input,str):
        if input.upper() in available_verboses:
            verbose = input.upper()
        else:
            raise ValueError('Expected a string with VALUE "FULL", "RESTRICTED"')
    else:
        raise TypeError('Expected None or a STRING with value "FULL" or "RESTRICTED"')        
    return verbose

def arg2device(input):
    if "cuda" in input.lower():
        device =  "cuda" #torch.device("cuda:0")
    elif "cpu" in input.lower():
        device = "cpu" #torch.device("cpu")
    #else:
        # Implement AssertionError
    return device

def arg2dict(input):
    if input is None:
        dictionnary = {}
    elif isinstance(input, dict):
        dictionnary = input
    elif isinstance(input, str):
        if '{"' in input:
            dictionnary = json.loads(input)
        elif "{'" in input:
            dictionnary  = eval(input)
        else:
            raise TypeError('String does not appear to respect dict-structure (json or python default)')
    else:
        raise TypeError('Expected a dict-like structure or None')
    return dictionnary

def arg2int(string):
    integer = 0
    return integer

def arg2seed(input):
    if input is None:
        seed = None
    elif isinstance(input, int):
        seed = input
    elif isinstance(input, str):
        seed = int(input)
    else:
        raise TypeError("Expected a seed-like structure (int)")

def arg2bool(input):
    '''
    Checks and converts input to a boolean if it is a string in TRUE_LIST or FALSE_LIST.
    Parameters
    ----------
    input : boolean or string 
        a string with a boolean meaning or a boolean
    Returns
    -------
    ouput : boolean
        A simple boolean retrieved from a boolean-like string
    '''
    TRUE_LIST = ['true', 't', 'yes', 'y', '1', 'on']
    FALSE_LIST = ['false', 'f', 'no', 'n', '0', 'off']
    if isinstance(input, bool):
        output = input
    elif isinstance(input,str):
        if input.lower() in TRUE_LIST:
            output = True
        elif input.lower() in FALSE_LIST:
            output = False
        else:
            #raise ValueError('Expected a string in TRUE_LIST or FALSE_LIST.')
            raise argparse.ArgumentTypeError('Expected a string in TRUE_LIST or FALSE_LIST.')
    else:
        #raise TypeError('Expected a boolean or a string in TRUE_LIST or FALSE_LIST')
        raise argparse.ArgumentTypeError('Expected a boolean or a string in TRUE_LIST or FALSE_LIST')
    return output

# Moved both selector in arg2criterion and arg2optimizer would be nice
# Issue : cannot instantiate them with criterion_kwargs and optimizer_kwargs
# Idea : arg2criterion/optimizer either return an instance or a callable which can be given args to return the instance.
def arg2criterion(input):
    if isinstance(input, Callable):
        criterion = input
    elif input is None:
        criterion = supported_criterion["Default"]#(to_onehot_y=True, softmax=True)
    elif isinstance(input, str):
        if input in supported_criterion:
                if input == "Default":
                    criterion = supported_criterion["Default"]#(to_onehot_y=True, softmax=True)
                else:
                    criterion = supported_criterion[input]#(**criterion_kwargs)
        else:
            raise ValueError('Expected criterion to be : a torch criterion, a string in the valid list of criterion or None(default criterion).')
    return criterion


def arg2optimizer(input):
    if isinstance(input, Callable):
        optimizer = input
    elif input is None:
        optimizer = supported_optimizer["Default"]#(self.model.parameters(), lr=0.0001, momentum=0.99, weight_decay=3e-5, nesterov=True)
    elif isinstance(input, str):        
        if input == "Default":
            optimizer = supported_optimizer["Default"] #(self.model.parameters(), **optimizer_kwargs)                
        else:
            optimizer = supported_optimizer[input] #(self.model.parameters(), **optimizer_kwargs)
    else:
        raise ValueError('Expected optimizer to be : a torch optimizer callable, a string in the valid list of optimizer or None(default optim).')
    return optimizer

def arg2loop(input):
    from ..runners import supported_loop
    
    if isinstance(input, Callable):
        loop = input
    elif input is None:
        loop = supported_loop["Default_train"]#(self.model.parameters(), lr=0.0001, momentum=0.99, weight_decay=3e-5, nesterov=True)
    elif isinstance(input, str):        
        if input == "Default":
            loop = supported_loop["Default_train"]                 
        else:
            loop = supported_loop[input] 
    else:
        raise ValueError('Expected a train loop: : a callable or a string in the valid list of loop or None(default train loop).')
    return loop


def arg2step(input):
    from ..runners import supported_step
    
    if isinstance(input, Callable):
        step = input
    elif input is None:
        step = supported_step["Default_train"]#(self.model.parameters(), lr=0.0001, momentum=0.99, weight_decay=3e-5, nesterov=True)
    elif isinstance(input, str):        
        if input == "Default":
            step = supported_step["Default_train"]                 
        else:
            step = supported_step[input] 
    else:
        raise ValueError('Expected a train step: : a callable or a string in the valid list of step or None(default train step).')
    return step


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id , 'confirm': 1 }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

                
def gdrive_url2id(url):
    prefix = "https://drive.google.com/file/d/"
    suffix = "/view?usp=share_link"
    id = url.removeprefix(prefix).removesuffix(suffix)
    return id
    