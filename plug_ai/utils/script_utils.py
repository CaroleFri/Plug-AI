import argparse
import yaml
import torch
import inspect
import os

def filter_dict(func, kwarg_dict):
    if kwarg_dict:
        sign = inspect.signature(func).parameters.values()
        sign = set([val.name for val in sign])

        common_args = sign.intersection(kwarg_dict.keys())
        filtered_dict = {key: kwarg_dict[key] for key in common_args}
    else:
        filtered_dict = dict()    
    return filtered_dict

def call_func(class_method, kwargs):        
    kwargs_filtered = filter_dict(class_method, kwargs)
    output = class_method(kwargs_filtered)
    #if kwargs_filtered:
    #else:
    # output = class_method()
    return output






def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f) #load(f, Loader=yaml.FullLoader)


def arg2path(input):
    #print("checking path")
    if input is None:
        path = None
    elif isinstance(input,str):
        
        path = os.path.realpath(input)
    else : 
        raise argparse.ArgumentTypeError('Expected a path or filename')        
    return path    
    
"""
def arg2mode(input):
    if isinstance(input,str):
        if input is in available_modes:
            mode = available_modes["input"]
        else:
            raise argparse.ArgumentValueError('Expected a string with value "Training", "Inference" or "Evaluation"')
    else:
        raise argparse.ArgumentTypeError('Expected a string with value "Training", "Inference" or "Evaluation"')        
    return mode
"""

def arg2device(string):
    if "cuda" in string.lower():
        device = torch.device("cuda:0")
    elif "cpu" in string.lower():
        device = torch.device("cpu")
    #else:
        # Implement AssertionError
    return device

def arg2int(string):
    integer = 0
    return integer

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
        output = string
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
