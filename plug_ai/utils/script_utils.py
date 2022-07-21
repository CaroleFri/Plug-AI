import argparse
import yaml
import torch

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f) #load(f, Loader=yaml.FullLoader)

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
