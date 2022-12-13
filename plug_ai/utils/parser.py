import argparse
import yaml as yl
import ruamel.yaml
from .script_utils import *
import os

utils_folder = os.path.dirname(os.path.realpath(__file__))
default_config_file = os.path.join(utils_folder, "default_config.yaml")
print("default config path", default_config_file)

import sys

def createGlobalParser():
    """
    A parser to use as a parent parser to handle conflicts and put args that are present in multiple other parents in a global section.
    """
    parser = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS, conflict_handler='resolve')    
    # Global config
    global_args = parser.add_argument_group('Global arguments')
    global_args.add_argument("--config_file", type=str, default=None, help = '''Path : The config file to set parameters more 
    easily''') # default="./test_config.yaml"
    global_args.add_argument("--export_config", type=arg2path, help='''Path : If given, save the full config(combining CLI and 
                                                                    config file) at the given path''') # why str ?
    global_args.add_argument("--mode", type=str, help = '''String : A mode between "TRAINING", "EVALUATION" and "INFERENCE"''')
    global_args.add_argument("--seed", type=int, help = '''Int : If given, sets random aspect by setting random numbers
                                                        generators''')
    global_args.add_argument("--verbose", type=str, help = '''String or None: The level of verbose wanted. None, "RESTRICTED" or "FULL"''')
    return parser

def cli2config(parents=[]):
    """
    A CLI arguments reader
    Reads sys.argv and returns a dict of arguments
    Combines global args with parents args
    Args :
        parents : parents ArgumentParser
    Returns :
        cli_config : a dict of args read from CLI
    """
    parser = argparse.ArgumentParser(add_help = True, 
                                     parents = parents, 
                                     argument_default=argparse.SUPPRESS, 
                                     conflict_handler='resolve') 
    cli_args = parser.parse_args()
    cli_config = vars(cli_args)
    return cli_config

def yaml2config(file_path, parents = []):
    """
    A yaml arguments parser
    Args :
        parents : parents ArgumentParser
    Returns :
        cli_config : a dict of args parsed from CLI
    """

    config_file = read_yaml(file_path)
    
    # 1st approach : reuse cli parsing for yaml, means that we do not use a yaml file but a cli file
    '''
    parser = argparse.ArgumentParser(add_help = False, 
                                     parents = parents, 
                                     argument_default=argparse.SUPPRESS, 
                                     conflict_handler='resolve')
    
    vals = []
    for k,v in config_file.items():
        vals.append(f"--{k}")
        vals.append(f"{v}")
    print(vals)
    print("SYS",sys.argv[1:])

    file_args = parser.parse_args(vals)
    file_config = vars(file_args)

    '''
    # 2nd approach to have 2 different parsing for cli and yaml files
    #config_file = check_parse_config(config_file) 
    return config_file

def check_parse_config(config = {}, source="CLI"):
    """
    Checks and parses a given config to return appropriate classes for each attribute
    Args :
        config : the config to type check and parse 
        source : the source from where the config is config from. WIP : indicate the error the source
    Returns :
        config_parsed : a dict of args read from CLI and updated using plug_ai rules
    """
    config_parsed = dict()
    
    keys2parser = {"dataset" : str,
                   "dataset_kwargs" : arg2dict,
                   "preprocess" : str,
                   "preprocess_kwargs" : arg2dict,
                   "generate_signature" : arg2bool, 
                   "batch_size" : int,
                   "train_ratio" : float,
                   "val_ratio" : float, #Will need checkers for ratios
                   "limit_sample" : int,
                   "shuffle" : arg2bool,
                   "drop_last" : arg2bool,
                   
                   "model" : str,
                   "checkpoints_path" : str,
                   "model_kwargs" : arg2dict,
                   "use_signature" : arg2bool,
                   "res_out" : arg2bool,
                   
                   "loop" : arg2loop,
                   "loop_kwargs" : arg2dict,
                   "nb_epoch" : int,
                   "learning_rate" : float,
                   "device" : arg2device,
                   "report_log" : arg2bool,
                   "criterion" : arg2criterion,
                   "criterion_kwargs" : arg2dict,
                   "optimizer" : arg2optimizer,
                   "optimizer_kwargs" : arg2dict,
                   "execution_kwargs" : arg2dict, 

                   "config_file" : str,
                   "export_config" : arg2path,
                   "mode" : arg2mode,
                   "seed" : arg2seed,
                   "verbose" : arg2verbose}
    
    for key in config.keys():
        if key in keys2parser.keys():
            config_parsed[key] = keys2parser[key](config[key])
            
    return config_parsed
        
        

def export_yaml_config(config, yaml_template_path = default_config_file, export_path = False):
    '''
    Auto-modification of a predefined config file seems to work well with ruyaml even supporting extra arguments that were not initialy present in the file (added at the end)
    Args :
        config : a dict of args and corresponding value 
        yaml_template_path : a path to a yaml file that serves as a template for comments and args positions
        export_path : path where config file must be saved
    Return : 
        export_path : path where config file is saved
    '''

    # RUYAML class handling yaml files    
    ruyaml = ruamel.yaml.YAML()
    #typ='rt' (which is the default) inherit the safe type to manage comments. So it should be safe. Problem we appear if we want to allow loading any class/code with type='unsafe'
    ruyaml.preserve_quotes = True
    
    # Retrieve the template file
    with open(yaml_template_path) as cf:
        config_file = ruyaml.load(cf)
    
    # Update it with user config
    config_file.update(config)

    # Save config to YAML file
    with open(export_path, 'w') as file:
        ruyaml.dump(config_file, file)
        #config_used = yaml.dump(args, stream=file,default_flow_style=False, sort_keys=False) #no comments preserved with simple yaml         

    return export_path


def parse_config(parents=[]):
    """
    A parser that combines default arguments with cli arguments and config file arguments
    CLI args overwrite config_file args which overwrite default args.
    """
    # Retrieve CLI arguments
    cli_config = cli2config(parents)
    check_parse_config(config = cli_config, source = "CLI") # Only check
    
    
    # Load default config
    default_config = yaml2config(default_config_file, parents)
    check_parse_config(config = default_config, source = "Default config") # Only check
    

    # If user gave a config file, update config with config file
    config_file = dict()
    if cli_config["config_file"] is not None :
        config_file = yaml2config(cli_config["config_file"])
        check_parse_config(config = config_file, source = "Config file") # Only check
        # Add a check for config file in config file? Allow for config file cascade?
        
    # Combine default config, config file and cli
    #args = {key: value[:] for key, value in default_args.items()}
    config = default_config.copy()
    config.update(config_file)
    config.update(cli_config) #use argparse.SUPPRESS as default values for args, CAREFUL ADD DEFAULT SUPRESS ALSO IN PARENTS
    #config.update({k: v for k, v in cli_config.items() if v is not None})  # Update with cli_config if arg is not None
    
    if config["export_config"] is not None:
        export_yaml_config(config, default_config_file, config["export_config"])

    # Parsing is only done once to allow saving in YAML (specific classes cannot be saved...)
    # config = check_parse_config(config) 

    #print("\ndefault config YAML: ", default_config)
    #print("\nCLI args: ", cli_config)
    #print("\nfrom file", config_file)
    #print("\nFinal config: ", config)
    
    
    return config

