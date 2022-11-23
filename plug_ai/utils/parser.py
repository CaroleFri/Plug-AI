import argparse
import yaml as yl
import ruamel.yaml
from .script_utils import *
import os

utils_folder = os.path.dirname(os.path.realpath(__file__))
default_config_file = os.path.join(utils_folder, "default_config.yaml")



def createGlobalParser():
    parser = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS, conflict_handler='resolve')    
    # Data information
    global_args = parser.add_argument_group('Global arguments')
    global_args.add_argument("--config_file", type=str, default=None) # default="./test_config.yaml"
    global_args.add_argument("--export_config", type=arg2path, help="test") # why str ?
    global_args.add_argument("--mode", type=str)
    global_args.add_argument("--verbose", type=str)

    return parser

def parse_cli(parents=[]):
    """
    A CLI arguments parser
    Reads sys.argv and returns a dict of arguments
    Combines global args with parents args
    Args :
        parents : parents ArgumentParser
    Returns :
        cli_config : a dict of args coming from CLI
    """
    parser = argparse.ArgumentParser(add_help = True, parents = parents, argument_default=argparse.SUPPRESS, conflict_handler='resolve') 
    # Use parents to heridate arguments from others objets (Managers)
    # Instead of a list of args, args should be splitted in each class with an init args (cf Mickael implementation)
    # Global config
    '''
    global_args = parser.add_argument_group('Global arguments')
    global_args.add_argument("--config_file", type=str, default=None) # default="./test_config.yaml"
    global_args.add_argument("--export_config", type=arg2path, help="test") # why str ?
    global_args.add_argument("--mode", type=str)
    global_args.add_argument("--verbose", type=str)
    '''
    #parser._action_groups.reverse()
    cli_args = parser.parse_args()
    cli_config = vars(cli_args)
    return cli_config

def parse_yaml_file(file_path):
    yaml = ruamel.yaml.YAML(typ='safe')
    yaml.preserve_quotes = True
    with open(file_path) as cf:
        config_file  = yaml.load(cf)
    config_file = read_yaml(file_path)
    return config_file

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
    # Load default config
    default_config = parse_yaml_file(default_config_file)

    # Retrieve CLI arguments
    cli_config = parse_cli(parents)

    # If user gave a config file, update config with config file
    config_file = dict()
    if cli_config["config_file"] is not None :
        config_file = parse_yaml_file(cli_config["config_file"])
     
    # Combine default config, config file and cli
    #args = {key: value[:] for key, value in default_args.items()}
    config = default_config.copy()
    config.update(config_file)
    config.update(cli_config) #use argparse.SUPPRESS as default values for args, CAREFUL ADD DEFAULT SUPRESS ALSO IN PARENTS
    #config.update({k: v for k, v in cli_config.items() if v is not None})  # Update with cli_config if arg is not None
    
    if config["export_config"] is not None:
        export_yaml_config(config, default_config_file, config["export_config"])

    #print("\ndefault config YAML: ", default_config)
    #print("\nCLI args: ", cli_config)
    #print("\nfrom file", config_file)
    #print("\nFinal config: ", config)
    
    
    return config


'''
# Default config has been moved to default_config.yaml in plug_ai/utils. (not with exact params)
default_config = {
    'mode': 'Training',
    'model_name': 'model_test',
    'dataset_dir': '/gpfsscratch/idris/sos/ssos022/Medical/Task01_BrainTumour/',
    'task': 'Segmentation',
    'categories': ['cat0', 'cat1', 'cat2', 'cat3'],
    'limit_sample': 'None',
    'batch_size': 2,
    'nb_epoch': 1,
    'learning_rate': '5e-05',
    'device': 'cuda',
    'random_seed': 2022,
    'verbose': 'Full',
    'export_config': False,
    'report_log': False,
    'checkpoints_path': './checkpoints',
    'model_type': 'DynUnet',
    'model_args': None,
    'model_kwargs': None,
    'res_out': False # Use the residual output of Unet (not working for now)
}
'''