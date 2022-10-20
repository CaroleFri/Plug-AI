import argparse
import yaml as yl
import ruamel.yaml
from .script_utils import *

'''
For now we let default be defined here.
Later we will move default parameters localy corresponding to each class and import them at loading time
'''
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


def parse_config():
    """
    A parser that combines default arguments with cli arguments and config file arguments
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./test_config.yaml")
    # Global config
    parser.add_argument("--mode", type=str) # Training, Evaluation, Inference,
    parser.add_argument("--export_config", type=arg2bool, help="test") # why str ?
    parser.add_argument("--model_name", type=str)
    # Data information
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--task", type=str, default="THIS IS A TEST")
    parser.add_argument("--categories", type=list)
    parser.add_argument("--limit_sample", type=int)
    # Training information
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--nb_epoch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", type=arg2device)
    parser.add_argument("--report_log", type=arg2bool) # why str ?
    parser.add_argument("--checkpoints_path", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_args", type=list)
    parser.add_argument("--model_kwargs", type=list)

    cli_config = parser.parse_args()
    cli_config = vars(cli_config)
    
    #args = {key: value[:] for key, value in default_args.items()}
    config = dict(default_config)

    if cli_config["config_file"] is not None :
        yaml = ruamel.yaml.YAML(typ='safe')
        yaml.preserve_quotes = True
        with open(cli_config["config_file"]) as cf:
            config_file  = yaml.load(cf)
        config_file = read_yaml(cli_config["config_file"])
        config.update(config_file)
    #args.update(cli_args)
    config.update({k: v for k, v in cli_config.items() if v is not None})  # Update with cli_config if arg is not None, alternative use argparse.SUPPRESS
    
    print("Final config:", config)
    
    
    '''
    Auto-modification of a predefined config file seems to work well with ruyaml even supporting extra arguments that were not initialy present in the file (added at the end)
    '''
    if config["export_config"]:
        ruyaml = ruamel.yaml.YAML()#typ='safe'
        ruyaml.preserve_quotes = True
        with open(cli_config["config_file"]) as cf:
            config_file = ruyaml.load(cf)
        print(config_file)
        # Problème ici, j'ai fait un changement mais c'est bizarre ................................
        print(config)
        config_file.update(config)
        print(config)
        # .........................................................................................
        with open('config_saved.yaml', 'w') as file:
             ruyaml.dump(config_file, file)
        #with open('config_used.yaml', 'w') as file:
            #config_used = yaml.dump(args, stream=file,default_flow_style=False, sort_keys=False)        
    
    return config



'''
    parser.add_argument("--mode", type=str, default="Training") # Training, Evaluation, Inference,
    # Global config
<<<<<<< HEAD
    parser.add_argument("--config_file", type=str, default="/gpfsdswork/projects/idris/sos/ssos023/Projects/Plug-AI/Plug-AI/JZ_experiments/config_BrainTumour.yaml")
=======
    parser.add_argument("--config_file", type=str, default="/gpfsdswork/projects/idris/sos/ssos023/Projects/Plug-AI/Plug-AI/JZ_experiments/config.yaml")
>>>>>>> 42851b017d6539f35b9d56bfeda121d1ce05e228
    parser.add_argument("--export_config", type=str, default="/gpfsdswork/projects/idris/sos/ssos023/Projects/Plug-AI/Plug-AI/JZ_experiments/config_export.yaml")
    # Data information
    parser.add_argument("--dataset_dir", type=str, default="/gpfsscratch/idris/sos/ssos022/Medical/Task01_BrainTumour/")
    parser.add_argument("--dataset_type", type=str, default="Segmentation")
    # Training information
    parser.add_argument("--limit_sample", type=int)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--nb_epoch", type=int, default=1)
    parser.add_argument("--n_class", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-05)
    parser.add_argument("--device", type=arg2device, default="cuda")
'''

