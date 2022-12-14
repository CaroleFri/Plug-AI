'''
This serves to import plug_ai as a library.
In the final version, this should not be necessary. Instead user will need to do a pip install plug_ai in order to be able to import plug_ai.
Warning : there seem to be issues when importing monai and thus plug_ai without a gpu.
'''

# Workaround until plug-ai is a package
import os
import sys
path_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_file+'/..')

# Imports
from datetime import datetime
import plug_ai
#from plug_ai.models import ModelManager
#from plug_ai.execution import ExecManager
#from plug_ai.utils.parser import parse_config


def main(kwargs):
    print("Plug-AI running with config:")
    [print('\t', key,':',value) for key, value in kwargs.items()]
    
    start = datetime.now()

    # HOW TO ADD AN AUTO BATCH_SIZE? Not possible before loading/estimating model size...
    # IDEA, if arg="auto" => execution manager, first do a spinup round to find params and then run real execution
    # gets all params, check if any param auto, if param auto=> determine param and fix it to new value
    #params_manager = 

    
    # Dataset_manager contains the dataset/dataloader, eventually the signature if user asked for it
    dataset_manager = plug_ai.managers.DatasetManager(dataset = kwargs["dataset"],
                                                      dataset_kwargs = kwargs["dataset_kwargs"],
                                                      preprocess = kwargs["preprocess"],
                                                      preprocess_kwargs = kwargs["preprocess_kwargs"],
                                                      mode = kwargs["mode"],
                                                      batch_size = kwargs["batch_size"],
                                                      train_ratio = kwargs["train_ratio"],
                                                      val_ratio = kwargs["val_ratio"],
                                                      limit_sample = kwargs["limit_sample"],
                                                      shuffle = kwargs["shuffle"],
                                                      drop_last = kwargs["drop_last"],
                                                      seed = kwargs["seed"], # doit permettre de fixer toutes les seed:  A généraliser
                                                      verbose = kwargs["verbose"])
    
    #generate_signature = kwargs["generate_signature"], # à potentiellement déplacer soit dans dataset_kwargs (dataset method) ou laisser gérer par manager
    
    
    # Model_manager contains the model. adapted to the dataset signature if user asked for it
    model_manager = plug_ai.managers.ModelManager(plug_dataset = dataset_manager, # à dégager
                                                  model = kwargs["model"],
                                                  device=kwargs["device"],
                                                  model_kwargs = kwargs["model_kwargs"],
                                                  mode = kwargs["mode"],
                                                  verbose = kwargs["verbose"]
                                                  )
    #                                              use_signature = kwargs["use_signature"], # a bouger dans model_kwargs
    #                                              checkpoints_path = kwargs["checkpoints_path"],
    #                                              res_out = kwargs["res_out"],

    # Execution_manager runs a training/evaluation/inference process. 
    # Some parameters such as batch_size="auto" will run first a finding params loop.
    execution_manager = plug_ai.managers.ExecutionManager(dataset_manager = dataset_manager, 
                                                          model_manager = model_manager,
                                                          loop = kwargs["loop"], # txt | callable (doit respecter un formalisme)
                                                          loop_kwargs = kwargs["loop_kwargs"],
                                                          mode = kwargs["mode"],
                                                          nb_epoch = kwargs["nb_epoch"],
                                                          learning_rate = kwargs["learning_rate"],
                                                          device = kwargs["device"],
                                                          seed = kwargs["seed"],
                                                          report_log = kwargs["report_log"],
                                                          criterion = kwargs["criterion"], # txt | callable | instance herite loss pytorch
                                                          criterion_kwargs = kwargs["criterion_kwargs"], 
                                                          optimizer = kwargs["optimizer"], # txt | callable
                                                          optimizer_kwargs = kwargs["optimizer_kwargs"],
                                                          verbose = kwargs["verbose"])

    
    # TODO, remove config export from parser and put it in execution_manager
    
    print(">>> Complete execution in: " + str(datetime.now() - start))    
    
 

   
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    
    # Args are now in a dictionnary => args.value is now args["value"]
    config = plug_ai.utils.parse_config(parents = [plug_ai.managers.DatasetManager.createParser(),
                                                   plug_ai.managers.ModelManager.createParser(),
                                                   plug_ai.managers.ExecutionManager.createParser(),
                                                   plug_ai.utils.parser.createGlobalParser()
                                                  ])
    main(config)
