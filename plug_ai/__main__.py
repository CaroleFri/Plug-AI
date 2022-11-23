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
    start = datetime.now()
    
    # gets all params, check if any param auto, if param auto=> determine param and fix it to new value
    #params_manager = 
    
    # Dataset_manager contains the dataset/dataloader, eventually the signature if user asked for it
    dataset_manager = plug_ai.managers.DatasetManager(dataset = kwargs["dataset"],
                                                      dataset_kwargs = kwargs["dataset_kwargs"],
                                                      preprocess = kwargs["preprocess"],
                                                      mode = kwargs["mode"],
                                                      generate_signature = kwargs["generate_signature"],
                                                      limit_sample = kwargs["limit_sample"],
                                                      batch_size = kwargs["batch_size"],
                                                      shuffle = kwargs["shuffle"],
                                                      drop_last = kwargs["drop_last"],
                                                      verbose = kwargs["verbose"])
    
    # Model_manager contains the model. adapted to the dataset signature if user asked for it
    model_manager = plug_ai.managers.ModelManager(plug_dataset = dataset_manager,
                                                  model = kwargs["model"],
                                                  checkpoints_path = kwargs["checkpoints_path"],
                                                  model_kwargs = kwargs["model_kwargs"],
                                                  mode = kwargs["mode"],
                                                  use_signature = kwargs["use_signature"],
                                                  res_out = kwargs["res_out"])
    
    # Execution_manager runs a training/evaluation/inference process. 
    # Some parameters such as batch_size="auto" will run first a finding params loop.
    execution_manager = plug_ai.managers.ExecutionManager(dataset_manager = dataset_manager, 
                                                          model_manager = model_manager,
                                                          mode = kwargs["mode"],
                                                          nb_epoch = kwargs["nb_epoch"],
                                                          learning_rate = kwargs["learning_rate"],
                                                          device = kwargs["device"],
                                                          random_seed = kwargs["random_seed"],
                                                          report_log = kwargs["report_log"],
                                                          execution_kwargs = kwargs["execution_kwargs"])
    
    # TODO, remove config export from parser and put it in execution_manager
    
    print(">>> Complete execution in: " + str(datetime.now() - start))    
    
 

   
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Plug-AI CLI running :")
    
    # Args are now in a dictionnary => args.value is now args["value"]
    config = plug_ai.utils.parse_config(parents = [plug_ai.managers.DatasetManager.createParser(),
                                                   plug_ai.managers.ModelManager.createParser(),
                                                   plug_ai.managers.ExecutionManager.createParser(),
                                                   plug_ai.utils.parser.createGlobalParser()
                                                  ]) 
    main(config)
