# Imports
import os
from datetime import datetime
import plug_ai


def main(kwargs):
    '''
    Placeholder
    '''
    
    print("Plug-AI given kwargs:")
    [print('\t', key,':',value) for key, value in kwargs.items()]
    
    start = datetime.now()
    
    # Dataset_manager contains the dataset/dataloader and data related parameters
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
                                                      seed = kwargs["seed"], # doit permettre de fixer toutes les seed:  WIP A généraliser
                                                      verbose = kwargs["verbose"],
                                                      export_dir = kwargs["export_dir"])
    
    
    # Model_manager contains the model and model related parameters
    model_manager = plug_ai.managers.ModelManager(plug_dataset = dataset_manager,
                                                  model = kwargs["model"],
                                                  device=kwargs["device"],
                                                  model_kwargs = kwargs["model_kwargs"],
                                                  model_weights_path = kwargs["model_weights_path"],
                                                  mode = kwargs["mode"],
                                                  verbose = kwargs["verbose"],
                                                  export_dir = kwargs["export_dir"])

                                                  
                                                  
    # Execution_manager runs a training/evaluation/inference process. 
    execution_manager = plug_ai.managers.ExecutionManager(dataset_manager = dataset_manager, 
                                                          model_manager = model_manager,
                                                          loop = kwargs["loop"], # txt | callable (doit respecter un formalisme)
                                                          loop_kwargs = kwargs["loop_kwargs"],
                                                          mode = kwargs["mode"],
                                                          nb_epoch = kwargs["nb_epoch"],
                                                          device = kwargs["device"],
                                                          seed = kwargs["seed"],
                                                          report_log = kwargs["report_log"],
                                                          criterion = kwargs["criterion"], # txt | callable | instance herite loss pytorch
                                                          criterion_kwargs = kwargs["criterion_kwargs"], 
                                                          metric = kwargs["metric"],
                                                          metric_kwargs = kwargs["metric_kwargs"],
                                                          optimizer = kwargs["optimizer"], # txt | callable
                                                          optimizer_kwargs = kwargs["optimizer_kwargs"],
                                                          lr_scheduler = kwargs["lr_scheduler"],
                                                          lr_scheduler_kwargs = kwargs["lr_scheduler_kwargs"],
                                                          verbose = kwargs["verbose"],
                                                          export_dir = kwargs["export_dir"])

    
    print(">>> Complete execution in: " + str(datetime.now() - start))    
    

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    '''
    Plug-AI main application
    Placeholder for detailed documentation
    '''
    
    config = plug_ai.utils.parse_config(parents = [plug_ai.managers.DatasetManager.createParser(),
                                                   plug_ai.managers.ModelManager.createParser(),
                                                   plug_ai.managers.ExecutionManager.createParser(),
                                                   plug_ai.utils.parser.createGlobalParser()
                                                  ])
    main(config)

