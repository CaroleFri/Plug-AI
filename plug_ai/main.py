'''
This serves to import plug_ai as a library.
In the final version, this should not be necessary. Instead user will need to do a pip install plug_ai in order to be able to import plug_ai.
Warning : there seem to be issues when importing monai and thus plug_ai without a gpu.
'''

# Workaround until plug-ai is a package
import sys
import os
path_file = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_file+'/..')

# Imports
from datetime import datetime
from plug_ai.data.dataset import get_dataset, get_infer_dataset
from plug_ai.models import ModelManager
from plug_ai.execution import ExecManager
import torch
from plug_ai.utils.parser import parse_config


#Eventually move the main function and everything inside the "library" to have it "defined" and untouchable directly while allowing easy reuse of it
if __name__ == '__main__':
    start = datetime.now()
    print("Initialize training...")
    config = parse_config() # config are now in a dictionary => config.value is now config["value"]
    print("Arguments initialized")

    if config["mode"] == "Training":
        dataloader = get_dataset(config["dataset_dir"], batch_size=config["batch_size"],
                                   limit_sample=config["limit_sample"])
    elif config["mode"] == "Inference":
        dataloader = get_infer_dataset(config["dataset_dir"], batch_size=config["batch_size"],
                                       limit_sample=config["limit_sample"])
    print("Dataset loaded")

    model_manager = ModelManager(config)
    model = model_manager.get_model()
    print("Model loaded")

    exec_manager = ExecManager(config=config, model=model, dataloader=dataloader)
    print("Execution initialised")

    if config["mode"] == "Training":
        model = exec_manager.training()
        print("Training complete, saving model...")
        torch.save(model.state_dict(), os.path.join(config['checkpoints_path'], f'{config["model_name"]}.pt'))
        print("model saved !")

    elif config["mode"] == "Inference":
        print("Inference Mode")
        result = exec_manager.inference()

    else:
        print('Mode incorrect')

    print(">>> Complete execution in: " + str(datetime.now() - start))
