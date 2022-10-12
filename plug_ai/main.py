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
from plug_ai.data.dataset import get_dataset, get_infer_dataset
from plug_ai.models.DynUNet import PlugDynUNet
from plug_ai.optim.trainer import train_loop
from plug_ai.infer.inference import infer_loop
from monai.losses import DiceCELoss
from torch.optim import SGD
import torch
from plug_ai.utils.parser import parse_args


#Eventually move the main function and everything inside the "library" to have it "defined" and untouchable directly while allowing easy reuse of it
def train(args):

    train_loader = get_dataset(args["dataset_dir"], batch_size=args["batch_size"], limit_sample=args["limit_sample"])
    print("dataset loaded ! Loading checkpoints...")

    # Dynunet is automatically choose but we'll change that when we develop the model manager
    model = PlugDynUNet(args['in_channels'], args["n_class"], args['dynunet_kernels'], args['dynunet_strides']).to(args["device"])
    print("Model loaded ! Initialize training...")

    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.99, weight_decay=3e-5, nesterov=True)

    model = train_loop(train_loader, model, optimizer, criterion, args)
    print("Training complete, saving model...")
    torch.save(model.state_dict(), os.path.join(args['checkpoints_path'], f'{args["model_name"]}.pt'))
    print("model saved !")


def infer(args):
    infer_loader = get_infer_dataset(args["dataset_dir"], batch_size=args["batch_size"], limit_sample=args["limit_sample"])
    print("dataset loaded ! Loading checkpoints...")

    # example of Hyperparameter of Dynunet, to change/automatize !!
    model = PlugDynUNet(args['in_channels'], args["n_class"], args['dynunet_kernels'], args['dynunet_strides']).to(args["device"])
    model.load_state_dict(torch.load(os.path.join(args['checkpoints_path'], f'{args["model_name"]}.pt')))
    model.eval()
    print("Model loaded ! Initialize inferance...")

    # We have to think what we'll do with that
    result = infer_loop(infer_loader, model, args)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Initialize training...")
    args = parse_args() # Args are now in a dictionnary => args.value is now args["value"]
    print("arguments loaded")

    if args["mode"] == "Training":
        train(args)
    elif args["mode"] == "Inference":
        infer(args)
    else:
        print('Mode incorrect')

