'''
This serves to import plug_ai as a library.
In the final version, this should not be necessary. Instead user will need to do a pip install plug_ai in order to be able to import plug_ai.
Warning : there seem to be issues when importing monai and thus plug_ai without a gpu.
'''
import sys
sys.path.append('../')

# Imports
from plug_ai.data.dataset import get_dataset
from plug_ai.models.DynUNet import get_model
from plug_ai.optim.trainer import train_loop
from monai.losses import DiceCELoss
from torch.optim import SGD
import torch
from plug_ai.utils.parser import parse_args


#Eventually move the main function and everything inside the "library" to have it "defined" and untouchable directly while allowing easy reuse of it
def main(args):
    print("Initialize training")

    train_loader = get_dataset(args["dataset_dir"], batch_size=args["batch_size"], limit_sample=args["limit_sample"])
    print("dataset loaded ! Loading model...")

    # example of Hyperparameter of Dynunet, to change/automatize !!
    kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    in_channels = 4
    n_class = args["n_class"]
    model = get_model(in_channels, n_class, kernels, strides).to(args["device"])
    print("Model loaded ! Initialize training...")

    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.99, weight_decay=3e-5, nesterov=True)

    train_loop(train_loader, model, optimizer, criterion, args["nb_epoch"], args["device"])

    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parse_args() # Args are now in a dictionnary => args.value is now args["value"]
    main(args)
