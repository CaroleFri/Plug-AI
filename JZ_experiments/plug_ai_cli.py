import os
import argparse
import sys
sys.path.append('/gpfswork/rech/ibu/ssos023/Plug-AI')
import plug_ai


def main(args):
    print(args)
    print("Plug-AI CLI running :")
    dataset_manager = plug_ai.managers.DatasetManager(**args)
    model_manager = plug_ai.managers.ModelManager(dataset_manager.dataset, **args)
    execution_manager = plug_ai.optim.manager.ExecutionManager(dataset_manager.dataset, model_manager.model, **args)
    
 

   
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = plug_ai.utils.parse_args() # Args are now in a dictionnary => args.value is now args["value"]
    main(args)

    

    
    
"""
#Eventually move the main function and everything inside the "library" to have it "defined" and untouchable directly while allowing easy reuse of it
def main(args):

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
"""
    

