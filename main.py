import argparse
from a_dataset import get_dataset
from c_model import get_model
from d_train_loop import train_loop
from monai.losses import DiceCELoss
from torch.optim import SGD
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default="/gpfsscratch/idris/sos/ssos022/Medical/Task01_BrainTumour/")
    parser.add_argument("--limit_sample", type=int)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--nb_epoch", type=int, default=1)
    parser.add_argument("--n_class", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-05)
    parser.add_argument("--device", type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()
    return args


def main(args):
    train_loader = get_dataset(args.dataset_dir, batch_size=args.batch_size, limit_sample=args.limit_sample)
    print("dataset loaded ! Loading model...")

    # example of Hyperparameter of Dynunet, to change/automatize !!
    kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    in_channels = 4
    n_class = args.n_class
    model = get_model(in_channels, n_class, kernels, strides).to(args.device)
    print("Model loaded ! Initialize training...")

    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.99, weight_decay=3e-5, nesterov=True)

    train_loop(train_loader, model, optimizer, criterion, args.nb_epoch, args.device)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Initialize training")
    args = parse_args()

    main(args)
