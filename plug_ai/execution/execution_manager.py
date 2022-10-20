import torch
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceCELoss
from torch.optim import SGD


class ExecManager():
    """
    Class to configure and run the training and the inference
    """

    def __init__(self, config, model, dataloader=None, optimizer=None, criterion=None):
        self.config = config
        self.model = model.to(self.config["device"])

        if self.config['mode'] == 'Training':
            print("TRAINING MODE")
            self.training_loader = dataloader
            self.inference_loader = None

            if optimizer:
                self.optimizer = optimizer
            else:
                self.optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.99, weight_decay=3e-5, nesterov=True)

            if criterion:
                self.criterion = criterion
            else:
                self.criterion = DiceCELoss(to_onehot_y=True, softmax=True)

        else:
            print("INFERENCE MODE")
            self.inference_loader = dataloader

        if self.config["report_log"]:
            self.writer = SummaryWriter(f'./report_log/{self.config["model_name"]}')
            print("recording tensorboard logs")

    def inference_step(self, sample_inp):
        """
        Infer on one sample with the model
        :param sample_inp: torch.Tensor
        :return: torch.Tensor
        """

        sample_inp = sample_inp.to(self.config["device"])
        return self.model(sample_inp)

    def training_step(self, sample):
        self.optimizer.zero_grad()

        output = self.inference_step(sample["input"])
        loss = self.criterion(output, sample["label"].to(self.config["device"]))

        loss.backward()
        self.optimizer.step()

        return loss

    @torch.no_grad()
    def inference(self):
        self.model.eval()
        total_infer_step = len(self.inference_loader)
        print(f"start inference loop, {total_infer_step} steps")
        print("WARNING, the inference will probably cause an OOM")
        result = []

        for i, sample in enumerate(self.inference_loader):
            output = self.inference_step(sample["input"])
            result.append(output)  # will cause OOM probably

            print(f'Inference_step {i + 1}/{total_infer_step}')

        return result

    def training(self):
        total_train_step = len(self.training_loader)
        print(f"start training loop, {total_train_step} steps per epoch with {self.config['nb_epoch']} epoch")

        for epoch in range(self.config["nb_epoch"]):
            self.model.train()
            for i, sample in enumerate(self.training_loader):
                loss = self.training_step(sample)
                print(f'[Epoch {epoch + 1}/{self.config["nb_epoch"]} |  Step_Epoch {i + 1}/{total_train_step} | Loss {loss.item()}]')

                if self.config["report_log"]:
                    self.writer.add_scalar('Loss/train', loss.item(), i + epoch * total_train_step)

            # Evaluation here (not implemented yet)
            if self.inference_loader:
                self.inference()

            print(f"Epoch {epoch} finished")

        return self.model


def train_loop(train_loader, model, optimizer, criterion, args):
    if args["report_log"]:
        writer = SummaryWriter(f'./report_log/{args["model_name"]}')
        print("recording tensorboard logs")
    total_train_step = len(train_loader)
    print(f"start training loop, {total_train_step} steps per epoch")
    for epoch in range(args["nb_epoch"]):

        for i, x in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            inp = x["input"].to(args["device"])
            targets = x["label"].to(args["device"])

            out = model(inp)

            loss = criterion(out, targets)

            loss.backward()
            optimizer.step()

            if args["report_log"]:
                writer.add_scalar('Loss/train', loss.item(), i+epoch*total_train_step)
            print(f'[Epoch {epoch+1}/{args["nb_epoch"]} |  Step_Epoch {i+1}/{total_train_step} | Loss {loss.item()}]')

    return model
