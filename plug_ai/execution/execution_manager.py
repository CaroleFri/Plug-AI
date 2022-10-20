import torch
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceCELoss
from torch.optim import SGD


class ExecManager():
    """
    Class to configure and run the training and the inference
    """

    def __init__(self,  model, dataloader=None, optimizer=None, criterion=None, mode='Training', device="cpu",
                 nb_epoch=1, report_log=False, model_name="tes_model"):
        """

        :param model:
        :param dataloader:
        :param optimizer:
        :param criterion:
        :param mode:
        :param device:
        :param nb_epoch:
        :param report_log:
        :param model_name:
        """
        self.device = device
        self.nb_epoch = nb_epoch
        self.report_log = report_log
        self.model_name = model_name
        self.model = model.to(self.device)

        if mode == 'Training':
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

        if self.report_log:
            self.writer = SummaryWriter(f'./report_log/{self.model_name}')
            print("recording tensorboard logs")

    def inference_step(self, sample_inp):
        """
        Infer on one sample with the model
        :param sample_inp: torch.Tensor
        :return: torch.Tensor
        """

        sample_inp = sample_inp.to(self.device)
        return self.model(sample_inp)

    def training_step(self, sample):
        """

        :param sample:
        :return:
        """
        self.optimizer.zero_grad()

        output = self.inference_step(sample["input"])
        loss = self.criterion(output, sample["label"].to(self.device))

        loss.backward()
        self.optimizer.step()

        return loss

    @torch.no_grad()
    def inference(self):
        """

        :return:
        """
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
        """

        :return:
        """
        total_train_step = len(self.training_loader)
        print(f"start training loop, {total_train_step} steps per epoch with {self.nb_epoch} epoch")

        for epoch in range(self.nb_epoch):
            self.model.train()
            for i, sample in enumerate(self.training_loader):
                loss = self.training_step(sample)
                print(f'[Epoch {epoch + 1}/{self.nb_epoch} |  Step_Epoch {i + 1}/{total_train_step} | Loss {loss.item()}]')

                if self.report_log:
                    self.writer.add_scalar('Loss/train', loss.item(), i + epoch * total_train_step)

            # Evaluation here (not implemented yet)
            if self.inference_loader:
                self.inference()

            print(f"Epoch {epoch} finished")

        return self.model
