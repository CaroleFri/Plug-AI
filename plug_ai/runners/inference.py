import torch
from torch.utils.tensorboard import SummaryWriter

class Infererer:
    def __init__(self, train_loader, model, optimizer, criterion, nb_epoch, device):
        print("Inference ...")
        self.train_loader = train_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.nb_epoch = nb_epoch
        self.device = device
        
        run(self.train_loader, self.model, self.optimizer, self.criterion, self.nb_epoch, self.device)

    @staticmethod
    def run(train_loader, model, optimizer, criterion, args):
        if args["report"]:
            writer = SummaryWriter(f'./report/args["model_name"]')
        total_train_step = len(train_loader)
        print(f"start training loop, {total_train_step} steps per epoch")
        for epoch in range(args["nb_epoch"]):
            for i, x in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()

                inp = x["input"].to(args["device"])
                targets = x["label"].to(args["device"])

                out = model(inp)
                out = torch.unbind(out, dim=1)
                loss = criterion(out[0], targets)

                loss.backward()
                optimizer.step()

                if args["report"]:
                    writer.add_scalar('Loss/train', loss.item(), i+epoch*total_train_step)
                print(f'[Epoch {epoch+1}/{args["nb_epoch"]} |  Step_Epoch {i+1}/{total_train_step} | Loss {loss.item()}]')

        return model



def train_loop(train_loader, model, optimizer, criterion, args):
    if args["report"]:
        writer = SummaryWriter(f'./report/args["model_name"]')
    total_train_step = len(train_loader)
    print(f"start training loop, {total_train_step} steps per epoch")
    for epoch in range(args["nb_epoch"]):
        for i, x in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            inp = x["input"].to(args["device"])
            targets = x["label"].to(args["device"])

            out = model(inp)
            out = torch.unbind(out, dim=1)
            loss = criterion(out[0], targets)

            loss.backward()
            optimizer.step()

            if args["report"]:
                writer.add_scalar('Loss/train', loss.item(), i+epoch*total_train_step)
            print(f'[Epoch {epoch+1}/{args["nb_epoch"]} |  Step_Epoch {i+1}/{total_train_step} | Loss {loss.item()}]')

    return model
