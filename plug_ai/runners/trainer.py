import torch

class Trainer:
    def __init__(self, 
                 train_loader,
                 model,
                 optimizer=None,
                 optimizer_kwargs = {},
                 criterion=None,
                 criterion_kwargs = {},
                 nb_epoch=2,
                 device="cuda",
                 !!!!!!!!!!!!!
                train_loop,
                infer_loop,
                step_,
                ):
        
        print("Training ...")
        self.train_loader = train_loader
        self.model = model
        self.nb_epoch = nb_epoch
        self.device = device

        
        train_loop(self.train_loader, self.model, self.optimizer, self.criterion, self.nb_epoch, self.device)

    @staticmethod
    def run(train_loader, model, optimizer, criterion, nb_epoch, device, train_step, report_log):
        total_train_step = len(train_loader)
        print(f"start training loop, {total_train_step} steps per epoch")
        model.train()
        for epoch in range(nb_epoch):
            for i, x in enumerate(train_loader):
                loss = self.training_step(sample)

                optimizer.zero_grad()

                inp = x["input"].to(device)
                targets = x["label"].to(device)

                out = model(inp)
                out = torch.unbind(out, dim=1)
                loss = criterion(out[0], targets)

                loss.backward()
                optimizer.step()

                print(f'[Epoch {epoch+1}/{nb_epoch} |  Step_Epoch {i+1}/{total_train_step} | Loss {loss.item()}]')


                

                print(f'[Epoch {epoch + 1}/{self.nb_epoch} |  Step_Epoch {i + 1}/{total_train_step} | Loss {loss.item()}]')

                if report_log:
                    self.writer.add_scalar('Loss/train', loss.item(), i + epoch * total_train_step)

            # Evaluation here (not implemented yet)
            #if self.inference_loader:
            #    self.inference()

            print(f"Epoch {epoch} finished")

        return model