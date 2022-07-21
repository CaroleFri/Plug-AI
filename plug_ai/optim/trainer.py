import torch


def train_loop(train_loader, model, optimizer, criterion, nb_epoch, device):
    total_train_step = len(train_loader)
    print(f"start training loop, {total_train_step} steps per epoch")
    for epoch in range(nb_epoch):
        for i, x in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            inp = x["input"].to(device)
            targets = x["label"].to(device)

            out = model(inp)
            out = torch.unbind(out, dim=1)
            loss = criterion(out[0], targets)

            loss.backward()
            optimizer.step()

            print(f'[Epoch {epoch+1}/{nb_epoch} |  Step_Epoch {i+1}/{total_train_step} | Loss {loss.item()}]')
