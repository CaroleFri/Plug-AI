import torch
from torch.utils.tensorboard import SummaryWriter


@torch.no_grad()
def infer_loop(infer_loader, model, args):

    total_infer_step = len(infer_loader)
    print(f"start inference loop, {total_infer_step} steps per epoch")
    result = []
    for i, x in enumerate(infer_loader):
        inp = x["input"].to(args["device"])

        out = model(inp)
        out = torch.unbind(out, dim=1)
        result.append(out[0].cpu()) # will cause OOM probably

        print(f'Inference_step {i+1}/{total_infer_step}')

    return result

