import torch
from torch.utils.tensorboard import SummaryWriter
import os
import torch
from ..utils.script_utils import filter_dict, print_verbose
from .evaluation import eval_loop
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import write_nifti

class Inferer_SW:
    def __init__(self, 
                 infer_loader, 
                 model,
                 infer_step="default",
                 step_kwargs={},
                 device="cuda",
                 export_dir = None,
                ):
        print("Inference ongoing ...")
        self.infer_loader = infer_loader
        self.model = model
        self.device = device
        self.export_dir = export_dir
        
        self.run()
        
        
    def run(self):
        # Set the output directory
        predictions_directory = os.path.join(self.export_dir, "Predictions")
        os.makedirs(predictions_directory, exist_ok=True)
        
        total_infer_step = len(self.infer_loader)
        
        # Put the model in evaluation mode
        self.model.eval()    
        
        with torch.no_grad():        
            for i, batch in enumerate(self.infer_loader):            
                # Move data to Device
                batch["input"] = batch["input"].to(self.device)

                # Run Inference
                outputs = self.infer_sw(batch["input"], self.model, roi_size=(240, 240, 160), sw_batch_size=1,overlap=0.5,post_transforms="Default")

                # Save predictions to nifti files
                for j, sample in enumerate(outputs):
                    # Get data_id for each sample
                    data_id = batch["data_id"][j]
                    # Define path to sample prediction
                    output_file_path = os.path.join(predictions_directory, f'{data_id}_prediction.nii.gz')
                    # Save the output as NIfTI file
                    write_nifti(sample, output_file_path)


    def infer_sw(self, input, predictor, roi_size=(240, 240, 160), sw_batch_size=1,overlap=0.5,post_transforms="Default"):
        if post_transforms == "Default":
            post_transforms = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
            
        outputs = sliding_window_inference(inputs=input,
                                           roi_size=roi_size,
                                           sw_batch_size=sw_batch_size,
                                           predictor=predictor,
                                           overlap=overlap).as_tensor()
        outputs =  decollate_batch(outputs)
        outputs = [post_transforms(i) for i in outputs]
        return outputs
    
        
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
