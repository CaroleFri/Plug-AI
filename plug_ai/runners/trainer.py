import os
import torch
from ..utils.script_utils import filter_dict, print_verbose
from .evaluation import eval_loop

class Default_Trainer:
    def __init__(self, 
                 train_loader,
                 model,
                 optimizer,
                 criterion,
                 val_loader = None,  
                 metric = None,
                 lr_scheduler = None,
                 optimizer_kwargs = {},
                 lr_scheduler_kwargs = {},
                 criterion_kwargs = {},
                 metric_kwargs = {},
                 nb_epoch=2,
                 device="cuda",
                 train_step = "default",
                 step_kwargs = {},
                 verbose = "RESTRICTED",
                 report_log = False,
                 export_dir = None,                 
                ):
        #train_loop,
        #infer_loop,
        #step_,
        print("Training ...")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.metric = metric
        self.nb_epoch = nb_epoch
        self.device = device

        # WIP To change to accept other eval_loop
        self.eval_loop = eval_loop
        
        # To be moved in a selector
        self.train_step = train_step
        if train_step == "default":
            self.train_step = self.default_training_step
        self.step_kwargs = filter_dict(self.train_step, step_kwargs) #Should this check be done? Responsability is up to who?
        self.verbose = verbose
        self.report_log = report_log
        self.export_dir = export_dir
        self.model_name = "TEST_MODEL_NAME" # => to be moved in a global experiment_name used for everything saved

        # Instantiation done here but I believe it is more "standard" that the train loop receives both model, criterion and optimizer initialized for the situation B where a dev gives his own loop
        # but for a situation like criterion being a callable that returns multiple criterions to be used in the train loop, if reading was done in training loop, would be no problem with this special case.
        criterion_kwargs_filtered = filter_dict(criterion, criterion_kwargs)
        self.criterion = criterion(**criterion_kwargs_filtered).train().cuda()
        
        if self.metric != None:
            metric_kwargs_filtered = filter_dict(metric, metric_kwargs) # WIP
            self.metric = metric(metric_kwargs_filtered)

        optimizer_kwargs_filtered = filter_dict(optimizer, optimizer_kwargs)
        self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs_filtered)
        
        
        # Setup learning rate scheduler and step time
        self.lr_scheduler = lr_scheduler["scheduler"]
        self.lr_scheduler_update = lr_scheduler["scheduler_update"]
        if self.lr_scheduler is not None:
            lr_scheduler_kwargs_filtered = filter_dict(self.lr_scheduler, lr_scheduler_kwargs)
            self.lr_scheduler = self.lr_scheduler(self.optimizer, **lr_scheduler_kwargs_filtered)
        
        print_verbose("Criterion is :", self.criterion, 
                      print_lvl = "RESTRICTED", 
                      verbose_lvl = self.verbose)
        print_verbose("Metric is :", self.metric, 
                      print_lvl = "RESTRICTED", 
                      verbose_lvl = self.verbose)
        print_verbose("Optimizer is :", self.optimizer, 
                      print_lvl = "RESTRICTED", 
                      verbose_lvl = self.verbose) 
        print_verbose("Learning Rate Scheduler is :", self.lr_scheduler, 
                      print_lvl = "RESTRICTED", 
                      verbose_lvl = self.verbose)
        
        if self.report_log:
            logs_directory = os.path.join(self.export_dir, "logs",)
            self.writer = SummaryWriter(logs_directory)
            print("Recording tensorboard logs")
        
        self.run() # model name, report log,... => train_loop kwargs
        

    def run(self):
        models_directory = os.path.join(self.export_dir, "models")
        os.makedirs(models_directory, exist_ok=True)

        total_train_step = len(self.train_loader)
        print(f"start training loop, {total_train_step} steps per epoch")
        
        best_metric = 0  # This can be either 'inf' or '-inf' depending on whether the metric should be minimized or maximized.
        best_val_loss = float('inf')  # Initialize it as infinity.
        
        for epoch in range(self.nb_epoch):
            self.model.train()
            
            for i, sample in enumerate(self.train_loader):
                loss = self.train_step(sample)
                
                print(f'[Epoch {epoch+1}/{self.nb_epoch} |  Step_Epoch {i+1}/{total_train_step} | Loss {loss.item()}]')

                if self.report_log:
                    self.writer.add_scalar('Loss/train', loss.item(), i + epoch * total_train_step)
            
            if self.val_loader is not None:
                val_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    for sample in self.val_loader:
                        output = self.inference_step(sample['input'])
                        loss = self.criterion(output, sample['label'].to(self.device))
                        val_loss += loss.item()
                val_loss /= len(self.val_loader) 
                
                # Track the best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_best_path = os.path.join(models_directory, f"model_best_{epoch}_val_loss_{best_val_loss:.3f}.pt")
                    torch.save(self.model.state_dict(), model_best_path)
                    print(f'Best model saved at epoch {epoch} with validation loss {best_val_loss:.3f}')                    
                          
            
                # Evaluation here 
                if self.metric is not None:
                    with torch.no_grad():
                        self.eval_metric = self.eval_loop(self)
                        
                self.model.train()

            # Learning Rate Update using lr_scheduler
            if self.lr_scheduler_update == "epoch":
                #lr_scheduler_step_kwargs = filter_dict(self.lr_scheduler.step, {"metrics":self.eval_metric}) #Add every parameter that might be in lr_scheduler.step()
                self.lr_scheduler.step()#lr_scheduler_step_kwargs
            
            print(f"Epoch {epoch} finished")

            # Checkpointing at each epoch
            model_epoch_path = os.path.join(models_directory,f"model_epoch_{epoch}.pt")
            torch.save(self.model.state_dict(), model_epoch_path)

            
            
        #model_best_path = os.path.join(models_directory,"model_best.pt")
        #torch.save(self.model.state_dict(), model_best_path)
        
        # Save the final model
        model_last_path = os.path.join(models_directory,"model_last.pt")
        torch.save(self.model.state_dict(), model_last_path)        
    
    
    def default_training_step(self, sample):
        """
        :param sample:
        :return:
        """
        self.optimizer.zero_grad()
        output = self.inference_step(sample["input"])        
        loss = self.criterion(output, sample["label"].to(self.device))

        loss.backward()
        self.optimizer.step()
        
        if self.lr_scheduler_update == "batch":
            #lr_scheduler_step_kwargs = filter_dict(self.lr_scheduler.step, {"metrics":self.eval_metric}) #Add every parameter that might be in lr_scheduler.step()
            self.lr_scheduler.step()
        
        return loss

    def inference_step(self, sample_inp):
        """
        Infer on one sample with the model
        :param sample_inp: torch.Tensor
        :return: torch.Tensor
        """
        sample_inp = sample_inp.to(self.device)
        return self.model(sample_inp)

    # @torch.no_grad()
    # def eval_loop(self):
    #     self.model.train()
    #     self.metric.reset()
    #     total_eval_step = len(self.val_loader)
    #     for i, sample in enumerate(self.val_loader):
    #         pred = self.inference_step(sample["input"])
    #         pred_shape = pred.shape
            
    #         # choose the class with the highest probability and reshape the tensor for the loss
    #         target = torch.nn.functional.one_hot(
    #             torch.argmax(pred, 1),
    #             num_classes=pred_shape[1] # get number of class from output channel
    #         )
    #         target = target.view(pred_shape).cpu() # reshape for computing metric

    #         # Better on cpu to avoid cuda out of memory
    #         eval_score = self.metric(target, sample["label"])

    #         print(f'[Step_Eval {i+1}/{total_eval_step}]')

    #         # if self.report_log:
    #         #     self.writer.add_scalar('Loss/train', loss.item(), i + epoch * total_train_step)

    #     print(f"Evaluation score for this epoch: {self.metric.aggregate()}")

        
#### same selection for train for loop and infer for loop
#### same for inference step
'''
if isinstance(train_step, callable):
    self.output = train_step(**execution_kwargs)
else:
    self.train_step = plugai.train_step
'''