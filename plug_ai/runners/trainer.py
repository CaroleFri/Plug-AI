import torch
from ..utils.script_utils import filter_dict, print_verbose

class Default_Trainer:
    def __init__(self, 
                 train_loader,
                 model,
                 optimizer,
                 criterion,
                 lr_scheduler,
                 optimizer_kwargs = {},
                 criterion_kwargs = {},
                 lr_scheduler_kwargs = {},
                 nb_epoch=2,
                 device="cuda",
                 train_step = "default",
                 step_kwargs = {},
                 verbose = "RESTRICTED",
                 report_log = False
                ):
        #train_loop,
        #infer_loop,
        #step_,
        print("Training ...")
        self.train_loader = train_loader
        self.model = model
        self.nb_epoch = nb_epoch
        self.device = device
        
        # To be moved in a selector
        self.train_step = train_step
        if train_step == "default":
            self.train_step = self.default_training_step
        self.step_kwargs = filter_dict(self.train_step, step_kwargs) #Should this check be done? Responsability is up to who?
        self.verbose = verbose
        self.report_log = report_log
        self.model_name = "TEST_MODEL_NAME" # => to be moved in a global experiment_name used for everything saved

        # Instantiation done here but I believe it is more "standard" that the train loop receives both model, criterion and optimizer initialized for the situation B where a dev gives his own loop
        # but for a situation like criterion being a callable that returns multiple criterions to be used in the train loop, if reading was done in training loop, would be no problem with this special case.
        criterion_kwargs_filtered = filter_dict(criterion, criterion_kwargs)
        self.criterion = criterion(**criterion_kwargs_filtered)

        optimizer_kwargs_filtered = filter_dict(optimizer, optimizer_kwargs)
        self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs_filtered)
        
        
        
        print_verbose("Criterion is :", self.criterion, 
                      print_lvl = "RESTRICTED", 
                      verbose_lvl = self.verbose)
        print_verbose("Optimizer is :", self.optimizer, 
                      print_lvl = "RESTRICTED", 
                      verbose_lvl = self.verbose)        
        
        if self.report_log:
            self.writer = SummaryWriter(f'./report_log/{self.model_name}')
            print("recording tensorboard logs")
        
        self.run()# model name, report log,... => train_loop kwargs
        


    def run(self):
        total_train_step = len(self.train_loader)
        print(f"start training loop, {total_train_step} steps per epoch")
        

        
        for epoch in range(self.nb_epoch):
            self.model.train()
            
            for i, sample in enumerate(self.train_loader):
                
                loss = self.train_step(sample)

                print(f'[Epoch {epoch+1}/{self.nb_epoch} |  Step_Epoch {i+1}/{total_train_step} | Loss {loss.item()}]')

                if self.report_log:
                    self.writer.add_scalar('Loss/train', loss.item(), i + epoch * total_train_step)

            # Evaluation here (not implemented yet)
            #if self.inference_loader:
            #    self.inference()

            print(f"Epoch {epoch} finished")

        
    
    
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

        return loss

    def inference_step(self, sample_inp):
        """
        Infer on one sample with the model
        :param sample_inp: torch.Tensor
        :return: torch.Tensor
        """

        sample_inp = sample_inp.to(self.device)
        return self.model(sample_inp)

#### same selection for train for loop and infer for loop
#### same for inference step
'''
if isinstance(train_step, callable):
    self.output = train_step(**execution_kwargs)
else:
    self.train_step = plugai.train_step
'''
