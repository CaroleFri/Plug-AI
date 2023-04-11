from torch.optim import Adadelta, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, LBFGS, NAdam, RAdam, RMSprop, Rprop, SGD
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR, StepLR, MultiStepLR, ConstantLR, LinearLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts


supported_optimizer = {
    'Default' : SGD,
    'Adadelta' : Adadelta,
    'Adagrad' : Adagrad,
    'Adam' : Adam,
    'AdamW' : AdamW,
    'Adamax' : Adamax,
    'ASGD' : ASGD,
    'NAdam' : NAdam,
    'RAdam' : RAdam,
    'RMSprop' : RMSprop,
    'Rprop' : Rprop,
    'SGD' : SGD    
}
## WIP
#    'SparseAdam' : SparseAdam, => Not added because not tested. Need of sparse tensors for it
#     'LBFGS' : LBFGS, => need a special optimizer.step(closure) in the loop



supported_lr_scheduler_old = {
    None : None,
    'None' : None,
    'Default' : None,
    'StepLR' : StepLR,
    'MultiStepLR' : MultiStepLR,
    'ConstantLR' : ConstantLR,
    'LinearLR' : LinearLR,
    'ExponentialLR' : ExponentialLR,
    'CosineAnnealingLR' : CosineAnnealingLR,
    'ReduceLROnPlateau' : ReduceLROnPlateau,
    'CyclicLR' : CyclicLR,
    'OneCycleLR' : OneCycleLR,
    'CosineAnnealingWarmRestarts' : CosineAnnealingWarmRestarts
}

supported_lr_scheduler = {
    None : {'scheduler':None, 'scheduler_update':None},
    'None' : {'scheduler':None, 'scheduler_update':None},
    'Default' : {'scheduler':None, 'scheduler_update':None},
    'StepLR' : {'scheduler':StepLR, 'scheduler_update':"epoch"},
    'MultiStepLR' : {'scheduler':MultiStepLR, 'scheduler_update':"epoch"},
    'ConstantLR' : {'scheduler':ConstantLR, 'scheduler_update':"epoch"},
    'LinearLR' : {'scheduler':LinearLR, 'scheduler_update':"epoch"},
    'ExponentialLR' : {'scheduler':ExponentialLR, 'scheduler_update':"epoch"},
    'CosineAnnealingLR' : {'scheduler':CosineAnnealingLR, 'scheduler_update':"epoch"},
    'ReduceLROnPlateau' : {'scheduler':ReduceLROnPlateau, 'scheduler_update':"epoch"},
    'CyclicLR' : {'scheduler':CyclicLR, 'scheduler_update':"batch"},
    'OneCycleLR' : {'scheduler':OneCycleLR, 'scheduler_update':"batch"},
    'CosineAnnealingWarmRestarts' : {'scheduler':CosineAnnealingWarmRestarts, 'scheduler_update':"epoch"},
}
# In order to add a scheduler : add an item to the suppoter_lr_scheduler.
# Item structure : {'scheduler':callable that instantiate the scheduler, 'scheduler_update': where the scheduler.step() is supposed to be positionned (for now only epoch/batch options)}
# To add another position, trainer must be adapted.
# Some LR_schedulers expects a specific parameter in steps. For example ReduceLROnPlateau needs a "metrics". In the trainer, put every 

## WIP 
# lr_scheduler.ChainedScheduler => nested dictionnaries to give multive schedulers in a chain?
# lr_scheduler.SequentialLR => same as ChainedScheduler
# lr_scheduler.PolynomialLR => cannot import it, available from 1.13+
# 'LambdaLR' : LambdaLR => how to allow for a lambda given by config file? Dev only compatibility?
# 'MultiplicativeLR' : MultiplicativeLR, #Function expects a lambda, risky injection from config file. Dev only compatibility?

