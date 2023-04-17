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
