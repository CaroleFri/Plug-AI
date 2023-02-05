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



supported_lr_scheduler = {
    'None' : None,
    'Default' : LambdaLR,
    'LambdaLR' : LambdaLR,
    'MultiplicativeLR' : MultiplicativeLR,
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
## WIP 
# lr_scheduler.ChainedScheduler => nested dictionnaries to give multive schedulers in a chain?
# lr_scheduler.SequentialLR => same as ChainedScheduler
# lr_scheduler.PolynomialLR => cannot import it, available from 1.13+