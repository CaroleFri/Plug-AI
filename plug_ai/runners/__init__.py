from .trainer import Default_Trainer
from .inference import *
from .evaluation import *
 
'''    
supported_modes = {
    'Training': Trainer,
    'Evaluation': run_eval,
    'Inference' : None
}#run_infer
'''

# Should we split train, infer, eval loops in different lists?
supported_loop = {
    'Default_train' : Default_Trainer
}

# Should we split train, infer, eval steps in different lists?
supported_step = {
    'Default_step' : Default_Trainer.default_training_step
}