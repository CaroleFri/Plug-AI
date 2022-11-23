from .trainer import *
from .inference import *
from .evaluation import *
 
    
supported_modes = {
    'Training': Trainer,
    'Evaluation': run_eval,
    'Inference' : None
}#run_infer