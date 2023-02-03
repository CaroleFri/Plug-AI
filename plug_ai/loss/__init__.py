from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MeanIoU # need to update monai

supported_criterion = {
    'Default' :  DiceCELoss,
    'DiceCELoss' : DiceCELoss
}

supported_metric = {
    'Default' :  DiceMetric,
    'MeanDice' : DiceMetric,
    'MeanIoU' : MeanIoU
}
