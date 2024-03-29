from monai.losses import DiceLoss, GeneralizedDiceLoss, GeneralizedWassersteinDiceLoss, DiceCELoss, DiceFocalLoss, GeneralizedDiceFocalLoss, FocalLoss, TverskyLoss, ContrastiveLoss
from monai.metrics import DiceMetric, MeanIoU # need to update monai
from .NonAdjLoss import NonAdjLoss

# Bellow are only added segmentation losses, all tested with brats and default loop/loss parameters
supported_criterion = {
    'Default' :  DiceCELoss,
    'DiceLoss' : DiceLoss,
    'GeneralizedDiceLoss' : GeneralizedDiceLoss,
    'DiceCELoss' : DiceCELoss,
    'DiceFocalLoss' : DiceFocalLoss,
    'GeneralizedDiceFocalLoss' : GeneralizedDiceFocalLoss,
    'FocalLoss' : FocalLoss,
    'TverskyLoss' : TverskyLoss
}




### Non-working :
## 'GeneralizedWassersteinDiceLoss' : GeneralizedWassersteinDiceLoss, => need a dist_matrix between classes to initialize
# Idea : parse the list into (Union[ndarray, Tensor]) which is the expected format for dist_matrix
## 'ContrastiveLoss' : ContrastiveLoss => Deprecated
# 'NonAdjLoss' : NonAdjLoss, => not fully automated yet

### WIP : 
## How to allow for the use of these loss which do not follow the "input->target formalism" of our loops.
# 'MaskedDiceLoss', 'MaskedLoss', 'MultiScaleLoss'  : # Issue : how to pass a different mask per data for example?
# Idea : have specific loops for specific cases?

## Because we only talked about segmentation, I only added segmentation losses, but how to tackle other usecases : reconstruction or registration losses for example

supported_metric = {
    None : None,
    'None' : None,
    'Default' :  None,
    'MeanDice' : DiceMetric,
    'MeanIoU' : MeanIoU
}
