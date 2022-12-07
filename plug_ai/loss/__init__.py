from monai.losses import DiceCELoss

supported_criterion = {
    'Default' :  DiceCELoss,
    'DiceCELoss' : DiceCELoss
}
