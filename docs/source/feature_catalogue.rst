
Feature Catalogue
=================

Plug-AI allows you to configure a complete pipeline using options from a predefined catalogue.
You can independantly select and configure:

- a dataset and dataloader
- a model to train or use
- an optimizer
- a learning rate scheduler
- a criterion (loss)
- a metric for evaluation
- a pipeline (training, evaluation, inference)

Datasets and dataloader
########

Models
######
- DynUnet
- unetr
- ModSegNet

nnU-Net is also available but follows a special pipeline non-compatible with the rest of the options.

Optimizers
##########
By default, the optimizer used is SGD but you can decide to use any of these following optimizers:

- Adadelta,
- Adagrad,
- Adam,
- AdamW,
- Adamax,
- ASGD,
- NAdam,
- RAdam,
- RMSprop,
- Rprop,
- SGD    

Learning rate schedulers
########################
By default, the learning rate is static and no learning rate scheduler is used but you can decide to use any of the following schedulers:

- StepLR
- MultiStepLR
- ConstantLR
- LinearLR
- ExponentialLR
- CosineAnnealingLR
- ReduceLROnPlateau
- CyclicLR
- OneCycleLR
- CosineAnnealingWarmRestarts

Criterions (loss)
#################
The available loss are:
- DiceLoss
- GeneralizedDiceLoss
- DiceCELoss
- DiceFocalLoss
- GeneralizedDiceFocalLoss
- FocalLoss
- TverskyLoss


Metrics for evaluation
######################
The available metrics are:

- MeanDice
- MeanIoU

Pipelines (training, evaluation, inference)
###########################################

- Training
- Evaluation
- Inference
