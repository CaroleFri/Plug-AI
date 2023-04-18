API Reference
=============

Plug-AI as an Application
#########################

Plug-AI can be used as an application with the following parameters by calling:

.. code-block:: bash

   python -m plug_ai --kwargs 

Below are the arguments for plug_ai

.. literalinclude:: plug_ai_help.txt
   :language: console
   
You can provide a configuration of all the arguments using --config_file. Arguments given both in the config file and the console are overwritten by the console. This can be useful if you are working with a general config file and are testing variations of just a few parameters.


Plug-AI as a package
####################

Plug-AI can also be imported and used in your python codes.
This can be useful if you want to use a module that is not in the catalogue of features while reusing parts of plug_ai.

In this situation, you would import the three managers plug_ai using : 

- dataset manager
- model manager
- execution manager

Many elements can be provided yourself to the managers.

For the dataset:

- a dataset as a Pytorch dataset herited class
- transformations as a collable

For the model:

- a Pytorch model 

For the execution manager:

- a training loop
- a training step
- an optimizer
- a loss
- a criterion

Example:
Let's suppose you have a custom dataset you want to use with the rest of Plug-AI.
You would instantiate your dataset and provide it to the dataset manager.
Then you would set the model and execution.

.. code-block:: python

   custom_dataset = ...
   dataset_manager = plug...(dataset = custom_dataset
                          ...)
   model_manager = ...
   execution_manager = ...






