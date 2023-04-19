In order to use Plug-AI after you installed it, you should simply call:

.. code-block:: bash

    python -m plug_ai -kwargs


There are quite a lot of arguments possible so instead of giving them from the command line, you can provide them through a yaml config file by using the argument --config_file


.. code-block:: bash

    python -m plug_ai -config_file path/to/config_file.yaml


Below is an example of config file that uses:

- MedNIST Dataset
- presets transformations
- DynUnet Model
- preset training loop

.. literalinclude:: ../../plug_ai/ressources/config_exemples/config_minimalist.yaml
   :language: yaml
