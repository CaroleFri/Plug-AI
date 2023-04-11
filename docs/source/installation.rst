Installation
============

In order to install and use Plug-AI, you have multiple options:
1. install from pip
2. install from source
3. if you are on Jean-Zay cluster, use a specific module (work in progress)

Install from pip (WIP)
-----------------------------------
.. code-block:: bash
    pip install plug-ai
Reminder: On jean-zay, you must do installations using :
.. code-block:: bash
    pip install --user --no-cache-dir plug-ai


Install from source
-------------------
If you want to install from sources, you can clone the git repositery and install from here:
.. code-block:: bash
    git clone WIP
    cd WIP
    pip install



Usage on Jean-zay cluster
-------------------------
The module containing plug_ai is (WIP)
You can load it using: 
.. code-block:: bash
    module load WIP

If you want to use a notebook on JupyterHub, you can load once JupyterHub is ready by:
    1. Going into the software section (blue hexagonal buttong at the bottom left)
    2. Searching for module WIP
    3. Loading the module
Alternatively, you can also define a specific module for one notebook and not the whole session by selecting the module as the running kernel after you opened the notebook.
    
   