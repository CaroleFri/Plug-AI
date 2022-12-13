Here are found tests to run to check Plug_ai.

- slurm_tests.ipynb is to be runned from a frontal node, it will start jobs on Jean-Zay.

- interactive_tests.ipynb must be run from a compute node. It will run from a frontal node but this is a bad practice as it will saturate them.

- WIP : some unitary tests for fundamentals blocs

config_test.yaml is the non-default config used by the tests.
Enrich the tests by adding extra config_test.yaml files to check various functionnality.


To be implemented : an automation of the tests 