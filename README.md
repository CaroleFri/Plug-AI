Pour lancer un job via slurm faire "sbatch plug_ai.slurm". Les sorties du jobs seront dans le fichiers plug_ai.out.
Des examples de configurations sont dans le répertoire config_exemples. Pour choisir quel fichier utiliser, il faut changer la dernière ligne du fichier slurm.
Il est possible de modifier ou créer un fichier config.

Pour l'instant, le code permet d'entrainer un DynUnet sur des base de données .nii (2 bases de données sont donnée en example).
Il est possible de faire un suivi avec tensorboard (option "report_log"), les logs seront enregistré dans le dossier report_log.
Les poids du modèle seront enregistrés par défaut dans le dossier checkpoints.

# Installation
Pour utiliser plug_ai comme une librairie, plusieurs options se présentent selon le contexte.

Sur Jean-Zay: 
    - module load pytorch-gpu/py3/1.10.1
    - git clone plug_ai
    - A la racine du dossier plug_ai : ```pip install --user --no-cache-dir -e .```
    ou
    - ```export PYTHONPATH=$PYTHONPATH:/path_to_Plug-AI/```
Les requirements sont dans le module pytorch chargé.


Ailleurs: 
    - git clone plug_ai
    - A la racine du dossier plug_ai : ```pip install --user --no-cache-dir -e .``` 
 WIP : faire les requirements pour l'install sans module  
