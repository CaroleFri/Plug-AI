Pour lancer un job via slurm faire "sbatch plug_ai.slurm". Les sorties du jobs seront dans le fichiers plug_ai.out.
Des examples de configurations sont dans le répertoire config_exemples. Pour choisir quel fichier utiliser, il faut changer la dernière ligne du fichier slurm.
Il est possible de modifier ou créer un fichier config.

Pour l'instant, le code permet d'entrainer un DynUnet sur des base de données .nii (2 bases de données sont donnée en example).
Il est possible de faire un suivi avec tensorboard (option "report_log"), les logs seront enregistré dans le dossier report_log.
Les poids du modèle seront enregistrés par défaut dans le dossier checkpoints.

# Installation
Pour utiliser plug_ai comme une librairie,pour le moment, ajouter le chemin de plug_ai au path Python.
```
export PYTHONPATH=$PYTHONPATH:/gpfsdswork/projects/rech/ibu/ssos023/Plug-AI/
```

A venir : permettre une installation via pip