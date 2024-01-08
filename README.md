# Deep Learning Project: Understanding OT Fusion

This is the repository for the corresponding report handed in by Benjamin Jäger, Pyrros Koussios, and Santiago Gallego Restrepo, titled "Understanding Model Fusion". The repository has the following relevant structure:

    -DeepLearningProject
    - -main.py
    - -experiments
    - - -resnet18_cifar10
    - - - -resnet18_cifar10_seed1
    - - - - ...
    - - -resnet18_cifar100
    - - - ...
    - -config
    - - -arguments.py
    - -utils
    - - -utils.py
    - - -datasets.py
    - - -models.py
    - -metrics
    - - -parameter_space.py
    - - -input_space.py
    - - -prediction_space.py
    - - -correlation_space.py
    - -fusion
    - - -fusion_main.py
    - - -fusion_utils.py
    - -training
    - - -training_main.py
    - - -training_utils.py

The results created using the `training` and `fusion` directories are not automatically referenced to by any of the other code, you will have to manually move the state dicts in accordance to this README. We do provide all model data that we created for the work conducted in the report.

## Training
Navigate to the `training` directory. Open the `training_main.py` file and adjust the model, data and hyperparameters. Run in order to create a `checkpoints` folder, which holds the epoch checkpoints, the initial, and the best validation state dict. 

## Fusion
After having obtained at least two model state dicts of the same configuration (which is required in order for fusion to work), navigate to the `fusion` directory. Open the `fusion_main.py` file and adjust the parameters and paths to said models. Run in order to create a `fusion` folder, which holds the state dicts of the naive and the geometric fusion.

## Evaluating the models
After creating at least one set of model data (including `parent_1, parent_2, parent_1_initial_weights, parent_2_initial_weights, naive_fused, geometric_fused`), assemble them under the `experiments` directory in a folder named by the configuration (see existing folders), in a subfolder numbered by its seed (e.g. `experiments/resnet18_cifar10/resnet18_cifar10_seed1`).
Run `main.py` with the flags corresponding to your experiment (see `config/arguments.py` for the available arguments). Locate the `results.csv` file to view the resulting output.
