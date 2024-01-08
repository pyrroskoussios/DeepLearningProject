# Deep Learning Project: Understanding OT Fusion

This is the repository for the corresponding report handed in by Benjamin Jäger, Pyrros Koussios, and Santiago Gallego Restrepo, titled "Understanding Model Fusion". 

## Repository Structure
The repository has the following relevant structure:
```
.
├── config
│   └── arguments.py
├── experiments
├── fusion
│   ├── fusion_train.py
│   ├── fusion_utils.py
├── main.py
├── metrics
│   ├── correlation_space.py
│   ├── input_space.py
│   ├── parameter_space.py
│   └── prediction_space.py
├── README.md
├── requirements.txt
├── training
│   ├── training_main.py
│   ├── training_utils.py
└── utils
    ├── datasets.py
    ├── models.py
    └── utils.py
```

## Installation


#### Step 1: Install Dependencies


Install the required packages using:
```bash
pip install -r requirements.txt
```
#### Step 2: Download or create Model Weights

We have conducted experiments using two model architectures (VGG11 and ResNet18) on two datasets (CIFAR10 and CIFAR100), creating four configurations. For each experiment, training was performed with five different seeds. 
You can either download the weights for the experiments we conducted or create new weights using the `/training` and `/fusion` frameworks (for this, see below). Either way, ensure to save the weights in the `/experiments` folder like so:
```
.
└── experiments
    └── <experiment_name>
        ├── <experiment_name>_seed1
        │   ├── parent_1.pth
        │   ├── parent_2.pth
        │   ├── geometric_fused.pth
        │   ├── naive_fused.pth
        │   ├── parent_1_initial_weights.pth
        │   └── parent_2_initial_weights.pth
        ├── <experiment_name>_seed2
        │   ├── ...
        │   ...
    │   ...
```
The weights used for the report can be found at [link to weights]. Following the Training and Fusion section is not necessary when downloading the weights.

## Training
Navigate to the `/training` directory. Open the `training_main.py` file and adjust the model, data and hyperparameters. Run in order to create a `/checkpoints` folder, which holds the epoch checkpoints, the initial, and the best validation state dict of the current run.

## Fusion
After having obtained at least two model state dicts of the same configuration (which is required in order for fusion to work), navigate to the `/fusion` directory. Open the `fusion_main.py` file and adjust the parameters and paths to said models. Run in order to create a `/fused` folder, which holds the state dicts of the naive and the geometric fusion.

## Conducting the measurements
Ensure at least having one seed of model data (as depicted above). 
To run the `main.py` script, use the following command structure:

```bash
python main.py --device <device> --experiment_name <experiment_name> --seeds <seeds> --model_type <model_type> --dataset_name <dataset_name> [--optional_flag <True/False>]
```

#### Command-Line Arguments

- `device`: Specify the device to run the analysis on (e.g. 'cuda').
- `experiment_name`: Specify which experiment (i.e. model architecture and dataset) to analyse.
- `model_type`: Specify the type of model to use (e.g. 'RESNET18_NOBIAS_NOBN').
- `dataset_name`: Define the name of the dataset to be used (e.g. 'CIFAR100').
- `seeds`: Specify the number of seeds to use for the analysis.


#### Optional Arguments
- `batch_size`: Define the batch size for training and evaluation. Default is 100.
- `accuracy_metric`: Specify the accuracy metric to be used (between 'accuracy' and 'generalisation_gap'). Default is 'accuracy'.
- `save_file_path`: Path and file name where the results will be saved. By default, they are saved in a `./results.csv` file.
- `input_space`: Enable input space analysis. Default is 'True'.
- `param_space`: Enable parameter space analysis. Default is 'True'.
- `pred_space`: Enable prediction space analysis. Default is 'True'.
- `correl_space`: Enable correlation space analysis. Default is 'True'.


#### Example Command

```bash
python main.py --device cuda --experiment_name resnet18_cifar10 --model_type RESNET18_NOBIAS_NOBN --dataset_name CIFAR10 --seeds 3 --save_file_path results_test --param_space False
```

In this example, we are only using three seeds and we are not computing the parameter space metrics. The resulting measurements are saved in `./results_test.csv`.
