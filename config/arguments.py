import os
import argparse

class Config:
    def __init__(self):
        cli_args = self._parse_args()

        self.device = cli_args.device
        self.experiment_name = cli_args.experiment_name
        self.model_type = cli_args.model_type
        self.dataset_name = cli_args.dataset_name

        self._check_validity()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--device", required=True, type=str)
        parser.add_argument("--experiment_name", required=True, type=str)
        parser.add_argument("--model_type", required=True, type=str)
        parser.add_argument("--dataset_name", required=True, type=str)
        cli_args = parser.parse_args()

        bool_dict = {"True": True, "False": False}
        bool_args = []

        for bool_arg in bool_args:
            if getattr(cli_args, bool_arg):
                assert getattr(cli_args, bool_arg) in bool_dict, "Boolean value must be either 'True' or 'False'."
                setattr(cli_args, bool_arg, bool_dict[getattr(cli_args, bool_arg)])
        return cli_args 

    def _check_validity(self):
        experiment_path = os.path.join(os.getcwd(), "experiments", self.experiment_name)
        
        assert self.device in ["cpu", "cuda", "mps"], "Device must be 'cpu', 'cuda' or 'mps'"
        assert os.path.exists(experiment_path), "Experiment folder does not exist"
        assert len(os.listdir(os.path.join(experiment_path, "parents"))), "Parents folder does not contain any models"
        assert len(os.listdir(os.path.join(experiment_path, "fused"))), "Fused folder does not contain any models"
        assert self.dataset_name in ["CIFAR10", "CIFAR100"], "Invalid dataset name, should be 'CIFAR10' or 'CIFAR100'"
        assert self.model_type in ["VGG11", "VGG11_NOBIAS", "VGG11_NOBIAS_NOBN", "RESNET18", "RESNET18_NOBIAS", "RESNET18_NOBIAS_NOBN"], "Invalid model type, should be 'VGG11' or 'RESNET18', with _NOBIAS and/or _NOBN in that order"
