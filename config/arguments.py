import argparse
import os


class Config:
    def __init__(self):
        cli_args = self._parse_args()

        self.device = cli_args.device
        self.experiment_name = cli_args.experiment_name
        self.model_type = cli_args.model_type
        self.dataset_name = cli_args.dataset_name
        self.seeds = cli_args.seeds
        self.batch_size = cli_args.batch_size
        self.accuracy_metric = cli_args.accuracy_metric
        self.save_file_path = cli_args.save_file_path
        self.input_space = cli_args.input_space
        self.param_space = cli_args.param_space
        self.pred_space = cli_args.pred_space
        self.correl_space = cli_args.correl_space
        self.colab = cli_args.colab
        self._check_validity()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--device", required=True, type=str)
        parser.add_argument("--experiment_name", required=True, type=str)
        parser.add_argument("--model_type", required=True, type=str)
        parser.add_argument("--dataset_name", required=True, type=str)
        parser.add_argument("--seeds", required=True, type=int)
        parser.add_argument("--batch_size", type=int, default=100)
        parser.add_argument("--save_file_path", type=str, default="results")
        parser.add_argument("--input_space", type=str, default="True")
        parser.add_argument("--param_space", type=str, default="True")
        parser.add_argument("--pred_space", type=str, default="True")
        parser.add_argument("--accuracy_metric", type=str, default="accuracy")
        parser.add_argument("--correl_space", type=str, default="True")
        parser.add_argument("--colab", type=str, default="False")
        cli_args = parser.parse_args()

        bool_dict = {"True": True, "False": False}
        bool_args = ["input_space", "param_space", "pred_space", "correl_space", "colab"]

        for bool_arg in bool_args:
            if getattr(cli_args, bool_arg):
                assert getattr(cli_args, bool_arg) in bool_dict, "Boolean value must be either 'True' or 'False'."
                setattr(cli_args, bool_arg, bool_dict[getattr(cli_args, bool_arg)])
        return cli_args 

    def _check_validity(self):
        root = "/content/drive/MyDrive/DeepLearningProject" if self.colab else os.getcwd()
        experiment_path = os.path.join(root, "experiments", self.experiment_name)
        
        assert self.device in ["cpu", "cuda", "mps"], "Device must be 'cpu', 'cuda' or 'mps'"
        assert os.path.exists(experiment_path), "Experiment folder does not exist"
        assert len(os.listdir(experiment_path)), "Experiment folder does not contain any models"
        assert self.dataset_name in ["CIFAR10", "CIFAR100"], "Invalid dataset name, should be 'CIFAR10' or 'CIFAR100'"
        assert self.model_type in ["VGG11", "VGG11_NOBIAS", "VGG11_NOBIAS_NOBN", "RESNET18", "RESNET18_NOBIAS", "RESNET18_NOBIAS_NOBN"], "Invalid model type, should be 'VGG11' or 'RESNET18', with _NOBIAS and/or _NOBN in that order"
        assert self.seeds > 0 and self.seeds <= 5, "Choose an amount of seeds to use between 1 and 5"
        assert self.accuracy_metric in ["accuracy", "generalisation_gap"], "Prediction metric is either accuracy or generalisation gap"
        assert not (self.correl_space and self.seeds <= 1), "Correlation can only be computed by using more than one seed"
        assert not (self.correl_space and not self.pred_space), "Correlation can only be computed by computing a prediction metric first"
        assert self.batch_size >= 1 and self.batch_size <= 10000, "Batch Size must be between 1 and 10000"
