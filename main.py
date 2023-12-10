from config.arguments import Config
from utils.models import ModelLoader
from utils.datasets import DatasetLoader
from metrics.input_space import InputSpaceMetrics 
from metrics.parameter_space import ParameterSpaceMetrics

def main(config):
    model_loader = ModelLoader(config)
    dataset_loader = DatasetLoader(config)
    input_space_metrics = InputSpaceMetrics(config)
    parameter_space_metrics = ParameterSpaceMetrics(config)

    parent_one, parent_two, fused_naive, fused_geometric = model_loader.load_models()
    train_set, test_set = dataset_loader.load_dataset()

if __name__ == "__main__":
    config = Config()
    main(config)

