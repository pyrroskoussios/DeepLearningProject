from collections import defaultdict
from matplotlib import pyplot as plt

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

    models = model_loader.load_models()
    train_set, test_set = dataset_loader.load_dataset()
    
    results = defaultdict(dict)
    for model, name in models:
        results[name]["input_space"] = input_space_metrics.calculate_all(model)
    for model, name in models:
        results[name]["parameter_space"] = parameter_space_metrics.calculate_all(model, test_set)

    return results

def pretty_print(dictionary):
    import json
    print(json.dumps(dictionary, indent=4))

if __name__ == "__main__":
    config = Config()
    results = main(config)
    pretty_print(results)

