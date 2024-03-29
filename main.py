import json
from collections import defaultdict


from config.arguments import Config
from metrics.input_space import InputSpaceMetrics
from metrics.parameter_space import ParameterSpaceMetrics
from metrics.prediction_space import PredictionSpaceMetrics
from metrics.correlation_space import CorrelationSpaceMetrics
from utils.utils import *
from utils.datasets import DatasetLoader
from utils.models import ModelLoader


def main(config):
    seeds = config.seeds

    model_loader = ModelLoader(config)
    dataset_loader = DatasetLoader(config)
    input_space_metrics = InputSpaceMetrics(config)
    parameter_space_metrics = ParameterSpaceMetrics(config)
    prediction_space_metrics = PredictionSpaceMetrics(config)
    correlation_space_metrics = CorrelationSpaceMetrics(config)
    train_set, test_set = dataset_loader.load_dataset()

    results = defaultdict(dict)
    for seed in range(1, seeds + 1):
        models = model_loader.load_models(seed)
        initial_weights = model_loader.load_initial_weights(seed)

        for model, name in models:
            results[name]["input_space"] = input_space_metrics.calculate_all(model)
        for model, name in models:
            results[name]["parameter_space"] = parameter_space_metrics.calculate_all(name, model, initial_weights[name], train_set, test_set)
        for model, name in models:
            results[name]["prediction_space"] = prediction_space_metrics.calculate_all(name, model, train_set, test_set)
        print(f"*-finished measurements of seed {seed}-*")
    results = correlation_space_metrics.calculate_all(results)
    return results

def display_dict(dictionary, config):
    print(json.dumps(dictionary, indent=4))
    if config.save_file_path:
        write_to_csv(dictionary, config)

if __name__ == "__main__":
    
    config = Config()
    results = main(config)
    display_dict(results, config)


