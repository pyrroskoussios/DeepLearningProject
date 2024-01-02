from collections import defaultdict

from config.arguments import Config
from matplotlib import pyplot as plt
from metrics.input_space import InputSpaceMetrics
from metrics.parameter_space import ParameterSpaceMetrics
from metrics.prediction_space import PredictionSpaceMetrics
from metrics.correlation_space import CorrelationSpaceMetrics
from utils.utils import write_to_csv
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

    if seeds == 1:

        models = model_loader.load_models()    
        results = defaultdict(dict)
        for model, name in models:
            results[name]["input_space"] = input_space_metrics.calculate_all(model)
        for model, name in models:
            results[name]["parameter_space"] = parameter_space_metrics.calculate_all(model, train_set, test_set)

        return results

    else:
        for seed in range(seeds):
            models = model_loader.load_models(seed)
            for model, name in models:
                results[name]["input_space"] = input_space_metrics.calculate_all(model)
            for model, name in models:
                results[name]["parameter_space"] = parameter_space_metrics.calculate_all(model, train_set, test_set)
            for model, name in models:
                results[name]["prediction_space"] = prediction_space_metrics.calculate_all(model, train_set, test_set)
        results = correlation_space_metrics.calculate_all(results, train_set, test_set)
        return results

def pretty_print(dictionary, save_file_path):
    import json
    print(json.dumps(dictionary, indent=4))
    if save_file_path:
        write_to_csv(dictionary, save_file_path)


if __name__ == "__main__":
    
    config = Config()
    results = main(config)
    pretty_print(results, config.save_file_path)


