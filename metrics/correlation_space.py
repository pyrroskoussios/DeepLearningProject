import torch
import torchvision
from torch.utils.data import DataLoader

from scipy.stats import pearsonr
from scipy.stats import kendalltau
import numpy as np
from collections import defaultdict




class CorrelationSpaceMetrics:
	def __init__(self, config):
		
		self.device = config.device
		self.mode = config.accuracy_metric
		self.seeds = config.seeds
		self.batch_size = config.batch_size
		
	def calculate_all(self, all_results):
		accuracies, sorted_results = self.sort_results(all_results)
		correlation_space_results = dict()

		for model_name, _, metric_name, metric_values in sorted_results:
			correlation_space_results[str(model_name + '_Kendall_' + metric_name)] = self.kendall_rank_correlation(metric_values, accuracies[model_name])
			correlation_space_results[str(model_name + '_Pearson_' + metric_name)] = self.pearson_correlation(metric_values, accuracies[model_name])

	def kendall_rank_correlation(self, metric, accuracy):
		return kendalltau(metric, accuracy)[0]

	def pearson_correlation(self, metric, accuracy):
		return pearsonr(metric, accuracy)[0]

	def sort_results(self, results):
		grouped_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

		for model_name, metric_space in results.items():
			model_type, seed = model_name.rsplit('_', 1)
			for metric_type, metrics in metric_space.items():
				for metric_name, metric_value in metrics.items():
					grouped_metrics[model_type][metric_type][metric_name].append((int(seed), metric_value))

		grouped_metrics_dict = dict(grouped_metrics)
		grouped_list = []
		for model_type, metric_types in grouped_metrics_dict.items():
			for metric_type, metrics in metric_types.items():
				for metric_name, seed_values in metrics.items():
					sorted_seed_values = sorted(seed_values, key=lambda x: x[0])
					values = [value for _, value in sorted_seed_values]
					grouped_list.append((model_type, metric_type, metric_name, values))

		accuracies = {model_name: values for model_name, metric_type, _, values in grouped_list if metric_type == "prediction_space"}
		grouped_list = [x for x in grouped_list if x[1] != "prediction_space"]

		return accuracies, grouped_list



if __name__ == "__main__":
	cl = CorrelationSpaceMetrics(None)
	results = defaultdict(dict)
	results["fuse_1"]["input_space"] = {"hessian": 4, "gaussian": 1}
	results["fuse_1"]["param_space"] = {"integral": 2, "derivative": 8}
	results["fuse_2"]["param_space"] = {"integral": 6, "derivative": 3}
	results["fuse_2"]["input_space"] = {"hessian": 9, "gaussian": 7}
	results["parent_one_1"]["input_space"] = {"hessian": 22, "gaussian": 44}
	results["parent_one_2"]["input_space"] = {"hessian": 55, "gaussian": 66}
	results["parent_one_2"]["param_space"] = {"integral": 33, "derivative": 11}
	results["parent_two_1"]["param_space"] = {"integral": 222, "derivative": 333}
	results["parent_two_2"]["param_space"] = {"integral": 111, "derivative": 444}

	print(results)
	cl.sort_results(results)








