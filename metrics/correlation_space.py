import torch
import torchvision
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, kendalltau, spearmanr
import numpy as np
from collections import defaultdict


class CorrelationSpaceMetrics:
	def __init__(self, config):
		self.device = config.device
		self.mode = config.accuracy_metric
		self.seeds = config.seeds
		self.batch_size = config.batch_size
		self.measure = config.correl_space
		
	def calculate_all(self, all_results):
		accuracies, sorted_results = self.sort_filter_results(all_results)
		correlation_space_results = {"parent_1": {}, "parent_2": {}, "naive_fused": {}, "geometric_fused":{}}
		
		if self.measure:
			for model_name, metric_type, metric_name, metric_values in sorted_results:
			
				correlation_space_results[model_name][str("kendall_" + metric_name)] = self.kendall_rank_correlation(metric_values, accuracies[model_name])
				correlation_space_results[model_name][str("pearson_" + metric_name)] = self.pearson_correlation(metric_values, accuracies[model_name])
				correlation_space_results[model_name][str("spearman_" + metric_name)] = self.spearman_correlation(metric_values, accuracies[model_name])

			for model_name in ["parent_1", "parent_2", "naive_fused", "geometric_fused"]:
				all_results[model_name]["correlation_space"] = correlation_space_results[model_name]
			
		return all_results

	def kendall_rank_correlation(self, metric_values, accuracy):
		return kendalltau(metric_values, accuracy)[0]

	def pearson_correlation(self, metric_values, accuracy):
		return pearsonr(metric_values, accuracy)[0]

	def spearman_correlation(self, metric_values, accuracy):
		return spearmanr(metric_values, accuracy)[0]

	def sort_filter_results(self, results):
		grouped_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

		for model_name, metric_space in results.items():
			model_type, seed = model_name.rsplit('_', 1)
			seed = seed[-1]  # NEW
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



if __name__ == "__main__":  # ignore, used this for testing
	config = {"colab":False}
	results = load_from_csv("results_vgg11_cifar10", config, root = "/Users/benjaminjaeger/Downloads/")
	del results["geometric_fused"]
	del results["naive_fused"]
	del results["parent_1"]
	del results["parent_2"]
	print(json.dumps(results, indent=4))

	cl = CorrelationSpaceMetrics(None)
	"""
	results = defaultdict(dict)
	results["fuse_seed1"]["input_space"] = {"hessian": 4, "gaussian": 1}
	results["fuse_seed1"]["param_space"] = {"integral": 2, "derivative": 8}
	results["fuse_seed2"]["param_space"] = {"integral": 6, "derivative": 3}
	results["fuse_seed2"]["input_space"] = {"hessian": 9, "gaussian": 7}
	results["parent_one_seed1"]["input_space"] = {"hessian": 22, "gaussian": 44}
	results["parent_one_seed2"]["input_space"] = {"hessian": 55, "gaussian": 66}
	results["parent_one_seed2"]["param_space"] = {"integral": 33, "derivative": 11}
	results["parent_one_seed1"]["param_space"] = {"integral": 32, "derivative": 13}
	results["parent_two_seed1"]["param_space"] = {"integral": 222, "derivative": 333}
	results["parent_two_seed2"]["param_space"] = {"integral": 111, "derivative": 444}
	results["parent_two_seed2"]["prediction_space"] = {"accuracy": 97}
	results["parent_two_seed1"]["prediction_space"] = {"accuracy": 98}
	results["parent_one_seed2"]["prediction_space"] = {"accuracy": 86}
	results["parent_one_seed1"]["prediction_space"] = {"accuracy": 87}
	results["fuse_seed1"]["prediction_space"] = {"accuracy": 77}
	results["fuse_seed2"]["prediction_space"] = {"accuracy": 78}
	"""
	#print(results)
	print("-----------------")
	#cl.sort_results(results)
	results = cl.calculate_all(results)
	print("*-----------*")
	#print(results)
	print("---------")
	import json
	print(json.dumps(results, indent=4))

