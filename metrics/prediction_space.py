import torch
from torch.utils.data import DataLoader
import os


class PredictionSpaceMetrics:
	def __init__(self, config):
		self.device = config.device
		self.batch_size = config.batch_size
		self.mode = config.accuracy_metric
		self.measure = config.pred_space

	def calculate_all(self, name, model, train_set, test_set):
		model.to(self.device)

		prediction_space_results = dict()

		if self.measure:
			if self.mode == "accuracy":
				test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
				prediction_space_results["accuracy"] = self.run_epoch(name, model, test_loader)
			else:
				train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False)
				test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
				train_acc = self.run_epoch(name, model, train_loader)
				test_acc = self.run_epoch(name, model, test_loader)
				# generalisation gap reciprocal (so correlation has same sign as with accuracy)
				prediction_space_results["generalisation_gap"] = 1 / abs(train_acc - test_acc)
		return prediction_space_results

	def run_epoch(self, name, model, loader):
		sett = "training" if len(loader.dataset) == 50000 else "validation"
		seed = name.rsplit('_', 1)[-1]
		path = os.path.join(os.getcwd(), "experiments", self.experiment_name,  str(self.experiment_name + "_" + seed), "statistics")
		filename = os.path.join(path, f"{name}_{sett}_accuracy.pt")
		if os.path.exists(filename):
			return torch.load(filename)
		with torch.no_grad():
			model.eval()
			correct = 0
			total = 0
			for i, (x, ygt) in enumerate(loader):
				x, ygt = x.to(self.device), ygt.to(self.device)
				ypr = model(x)
				predicted = torch.argmax(ypr, 1)
				total += ygt.size(0)
				correct += (predicted == ygt).sum().item()
		accuracy = correct / total
		
		if not os.path.exists(path):
			os.makedirs(path)
		torch.save(accuracy, filename)

		print(f"---found {sett} accuracy")
		return accuracy
