import torch
from torch.utils.data import DataLoader


class PredictionSpaceMetrics:
	def __init__(self, config):
		self.device = config.device
		self.batch_size = config.batch_size
		self.mode = config.accuracy_metric

	def calculate_all(self, model, train_set, test_set):
		prediction_space_results = dict()
		if self.mode == "accuracy":
			test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
			prediction_space_results["accuracy"] = self.run_epoch(model, test_loader)
		else:
			train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False)
			test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
			train_acc = self.run_epoch(model, train_loader)
			test_acc = self.run_epoch(model, test_loader)
			prediction_space_results["generalisation_gap"] = 1 / abs(train_acc - test_acc)
		return prediction_space_results

	def run_epoch(self, model, loader):
		with torch.no_grad():
			model.eval()
			correct = 0
			total = 0
			for i, batch in enumerate(loader):
				x, ygt = batch
				ypr = model(x)
				predicted = torch.argmax(ypr, 1)
				total += ygt.size(0)
				correct += (predicted == ygt).sum().item()
		return 100 * correct / total
