import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path


# ------ MODEL INITIALISATION --------
class VGG(nn.Module):
	def __init__(self, vgg_name, num_classes, batch_norm=True, bias=True, relu_inplace=True):
		super(VGG, self).__init__()

		cfg = {
			'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
			'VGG11_quad': [64, 'M', 512, 'M', 1024, 1024, 'M', 2048, 2048, 'M', 2048, 512, 'M'],
			'VGG11_doub': [64, 'M', 256, 'M', 512, 512, 'M', 1024, 1024, 'M', 1024, 512, 'M'],
			'VGG11_half': [64, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 512, 'M'],
			'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
			'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
			'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
		}

		self.batch_norm = batch_norm
		self.bias = bias
		self.features = self._make_layers(cfg[vgg_name], relu_inplace=relu_inplace)
		self.classifier = nn.Linear(512, num_classes, bias=self.bias)

	def forward(self, x):
		out = self.features(x)
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		return out

	def _make_layers(self, cfg, relu_inplace=True):
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				if self.batch_norm:
					layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias),
						   nn.BatchNorm2d(x),
						   nn.ReLU(inplace=relu_inplace)]
				else:
					layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias),
						   nn.ReLU(inplace=relu_inplace)]
				in_channels = x
		layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return nn.Sequential(*layers)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, use_batchnorm=True):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		if not use_batchnorm:
			self.bn1 = self.bn2 = nn.Sequential()

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes) if use_batchnorm else nn.Sequential()
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out

class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10, use_batchnorm=True, linear_bias=True):
		super(ResNet, self).__init__()
		self.in_planes = 64
		self.use_batchnorm = use_batchnorm
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Sequential()
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
		self.linear = nn.Linear(512*block.expansion, num_classes, bias=linear_bias)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride, self.use_batchnorm))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out
# ------ MODEL INITIALISATION --------


# ------ DATASET INITIALISATION -------
def load_dataset(dataset_name, batch_size):
	if dataset_name == "CIFAR10":
		data_root = os.path.join(os.getcwd(), "CIFAR10")
		dataset = torchvision.datasets.CIFAR10
		data_mean = (0.4914, 0.4822, 0.4465)
		data_stddev = (0.2023, 0.1994, 0.2010)
	elif dataset_name == "CIFAR100":
		data_root = os.path.join(os.getcwd(), "CIFAR100")
		dataset = torchvision.datasets.CIFAR100 
		data_mean = (0.5071, 0.4867, 0.4408)
		data_stddev = (0.2675, 0.2565, 0.2761)
	else:
		raise WrongNameError
	train_transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(data_mean, data_stddev)
		])
	test_transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(data_mean, data_stddev)
		])
	train_set = dataset(root=data_root, train=True, download=True, transform=train_transform)
	test_set = dataset(root=data_root, train=False, download=True, transform=test_transform)
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

	return train_loader, test_loader
# ------ DATASET INITIALISATION -------



