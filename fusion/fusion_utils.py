# THIS CODE WAS WRITTEN BY AND BELONGS TO SIDAK PAL SINGH, MARTIN JAGGI
# THE AUTHORS OF THE PUBLICATION "MODEL FUSION VIA OPTIMAL TRANSPORT"
# PAPER https://arxiv.org/pdf/1910.05653.pdf
# GITHUB https://github.com/sidak/otfusion/tree/master



import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from pathlib import Path
import ot
import numpy as np

# -------- ARGUMENTS ---------
class Argument:
	def __init__(self, device, dataset_name, model_type):
		self.device = device
		self.dataset_name = dataset_name
		self.model_type = model_type
		self.validate()

	def validate(self):
		assert self.device in ["cpu", "cuda", "mps"], "Device must be 'cpu', 'cuda' or 'mps'"
		assert self.dataset_name in ["CIFAR10", "CIFAR100"], "Invalid dataset name, should be 'CIFAR10' or 'CIFAR100'"
		assert self.model_type in ["VGG11_NOBIAS_NOBN", "RESNET18_NOBIAS_NOBN"], "Invalid model type, should be 'VGG11' or 'RESNET18', followed by '_NOBIAS_NOBN' (fusion not implemented for bias/bn settings)"


# -------- ARGUMENTS ---------


# ------- MODEL INITIALISATION -------
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

def ResNet18(num_classes=10, use_batchnorm=True, linear_bias=True):
	return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, use_batchnorm=use_batchnorm, linear_bias=linear_bias)

def _load_individual_model(args, path = None):
			
	num_classes = 10 if args.dataset_name == "CIFAR10" else 100
	use_bias = "NOBIAS" not in args.model_type
	use_bn = "NOBN" not in args.model_type

	if "RESNET18" in args.model_type:
		model = ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, use_batchnorm=use_bn, linear_bias=use_bias)
	if "VGG11" in args.model_type:
		model = VGG("VGG11", num_classes, batch_norm=use_bn, bias=use_bias, relu_inplace=True)
	
	if path is not None:
		try:
			state = torch.load(path, map_location=(lambda s, _: torch.serialization.default_restore_location(s, args.device)))
			#model.load_state_dict(torch.load(path))
			model.load_state_dict(state)
			print("---loaded model")
		except RuntimeError as original_error:
			print(original_error)
			print(
				"\n\n\nOopsie woopsie youre a little dumb and tried to load saved weights into a network of different shape hihi! Check the original error above for leads! (most likely a difference in batchnorm or bias)\n\n\n")
			exit()
	return model
# ------- MODEL INITIALISATION -------


# --------- NAIVE ENSEMBLING -------
def get_avg_parameters(networks, weights=None):
	avg_pars = []
	for par_group in zip(*[net.parameters() for net in networks]):
		if weights is not None:
			weighted_par_group = [par * weights[i] for i, par in enumerate(par_group)]
			avg_par = torch.sum(torch.stack(weighted_par_group), dim=0)
		else:
			avg_par = torch.mean(torch.stack(par_group), dim=0)
		avg_pars.append(avg_par)
	return avg_pars

def naive_ensembling(args, networks):
	# simply average the weights in networks
	weights = [0.5, 0.5]
	avg_pars = get_avg_parameters(networks, weights)
	ensemble_network = _load_individual_model(args)
	# put on GPU
	ensemble_network.to(args.device)
	
	# set the weights of the ensembled network
	for idx, (name, param) in enumerate(ensemble_network.state_dict().items()):
		ensemble_network.state_dict()[name].copy_(avg_pars[idx].data)

	print("---done naive fusion")
	return ensemble_network
# --------- NAIVE ENSEMBLING -------


# -------- GEOMETRIC ENSEMBLING -------
def isnan(x):
	return x != x

class GroundMetric:
	"""
		Ground Metric object for Wasserstein computations:

	"""

	def __init__(self, params, not_squared = False):
		self.params = params
		self.ground_metric_type = "euclidean"
		self.ground_metric_normalize = "none"
		self.reg = 0.01
		self.squared = False

	def _normalize(self, ground_metric_matrix):

		if self.ground_metric_normalize == "log":
			ground_metric_matrix = torch.log1p(ground_metric_matrix)
		elif self.ground_metric_normalize == "max":
			ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.max()
		elif self.ground_metric_normalize == "median":
			ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.median()
		elif self.ground_metric_normalize == "mean":
			ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.mean()
		elif self.ground_metric_normalize == "none":
			return ground_metric_matrix
		else:
			raise NotImplementedError

		return ground_metric_matrix

	def _sanity_check(self, ground_metric_matrix):
		assert not (ground_metric_matrix < 0).any()
		assert not (isnan(ground_metric_matrix).any())

	def _cost_matrix_xy(self, x, y, p=2, squared = True):
		# TODO: Use this to guarantee reproducibility of previous results and then move onto better way
		"Returns the matrix of $|x_i-y_j|^p$."
		x_col = x.unsqueeze(1)
		y_lin = y.unsqueeze(0)
		c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
		if not squared:
			c = c ** (1/2)
		# print(c.size())
	   
		return c


	def _pairwise_distances(self, x, y=None, squared=True):
		'''
		Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
		Input: x is a Nxd matrix
			   y is an optional Mxd matirx
		Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
				if y is not given then use 'y=x'.
		i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
		'''
		x_norm = (x ** 2).sum(1).view(-1, 1)
		if y is not None:
			y_t = torch.transpose(y, 0, 1)
			y_norm = (y ** 2).sum(1).view(1, -1)
		else:
			y_t = torch.transpose(x, 0, 1)
			y_norm = x_norm.view(1, -1)

		dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
		# Ensure diagonal is zero if x=y
		dist = torch.clamp(dist, min=0.0)

		
		if not squared:
			dist = dist ** (1/2)

		return dist

	def _get_euclidean(self, coordinates, other_coordinates=None):
		# TODO: Replace by torch.pdist (which is said to be much more memory efficient)

		if other_coordinates is None:
			matrix = torch.norm(
				coordinates.view(coordinates.shape[0], 1, coordinates.shape[1]) \
				- coordinates, p=2, dim=2
			)
		else:
			matrix = self._pairwise_distances(coordinates, other_coordinates, squared=self.squared)
			
		return matrix

	def _normed_vecs(self, vecs, eps=1e-9):
		norms = torch.norm(vecs, dim=-1, keepdim=True)
		
		return vecs / (norms + eps)

	def _get_cosine(self, coordinates, other_coordinates=None):
		if other_coordinates is None:
			matrix = coordinates / torch.norm(coordinates, dim=1, keepdim=True)
			matrix = 1 - matrix @ matrix.t()
		else:
			matrix = 1 - torch.div(
				coordinates @ other_coordinates.t(),
				torch.norm(coordinates, dim=1).view(-1, 1) @ torch.norm(other_coordinates, dim=1).view(1, -1)
			)
		return matrix.clamp_(min=0)

	def _get_angular(self, coordinates, other_coordinates=None):
		pass

	def get_metric(self, coordinates, other_coordinates=None):
		get_metric_map = {
			'euclidean': self._get_euclidean,
			'cosine': self._get_cosine,
			'angular': self._get_angular,
		}
		return get_metric_map[self.ground_metric_type](coordinates, other_coordinates)

	def process(self, coordinates, other_coordinates=None):
		coordinates = self._normed_vecs(coordinates)
		if other_coordinates is not None:
			other_coordinates = self._normed_vecs(other_coordinates)

		ground_metric_matrix = self.get_metric(coordinates, other_coordinates)

		self._sanity_check(ground_metric_matrix)

		ground_metric_matrix = self._normalize(ground_metric_matrix)

		self._sanity_check(ground_metric_matrix)

		return ground_metric_matrix

def get_histogram(idx, cardinality, layer_name):
	# returns a uniform measure
	return np.ones(cardinality)/cardinality
	
def get_wassersteinized_layers_modularized(args, networks, eps=1e-7):
	'''
	Two neural networks that have to be averaged in geometric manner (i.e. layerwise).
	The 1st network is aligned with respect to the other via wasserstein distance.
	Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*

	:param networks: list of networks
	:param activations: If not None, use it to build the activation histograms.
	Otherwise assumes uniform distribution over neurons in a layer.
	:return: list of layer weights 'wassersteinized'
	'''

	# simple_model_0, simple_model_1 = networks[0], networks[1]
	# simple_model_0 = get_trained_model(0, model='simplenet')
	# simple_model_1 = get_trained_model(1, model='simplenet')

	avg_aligned_layers = []
	# cumulative_T_var = None
	T_var = None
	# print(list(networks[0].parameters()))
	previous_layer_shape = None
	ground_metric_object = GroundMetric(args)


	num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
	for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in \
			enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):

		assert fc_layer0_weight.shape == fc_layer1_weight.shape
		previous_layer_shape = fc_layer1_weight.shape

		mu_cardinality = fc_layer0_weight.shape[0]
		nu_cardinality = fc_layer1_weight.shape[0]

		# mu = np.ones(fc_layer0_weight.shape[0])/fc_layer0_weight.shape[0]
		# nu = np.ones(fc_layer1_weight.shape[0])/fc_layer1_weight.shape[0]

		layer_shape = fc_layer0_weight.shape
		if len(layer_shape) > 2:
			is_conv = True
			# For convolutional layers, it is (#out_channels, #in_channels, height, width)
			fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
			fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
		else:
			is_conv = False
			fc_layer0_weight_data = fc_layer0_weight.data
			fc_layer1_weight_data = fc_layer1_weight.data

		if idx == 0:
			if is_conv:
				M = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
								fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
				
			else:
				# print("layer data is ", fc_layer0_weight_data, fc_layer1_weight_data)
				M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)

			aligned_wt = fc_layer0_weight_data
		else:

			# aligned_wt = None, this caches the tensor and causes OOM
			if is_conv:
				T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
				aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)

				M = ground_metric_object.process(
					aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
					fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
				)
			else:
				if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
					# Handles the switch from convolutional layers to fc layers
					fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
					aligned_wt = torch.bmm(
						fc_layer0_unflattened,
						T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
					).permute(1, 2, 0)
					aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
				else:
					# print("layer data (aligned) is ", aligned_wt, fc_layer1_weight_data)
					aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
				M = ground_metric_object.process(aligned_wt, fc_layer1_weight)
			

		mu = get_histogram(0, mu_cardinality, layer0_name)
		nu = get_histogram(1, nu_cardinality, layer1_name)
		

		cpuM = M.data.cpu().numpy()

		T = ot.emd(mu, nu, cpuM)
		

		T_var = torch.from_numpy(T).to(args.device).float()
		

		# torch.set_printoptions(profile="full")
		# torch.set_printoptions(profile="default")

		# think of it as m x 1, scaling weights for m linear combinations of points in X
		marginals = torch.ones(T_var.shape[0]).to(args.device) / T_var.shape[0]
				
		marginals = torch.diag(1.0/(marginals + eps))  # take inverse
		T_var = torch.matmul(T_var, marginals)
			
		t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
		
		# Average the weights of aligned first layers
		
		geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))/2
		if is_conv and layer_shape != geometric_fc.shape:
			geometric_fc = geometric_fc.view(layer_shape)
		avg_aligned_layers.append(geometric_fc)
		

	return avg_aligned_layers

def get_network_from_param_list(args, param_list):

	new_network = _load_individual_model(args)
	new_network.to(args.device)

	# check the test performance of the network before
	

	# set the weights of the new network
	# print("before", new_network.state_dict())
	
	assert len(list(new_network.parameters())) == len(param_list)

	model_state_dict = new_network.state_dict()

	for layer_idx, (key, _) in enumerate(model_state_dict.items()):
		model_state_dict[key] = param_list[layer_idx]

	new_network.load_state_dict(model_state_dict)  

	return new_network

def geometric_ensembling(args, networks):
	
	avg_aligned_layers = get_wassersteinized_layers_modularized(args, networks)
	print("---done geometric fusion")
	return get_network_from_param_list(args, avg_aligned_layers)
# -------- GEOMETRIC ENSEMBLING -------

# ------- MODEL SAVING -------
def save_models(path, model1, model2):
	Path(path).mkdir(parents=True, exist_ok=True)
	
	model1_state = model1.state_dict()
	torch.save(model1_state, os.path.join(path, 'naive_fused.pth'))
	model2_state = model2.state_dict()
	torch.save(model2_state, os.path.join(path, 'geometric_fused.pth'))
	print("---saved models")
# ------- MODEL SAVING -------

def check_if_same_model(model1, model2):
	for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
		assert param1.size() == param2.size(), f"sizes unequal: {name1}: {param1.size()} and {name2}: {param2.size()}"
		assert name1 == name2, f"layer types unequal: {name1} and {name2}"
		if not torch.equal(param1, param2):
			print("layer is unequal")
		else:
			print("layer is equal")

