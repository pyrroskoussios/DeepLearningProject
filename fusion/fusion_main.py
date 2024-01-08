from fusion_utils import Argument, _load_individual_model, naive_ensembling, geometric_ensembling, save_models


def initialisation():
	if not os.path.exists(os.path.join(os.getcwd(), 'checkpoints')):
		os.mkdir(os.path.join(os.getcwd(), 'checkpoints'))
	fusion_path = os.path.join(os.getcwd(), 'fusion')
	print("GPU available: ", torch.cuda.is_available())
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.set_default_dtype(torch.float32)

	return fusion_path, device

if __name__ == "__main__":

	fusion_path, device = initialisation()
	dataset_name = "CIFAR10"
	model_type = "VGG11_NOBIAS_NOBN"

	path_parent1 = "parent_1.pth"
	path_parent2 = "parent_2.pth"
	
	args = Argument(device, dataset_name, model_type)

	
	parent1 = _load_individual_model(args, path = path_parent1)
	parent2 = _load_individual_model(args, path = path_parent2)
	parents = [parent1, parent2]

	naive_fused = naive_ensembling(args, parents)
	geometric_fused = geometric_ensembling(args, parents)
	save_models(fusion_path, naive_fused, geometric_fused)
	