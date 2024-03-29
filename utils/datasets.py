import os
import torchvision

class DatasetLoader:
    def __init__(self, config):
        self.dataset_name = config.dataset_name
        self.root = "/content/drive/MyDrive/DeepLearningProject" if config.colab else os.getcwd()


    def load_dataset(self):
        if self.dataset_name == "CIFAR10":
            data_root = os.path.join(self.root, "CIFAR10")
            dataset = torchvision.datasets.CIFAR10
            data_mean = (0.4914, 0.4822, 0.4465)
            data_stddev = (0.2023, 0.1994, 0.2010)
        elif self.dataset_name == "CIFAR100":
            data_root = os.path.join(self.root, "CIFAR100")
            dataset = torchvision.datasets.CIFAR100
            data_mean = (0.5071, 0.4867, 0.4408)
            data_stddev = (0.2675, 0.2565, 0.2761)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
            ])

        train_set = dataset(root=data_root, train=True, download=True, transform=transform)
        test_set = dataset(root=data_root, train=False, download=True, transform=transform)

        print("---loaded datasets")
        return train_set, test_set
