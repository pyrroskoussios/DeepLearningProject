import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        if not use_batchnorm:
            self.bn1 = self.bn2 = self.bn3 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes) if use_batchnorm else nn.Sequential()
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
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

def ResNet34(num_classes=10, use_batchnorm=True, linear_bias=True):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, use_batchnorm=use_batchnorm, linear_bias=linear_bias)

def ResNet50(num_classes=10, use_batchnorm=True, linear_bias=True):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, use_batchnorm=use_batchnorm, linear_bias=linear_bias)

def ResNet101(num_classes=10, use_batchnorm=True, linear_bias=True):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, use_batchnorm=use_batchnorm, linear_bias=linear_bias)

def ResNet152(num_classes=10, use_batchnorm=True, linear_bias=True):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, use_batchnorm=use_batchnorm, linear_bias=linear_bias)

class ModelLoader:
    def __init__(self, config):
        self.device = config.device
        self.experiment_name = config.experiment_name
        self.model_type = config.model_type
        self.dataset_name = config.dataset_name
        self.root = "/content/drive/MyDrive/DeepLearningProject" if config.colab else os.getcwd()

    def load_models(self, prefix):
        experiment_path = os.path.join(self.root, "experiments", self.experiment_name, str(self.experiment_name + f"_seed{prefix}"))

        parent_one = (self._load_individual_model(os.path.join(experiment_path, "parent_1.pth")), f"parent_1_seed{prefix}")
        parent_two = (self._load_individual_model(os.path.join(experiment_path, "parent_2.pth")), f"parent_2_seed{prefix}")
        fused_naive = (self._load_individual_model(os.path.join(experiment_path, "naive_fused.pth")), f"naive_fused_seed{prefix}")
        fused_geometric = (self._load_individual_model(os.path.join(experiment_path, "geometric_fused.pth")), f"geometric_fused_seed{prefix}")

        print("---loaded model family")
        return parent_one, parent_two, fused_naive, fused_geometric

    def load_initial_weights(self, prefix):
        """
        Load initial weights of the models. These will be used during the computation of the sharpness metrics.
        """
        experiment_path = os.path.join(self.root, "experiments", self.experiment_name, str(self.experiment_name + f"_seed{prefix}"))
        initial_weights = {}

        # Load initial weights of the parents
        for name in ["parent_1", "parent_2"]:
            path = os.path.join(experiment_path, f"{name}_initial_weights.pth")
            initial_weights[str(name + f"_seed{prefix}")]= torch.load(path, map_location=(lambda s, _: torch.serialization.default_restore_location(s, self.device)))


        # Load initial weights of the fused models
        # We define as "initial" weights, the weights of the parent for which the square distance is the largest.
        # To compute the distances, we first need to load the parents weights.
        path = os.path.join(experiment_path, "parent_1.pth")
        theta_1 = torch.load(path, map_location=(lambda s, _: torch.serialization.default_restore_location(s, self.device)))

        path = os.path.join(experiment_path, "parent_2.pth")
        theta_2 = torch.load(path, map_location=(lambda s, _: torch.serialization.default_restore_location(s, self.device)))

        for name in ["naive_fused", "geometric_fused"]:
            path = os.path.join(experiment_path, f"{name}.pth")
            theta_fused = torch.load(path, map_location=(lambda s, _: torch.serialization.default_restore_location(s, self.device)))

            theta_square_dist_1 = 0
            for param_name, param in theta_fused.items():
                param_init = theta_1[param_name]
                theta_square_dist_1 += ((param - param_init)**2).sum().item()
            
            theta_square_dist_2 = 0
            for param_name, param in theta_fused.items():
                param_init = theta_2[param_name]
                theta_square_dist_2 += ((param - param_init)**2).sum().item()
            
            if theta_square_dist_1 > theta_square_dist_2:
                initial_weights[str(name + f"_seed{prefix}")] = theta_1
            else:
                initial_weights[str(name + f"_seed{prefix}")] = theta_2

        print("---loaded initial model family")
        return initial_weights

    def _load_individual_model(self, path):
        state = torch.load(path, map_location=(lambda s, _: torch.serialization.default_restore_location(s, self.device)))
        
        num_classes = 10 if self.dataset_name == "CIFAR10" else 100
        use_bias = "NOBIAS" not in self.model_type
        use_bn = "NOBN" not in self.model_type

        if "RESNET18" in self.model_type:
            model = ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, use_batchnorm=use_bn, linear_bias=use_bias)
        if "VGG11" in self.model_type:
            model = VGG("VGG11", num_classes, batch_norm=use_bn, bias=use_bias, relu_inplace=True)
        
        try:
            #model.load_state_dict(torch.load(path))
            model.load_state_dict(state)
        except RuntimeError as original_error:
            print(original_error)
            print(
                "\n\n\nOopsie woopsie youre a little dumb and tried to load saved weights into a network of different shape hihi! Check the original error above for leads! (most likely a difference in batchnorm or bias)\n\n\n")
            exit()

        return model



