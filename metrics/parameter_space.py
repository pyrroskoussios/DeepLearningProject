from typing import Callable, Dict, Tuple

import numpy as np
import pyhessian
import torch
import torchvision.models as models
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


class ParameterSpaceMetrics:
    def __init__(self, config):
        self.device = config.device
        self.hessian_batch_size = config.hessian_batch_size

    def calculate_all(self, model, test_set):
        model.eval()
        parameter_space_results = dict()
        #parameter_space_results["hessian_eigenvalue_spectrum_density"] = self.hessian_eigenvalue_spectrum_density(model, test_set, self.hessian_batch_size)
        parameter_space_results["hessian_top_eigenvalue"] = self.hessian_top_eigenvalue(model, test_set, self.hessian_batch_size)
        parameter_space_results["hessian_trace"] = self.hessian_trace(model, test_set, self.hessian_batch_size)


        # FIXME: theta_0 should be the initial weights used during training.
        theta_0 = models.resnet18(pretrained=False).state_dict()
        loss_function = torch.nn.CrossEntropyLoss()
        # FIXME: in the paper, they say to use the training dataset: is that correct tho?
        dataset = test_set
        # TODO: MAYBE CREATE A LIST WITH ALL THE ACCURACIES THAT APPEAR ON THE PAPER, AND CHOOSE THE CORRECT VALUE DEPENDING ON THE MODEL EVALUATED
        model_accuracy = 0.9
        sharpness_flatness, sharpness_init, sharpness_orig = self.sharpness_measures(model, model.state_dict(), theta_0, loss_function, dataset, model_accuracy, self.device)
        
        parameter_space_results["sharpness_flatness"] = sharpness_flatness
        parameter_space_results["sharpness_init"] = sharpness_init
        parameter_space_results["sharpness_orig"] = sharpness_orig

        return parameter_space_results

    def hessian_top_eigenvalue(self, model, test_set, batch_size):
        hessian_module = self._create_hessian_module(model, test_set, batch_size)
        top_eigenvalue = hessian_module.eigenvalues(top_n=1)[0][0]
        model.zero_grad()
        return top_eigenvalue

    def hessian_trace(self, model, test_set, batch_size):
        hessian_module = self._create_hessian_module(model, test_set, batch_size)
        trace = np.mean(hessian_module.trace())
        model.zero_grad()
        return trace
    
    def hessian_eigenvalue_spectrum_density(self, model, test_set, batch_size):
        hessian_module = self._create_hessian_module(model, test_set, batch_size)
        eigenvalues, weights = hessian_module.density()
        density, grids = self._generate_density(eigenvalues, weights)
        fig, ax = plt.subplots()
        ax.semilogy(grids, density + 1.0e-7)
        ax.set_ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
        ax.set_xlabel('Eigenvlaue', fontsize=14, labelpad=10)
        ax.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
        return fig

    def _create_hessian_module(self, model, test_set, batch_size):
        test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()
        hessian_module = pyhessian.hessian(model, criterion, data=next(iter(test_loader)), cuda=(self.device == "cuda"))
        return hessian_module

    def _generate_density(self, eigenvalues, weights, num_bins=10000, sigma_squared=1e-5, overhead=0.01):
        eigenvalues = np.array(eigenvalues)
        weights = np.array(weights)
        lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
        lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead
        grids = np.linspace(lambda_min, lambda_max, num=num_bins)
        sigma = sigma_squared * max(1, (lambda_max - lambda_min))
        num_runs = eigenvalues.shape[0]
        density_output = np.zeros((num_runs, num_bins))
        for i in range(num_runs):
            for j in range(num_bins):
                x = grids[j]
                tmp_result = np.exp(-(x - eigenvalues[i, :])**2 / (2.0 * sigma)) / np.sqrt(2 * np.pi * sigma)
                density_output[i, j] = np.sum(tmp_result * weights[i, :])
        density = np.mean(density_output, axis=0)
        normalization = np.sum(density) * (grids[1] - grids[0])
        density = density / normalization
        return density, grids

    def _estimate_accuracy(self, model: torch.nn.Module,
                        theta: Dict[str, torch.Tensor],
                        loader: torch.utils.data.DataLoader,
                        M: int,
                        device: torch.device) -> float:
        """
        Estimate the accuracy of a given model on a dataset.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to be evaluated.
        theta : Dict[str, torch.Tensor]
            A dictionary containing the model's parameters.
        loader : torch.utils.data.DataLoader
            DataLoader that provides minibatches of the training dataset.
        M : int
            The number of minibatches to sample from the dataset for estimating accuracy.
        device : torch.device
            The device (e.g., CPU or GPU) on which the computation will be performed.

        Returns
        -------
        float
            The accuracy estimate of the model.
        """
        model.load_state_dict(theta)
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(M):
                minibatch = next(iter(loader))
                inputs, labels = minibatch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return correct / total
    
    def sigma_sharpness_bound(self, model: torch.nn.Module,
                          theta: Dict[str, torch.Tensor],
                          loss_function: Callable,
                          loader: torch.utils.data.DataLoader,
                          M1: int, M2: int, M3: int, M4: int,
                          sigma_max: float, sigma_min: float,
                          model_accuracy: float, accuracy_deviation: float,
                          device: torch.device,
                          lr: float = 3e-3, epsilon_d: float = 0.01, epsilon_sigma: float = 0.01) -> float:
        """
        Estimate the sigma for the sharpness bound.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to be evaluated.
        theta : Dict[str, torch.Tensor]
            A dictionary of the model's parameters.
        loss_function : Callable
            The loss function used for the model training.
        loader : torch.utils.data.DataLoader
            DataLoader providing minibatches of the training dataset.
        M1, M2, M3, M4 : int
            Iteration parameters.
        sigma_max : float
            The maximum value of sigma to start the binary search.
        sigma_min : float
            The minimum value of sigma to start the binary search.
        model_accuracy : float
            The accuracy of the model.
        accuracy_deviation : float
            The target deviation from the accuracy.
        device : torch.device
            The device on which to perform computations.
        lr : float, optional
            Learning rate for gradient ascent, by default 3e-3.
        epsilon_d : float, optional
            The accuracy convergence threshold, by default 0.01.
        epsilon_sigma : float, optional
            The sigma convergence threshold, by default 0.01.

        Returns
        -------
        float
            The estimated sigma for the sharpness bound.
        """
        for _ in range(M1):
            sigma_new = (sigma_max + sigma_min) / 2
            l_hat = np.inf


            for _ in range(M2):
                # Add perturbation to the model parameters
                theta_new = {}
                for key, value in theta.items():
                    theta_new[key] = value + (torch.rand(value.size()) - 0.5) * sigma_new

                model.train()
                for _ in range(M4):
                    # TODO: SHOULD I DO GRADIENT ASCENT THIS WAY? OR SHOULD I USE EPOCHS AND USING THE ENTIRE DATASET?
                    # Sample a minibatch uniformly at random from the dataset
                    minibatch = next(iter(loader))
                    inputs, labels = minibatch
                    inputs, labels = inputs.to(device), labels.to(device)

                    model.zero_grad()
                    outputs = model(inputs)

                    loss = loss_function(outputs, labels)
                    loss.backward()

                    # Perform a gradient ascent step
                    with torch.no_grad():
                        for param in model.parameters():
                            param.add_(lr * param.grad)

                        # Clip the parameters if their norm exceeds sigma_new
                        norm_theta = torch.sqrt(sum(p.pow(2.0).sum() for p in model.parameters()))
                        if norm_theta > sigma_new:
                            for param in model.parameters():
                                param.mul_(sigma_new / norm_theta)


                l_hat = min(l_hat, self._estimate_accuracy(model, theta_new, loader, M3, device))

            d_hat = abs(model_accuracy - l_hat)

            # Check the stopping condition
            if d_hat < epsilon_d or sigma_max - sigma_min < epsilon_sigma:
                return sigma_new

            # Binary search update
            if d_hat > accuracy_deviation:
                sigma_max = sigma_new
            else:
                sigma_min = sigma_new
        
    def sharpness_measures(self, model: torch.nn.Module,
                       theta: Dict[str, torch.Tensor],
                       theta_0: Dict[str, torch.Tensor],
                       loss_function: Callable,
                       dataset: torch.utils.data.Dataset,
                       model_accuracy: float,
                       device: torch.device,
                       delta: float = 0.05,
                       M1: int = 10, M2: int = 5, M3: int = 10, M4: int = 10,
                       sigma_max: float = 0.1, sigma_min: float = 0.001,
                       accuracy_deviation: float = 0.02,
                       batch_size: int = 64) -> Tuple[float, float, float]:
        """
        Compute various sharpness measures for a neural network model.

        This function calculates three sharpness measures for a given model: sharpness-flatness,
        sharpness-init, and sharpness-orig.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to be evaluated.
        theta : Dict[str, torch.Tensor]
            A dictionary of the model's current parameters.
        theta_0 : Dict[str, torch.Tensor]
            A dictionary of the model's initial parameters.
        loss_function : Callable
            The loss function used for the model training.
        dataset : torch.utils.data.Dataset
            The training dataset.
        model_accuracy : float
            The accuracy of the model.
        device : torch.device
            The device on which to perform computations.
        delta : float, optional
            The delta value used in sharpness calculations, by default 0.05.
        M1, M2, M3, M4 : int
            Iteration parameters.
        sigma_max : float
            The maximum value of sigma to start the binary search.
        sigma_min : float
            The minimum value of sigma to start the binary search.
        accuracy_deviation : float
            The target deviation from the accuracy.
        batch_size : int, optional
            Batch size for the DataLoader, by default 64.

        Returns
        -------
        Tuple[float, float, float]
            A tuple containing the values for sharpness-flatness, sharpness-init,
            and sharpness-orig measures.
        """
        model = model.to(device)
        m = len(dataset)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Compute sigma for the sharpness bound
        sigma = self.sigma_sharpness_bound(model, theta, loss_function, loader, M1, M2, M3, M4, sigma_max, sigma_min, model_accuracy, accuracy_deviation, device)

        with torch.no_grad():
            # Compute alpha
            omega = sum(p.numel() for p in model.parameters() if p.requires_grad)   # Number of parameters.
            alpha = sigma * np.sqrt(2*np.log(2 * omega / delta))

            # 1st Measure: "sharpness-flatness"
            sharpness_flatness = 1/(alpha**2)

            # 2nd Measure: "sharpness-init"
            square_dist = 0
            for key in theta:
                param = theta[key]
                param_init = theta_0[key]
                square_dist += ((param - param_init)**2).sum().item()

            # TODO: CHECK TO SEE IF THE FORMULA IS CORRECT, OR IF IT IS np.log(2*omega) instead.
            sharpness_init = square_dist * np.log(2 * omega / delta) / (4 * alpha**2) + np.log(m/delta) + 10

            # 3rd Measure: "sharpness-orig"
            param_square_norm = 0
            for key in theta:
                param = theta[key]
                param_square_norm += (param**2).sum().item()

            # TODO: CHECK TO SEE IF THE FORMULA IS CORRECT, OR IF IT IS np.log(2*omega) instead.
            sharpness_orig = param_square_norm * np.log(2 * omega / delta) / (4 * alpha**2) + np.log(m/delta) + 10

        return sharpness_flatness, sharpness_init, sharpness_orig
