import copy
import os
import warnings
from typing import Callable, Dict, Tuple
import pyhessian
import torch
import torchvision.models as models
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import gc
import numpy as np



class ParameterSpaceMetrics:
    def __init__(self, config):
        self.device = config.device
        self.hessian_batch_size = config.batch_size
        self.measure = config.param_space
        self.experiment_name = config.experiment_name


    def calculate_all(self, model, theta_0, train_set, test_set):
        model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = True

        parameter_space_results = dict()

        if self.measure:
            #parameter_space_results["hessian_eigenvalue_spectrum_density"] = self.hessian_eigenvalue_spectrum_density(model, test_set, self.hessian_batch_size)
            parameter_space_results["hessian_top_eigenvalue"] = self.hessian_top_eigenvalue(model, test_set, self.hessian_batch_size)
            parameter_space_results["hessian_trace"] = self.hessian_trace(model, test_set, self.hessian_batch_size)

            # Resetting the grad values. This is apparently necessary after using loss.backward(create_graph=True), 
            # which seems to be used by pyhessian in order to compute higher order derivatives.
            for param in model.parameters():
                param.grad = None

            dataset = train_set  # FIXME: in the paper, they say to use the training dataset: is that correct tho?
            loss_function = torch.nn.CrossEntropyLoss()

            # model_accuracy = self._get_accuracy(model, DataLoader(train_set, batch_size=64, shuffle=False))
            model_accuracy = self._get_loss(model, DataLoader(train_set, batch_size=self.hessian_batch_size, shuffle=False), loss_function)
            
            sharpness_flatness, sharpness_init, sharpness_orig, sharpness_mag_flat, sharpness_mag_init, sharpness_mag_orig = self.sharpness_measures(model, theta_0, loss_function, dataset, model_accuracy)
            
            parameter_space_results["sharpness_flatness"] = sharpness_flatness
            parameter_space_results["sharpness_init"] = sharpness_init
            parameter_space_results["sharpness_orig"] = sharpness_orig
            parameter_space_results["sharpness_mag_flat"] = sharpness_mag_flat
            parameter_space_results["sharpness_mag_init"] = sharpness_mag_init
            parameter_space_results["sharpness_mag_orig"] = sharpness_mag_orig

            pac_bayes_flatness, flatness_init, flatness_orig, pac_bayes_mag_flat, flatness_mag_init, flatness_mag_orig = self.flatness_measures(model, theta_0, loss_function, dataset, model_accuracy)

            parameter_space_results["pac_bayes_flatness"] = pac_bayes_flatness
            parameter_space_results["flatness_init"] = flatness_init
            parameter_space_results["flatness_orig"] = flatness_orig
            parameter_space_results["pac_bayes_mag_flat"] = pac_bayes_mag_flat
            parameter_space_results["flatness_mag_init"] = flatness_mag_init
            parameter_space_results["flatness_mag_orig"] = flatness_mag_orig

        return parameter_space_results

    def hessian_top_eigenvalue(self, model, test_set, batch_size):
        hessian_module = self._create_hessian_module(model, test_set, batch_size)
        top_eigenvalue = hessian_module.eigenvalues(top_n=1)[0][0]
        model.zero_grad()
        del hessian_module
        gc.collect()
        print("---found Hessian maximum eigenvalue")
        return top_eigenvalue

    def hessian_trace(self, model, test_set, batch_size):
        hessian_module = self._create_hessian_module(model, test_set, batch_size)
        trace = np.mean(hessian_module.trace())
        model.zero_grad()
        del hessian_module
        gc.collect()
        print("---found Hessian trace")
        return trace
    
    def hessian_eigenvalue_spectrum_density(self, model, test_set, batch_size):
        hessian_module = self._create_hessian_module(model, test_set, batch_size)
        eigenvalues, weights = hessian_module.density()
        density, grids = self._generate_density(eigenvalues, weights)
        fig, ax = plt.subplots()
        ax.semilogy(grids, density + 1.0e-7)
        ax.set_ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
        ax.set_xlabel('Eigenvalue', fontsize=14, labelpad=10)
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

    def _get_accuracy(self, name, model: torch.nn.Module,
                        loader: torch.utils.data.DataLoader) -> float:
        """
        Compute the empirical loss of a given model on a dataset.
        Check if the accuracy was previously calculated and saved. If so, load and return it.
        Otherwise, calculate, save, and return the accuracy.
        """
        model.eval()

        """
        path = os.path.join(os.getcwd(), "experiments", self.experiment_name, "accuracies")
        filename = os.path.join(path, f"{name}_accuracy.pt")

        # Check if the accuracy has already been computed and saved
        if os.path.exists(filename):
            return torch.load(filename)
        """
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_accuracy = correct / total
        
        # Save the computed loss
        """
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(avg_accuracy, filename)
        """
        return avg_accuracy

    def _estimate_accuracy(self, model: torch.nn.Module,
                        loader: torch.utils.data.DataLoader,
                        M: int) -> float:
        """
        Estimate the accuracy of a given model on a dataset.
        """
        model.eval()
        correct = 0
        total = 0

        iter_loader = iter(loader)
        with torch.no_grad():
            for _ in range(M):
                inputs, labels = next(iter_loader)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return correct / total
    
    def _get_loss(self, model: torch.nn.Module, loader: torch.utils.data.DataLoader, loss_function: Callable) -> float:
        """
        Compute the empirical loss of a given model on a dataset.
        Check if the loss was previously calculated and saved. If so, load and return it.
        Otherwise, calculate, save, and return the loss.
        """
        model.eval()
        """
        path = os.path.join(os.getcwd(), "experiments", self.experiment_name, "losses")
        filename = os.path.join(path, f"{name}_loss.pt")

        # Check if the loss has already been computed and saved
        if os.path.exists(filename):
            return torch.load(filename)
        """
        loss = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = model(inputs)
                loss += loss_function(outputs, labels)
                total += labels.size(0)

        avg_loss = loss / total

        """
        # Save the computed loss
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(avg_loss, filename)
        """
        return avg_loss

    def _estimate_loss(self, model: torch.nn.Module,
                        loader: torch.utils.data.DataLoader,
                        loss_function: Callable,
                        M: int) -> float:
        """
        Estimate the empirical loss of a given model on a dataset.
        """
        model.eval()
        loss = 0
        total = 0

        iter_loader = iter(loader)
        with torch.no_grad():
            for _ in range(M):
                inputs, labels = next(iter_loader)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = model(inputs)
                loss += loss_function(outputs, labels)
                total += labels.size(0)

        return loss / total
    
    def _grad_ascent_step(self, iter_loader, model, theta, sigma_new, loss_function, lr, magnitude_aware):
        # Sample a minibatch uniformly at random from the dataset
        inputs, labels = next(iter_loader)
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        model.zero_grad()
        outputs = model(inputs)

        loss = loss_function(outputs, labels)
        loss.backward()

        # Perform a gradient ascent step
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.add_(lr * param.grad)

                if magnitude_aware:
                    # Clip parameter if |param[i]| > original_param[i] + sigma_new * (|original_param[i]| + 1)
                    for name, param in model.named_parameters():
                        original_param = theta[name]
                        max_perturb = (torch.abs(original_param) + 1) * sigma_new
                        lower_bound = original_param - max_perturb
                        upper_bound = original_param + max_perturb
                        clipped_param = torch.max(torch.min(param, upper_bound), lower_bound)
                        param.data.copy_(clipped_param)

            
            # Alternative clippling method. This one requires all parameters to be up to date, prior to the evaluation of the condition (hence why it is outside the for loop).
            if not magnitude_aware:
                perturbation = {}
                perturbation_norm = 0
                for name, param in model.named_parameters():
                    value = param - theta[name]
                    perturbation[name] = value
                    perturbation_norm += (value**2).sum().item()
                perturbation_norm = np.sqrt(perturbation_norm)

                if perturbation_norm > sigma_new:
                    scaling_factor = sigma_new / perturbation_norm
                    for name, param in model.named_parameters():
                        scaled_pertubation = perturbation[name] * scaling_factor
                        param.data.copy_(theta[name] + scaled_pertubation)

    def sigma_sharpness_bound(self, model: torch.nn.Module,
                          loss_function: Callable,
                          loader: torch.utils.data.DataLoader,
                          magnitude_aware: bool,
                          binary_search_depth: int, monte_carlo_steps: int, M3: int, grad_asc_steps: int,
                          sigma_max: float, sigma_min: float,
                          model_accuracy: float, target_acc_deviation: float,
                          lr: float = 0.01, epsilon_d: float = 1e-2, epsilon_sigma: float = 5e-3) -> float:
        """
        Estimate the sigma for the sharpness bound.
        """
        theta = copy.deepcopy(model.state_dict())

        for _ in range(binary_search_depth):
            sigma_new = (sigma_max + sigma_min) / 2
            l_hat = np.inf

            for _ in range(monte_carlo_steps):
                # Add perturbation to the model parameters.
                theta_new = {}
                for name, param in theta.items():
                    theta_new[name] = param + (torch.rand(param.size(), device=self.device) - 0.5) * sigma_new
                model.load_state_dict(theta_new)

                # Perform gradient ascent step.
                model.train()
                iter_loader = iter(loader)
                for _ in range(grad_asc_steps):
                    self._grad_ascent_step(iter_loader, model, theta, sigma_new, loss_function, lr, magnitude_aware)
                model.eval()

                # Compute estimation of the model's accuracy.
                #FIXME: USE ACCURACY OR LOSS?
                # l_hat = min(l_hat, self._estimate_accuracy(model, loader, M3))
                l_hat = min(l_hat, self._estimate_loss(model, loader, loss_function, M3))

            # Compute estimation of the generalization gap.
            d_hat = abs(model_accuracy - l_hat)

            # Check the stopping condition
            if abs(d_hat - target_acc_deviation) < epsilon_d or sigma_max - sigma_min < epsilon_sigma:
                # Restore the original state model
                model.load_state_dict(theta)
                return sigma_new

            # Binary search update
            if d_hat > target_acc_deviation:
                sigma_max = sigma_new
            else:
                sigma_min = sigma_new

        warnings.warn("Algorithm did not converge within the specified binary search depth.", RuntimeWarning)
        model.load_state_dict(theta)
        return sigma_new

    def sharpness_measures(self, model: torch.nn.Module,
                       theta_0: Dict[str, torch.Tensor],
                       loss_function: Callable,
                       dataset: torch.utils.data.Dataset,
                       model_accuracy: float,
                       delta: float = 0.05,
                       binary_search_depth: int = 20, monte_carlo_steps: int = 15, M3: int = 10, grad_asc_steps: int = 20,
                       sigma_max: float = 2.0, sigma_min: float = 1e-5,
                       target_acc_deviation: float = 0.1,
                       batch_size: int = 64) -> Tuple[float, float, float, float, float, float]:
        """
        Compute various sharpness measures for a neural network model.
        """
        m = len(dataset)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Compute sigma for the sharpness bound (in the paper, they also call it alpha)
        sigma = self.sigma_sharpness_bound(model, loss_function, loader, False, binary_search_depth, monte_carlo_steps, M3, grad_asc_steps, sigma_max, sigma_min, model_accuracy, target_acc_deviation)

        with torch.no_grad():
            n_params = sum(p.numel() for p in model.parameters())

            # 1st Measure: "sharpness-flatness"
            sharpness_flatness = 1/(sigma**2)

            # 2nd Measure: "sharpness-init"
            theta_square_dist = 0
            for name, param in model.named_parameters():
                param_init = theta_0[name]
                theta_square_dist += ((param - param_init)**2).sum().item()

            sharpness_init = theta_square_dist * np.log(2 * n_params / delta) / (sigma**2) + np.log(m/delta) + 10
            sharpness_init = 1

            # 3rd Measure: "sharpness-orig"
            theta_square_norm = sum(p.pow(2.0).sum() for p in model.parameters()).item()

            sharpness_orig = theta_square_norm * np.log(2 * n_params / delta) / (sigma**2) + np.log(m/delta) + 10
        
        ##############################
        # Magnitude-aware Measurements
        ##############################

        sigma_mag = self.sigma_sharpness_bound(model, loss_function, loader, True, binary_search_depth, monte_carlo_steps, M3, grad_asc_steps, delta, sigma_max, sigma_min, model_accuracy, target_acc_deviation)

        with torch.no_grad():
            # 4th Measure: "sharpness-mag-flat"
            sharpness_mag_flat = 1/(sigma_mag**2)

            # 5th Measure: "sharpness-mag-init"
            epsilon = 1e-3  # Value chosen in the paper.
            d_KL = 0
            for name, param in model.named_parameters():
                param_init = theta_0[name]
                numerator = epsilon**2 + (sigma_mag**2 + 4*np.log(2*n_params / delta)) * theta_square_dist / n_params
                denominator = epsilon**2 + (sigma_mag * (param - param_init))**2
                d_KL += (torch.log(numerator/denominator)).sum().item()

            # sharpness_mag_init = d_KL/2 + np.log(m/delta) + 10
            sharpness_mag_init = 1


            # 6th Measure: "sharpness-mag-orig"
            tmp = 0
            for param in model.parameters():
                numerator = epsilon**2 + (sigma_mag**2 + 4*np.log(2*n_params / delta)) * theta_square_norm / n_params
                denominator = epsilon**2 + (sigma_mag * param)**2
                tmp += (torch.log(numerator/denominator)).sum().item()

            sharpness_mag_orig = tmp/2 + np.log(m/delta) + 10

        print("---found sharpness metrics")

        return sharpness_flatness, sharpness_init, sharpness_orig, sharpness_mag_flat, sharpness_mag_init, sharpness_mag_orig

    def sigma_flatness_bound(self, model: torch.nn.Module,
                          loss_function: Callable,
                          loader: torch.utils.data.DataLoader,
                          magnitude_aware: bool,
                          epsilon: float,
                          binary_search_depth: int, monte_carlo_steps: int, M3: int,
                          sigma_max: float, sigma_min: float,
                          model_accuracy: float, target_acc_deviation: float,
                          epsilon_d: float = 1e-2, epsilon_sigma: float = 5e-3) -> float:
        """
        Estimate the sigma for the sharpness bound.
        """
        model.eval()
        theta = copy.deepcopy(model.state_dict())

        for _ in range(binary_search_depth):
            sigma_new = (sigma_max + sigma_min) / 2
            l_hat = 0

            for _ in range(monte_carlo_steps):
                # Add perturbation to the model parameters.
                theta_new = {}
                for name, param in theta.items():
                    if magnitude_aware:
                        # FIXME: WHICH ONE TO USE?
                        # std = sigma_new * torch.abs(param) + epsilon          #  Alternative way of defining it.
                        std = torch.sqrt((sigma_new * param)**2 + epsilon**2)   # the way is defined in the paper "Fantastic generalization measures"
                        # std = sigma_new * (torch.abs(param) + epsilon)          # The way it was defined in the original paper (of sharpness)
                        perturbation = torch.normal(0.0, std).to(self.device)
                    else:
                        perturbation = torch.normal(0.0, sigma_new, param.shape).to(self.device)
                    theta_new[name] = param + perturbation
                model.load_state_dict(theta_new)

                # Compute model's accuracy.
                #FIXME: USE ACCURACY OR LOSS?
                # l_hat += self._estimate_accuracy(model, loader, M3)
                l_hat += self._estimate_loss(model, loader, loss_function, M3)
            
            l_hat /= monte_carlo_steps

            # Compute estimation of the generalization gap.
            d_hat = abs(model_accuracy - l_hat)

            # Check the stopping condition
            if abs(d_hat - target_acc_deviation) < epsilon_d or sigma_max - sigma_min < epsilon_sigma:
                # Restore the original state model
                model.load_state_dict(theta)
                return sigma_new

            # Binary search update
            if d_hat > target_acc_deviation:
                sigma_max = sigma_new
            else:
                sigma_min = sigma_new

        warnings.warn("Algorithm did not converge within the specified binary search depth.", RuntimeWarning)
        model.load_state_dict(theta)
        return sigma_new

    @torch.no_grad()
    def flatness_measures(self, model: torch.nn.Module,
                       theta_0: Dict[str, torch.Tensor],
                       loss_function: Callable,
                       dataset: torch.utils.data.Dataset,
                       model_accuracy: float,
                       delta: float = 0.05,
                       binary_search_depth: int = 30, monte_carlo_steps: int = 20, M3: int = 20,
                       sigma_max: float = 2.0, sigma_min: float = 1e-5,
                       target_acc_deviation: float = 0.1,
                       batch_size: int = 64) -> Tuple[float, float, float, float, float, float]:
        """
        Compute various flatness measures for a neural network model.
        """
        epsilon = 1e-3
        m = len(dataset)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        sigma = self.sigma_flatness_bound(model, loss_function, loader, False, epsilon, binary_search_depth, monte_carlo_steps, M3, sigma_max, sigma_min, model_accuracy, target_acc_deviation)

        n_params = sum(p.numel() for p in model.parameters())   # Number of parameters.

        # 1st Measure: "pac-bayes-flatness"
        pac_bayes_flatness = 1/(sigma**2)

        # 2nd Measure: "flatness-init"
        theta_square_dist = 0
        for name, param in model.named_parameters():
            param_init = theta_0[name]
            theta_square_dist += ((param - param_init)**2).sum().item()

        flatness_init = theta_square_dist / (2 * sigma**2) + np.log(m/delta) + 10
        flatness_init = 1

        # 3rd Measure: "flatness-orig"
        theta_square_norm = sum(p.pow(2.0).sum() for p in model.parameters()).item()

        flatness_orig = theta_square_norm / (2 * sigma**2) + np.log(m/delta) + 10
    
        ##############################
        # Magnitude-aware Measurements
        ##############################

        sigma_mag = self.sigma_flatness_bound(model, loss_function, loader, True, epsilon, binary_search_depth, monte_carlo_steps, M3, sigma_max, sigma_min, model_accuracy, target_acc_deviation)

        # 4th Measure: "flatness-mag-flat"
        pac_bayes_mag_flat = 1/(sigma_mag**2)

        # 5th Measure: "flatness-mag-init"
        d_KL = 0
        for name, param in model.named_parameters():
            param_init = theta_0[name]
            numerator = epsilon**2 + (sigma_mag**2 + 1) * theta_square_dist / n_params
            denominator = epsilon**2 + (sigma_mag * (param - param_init))**2
            d_KL += (torch.log(numerator/denominator)).sum().item()

        flatness_mag_init = d_KL/2 + np.log(m/delta) + 10
        flatness_mag_init = 1


        # 6th Measure: "flatness-mag-orig"
        d_KL = 0
        for param in model.parameters():
            numerator = epsilon**2 + (sigma_mag**2 + 1) * theta_square_norm / n_params
            denominator = epsilon**2 + (sigma_mag * param)**2
            d_KL += (torch.log(numerator/denominator)).sum().item()    # TODO: SEE IF NOT ADDING CAUSE INSTABILITYadd 1e-8 to avoid log(0).

        flatness_mag_orig = d_KL/2 + np.log(m/delta) + 10

        print("---found flatness metrics")
        return pac_bayes_flatness, flatness_init, flatness_orig, pac_bayes_mag_flat, flatness_mag_init, flatness_mag_orig


