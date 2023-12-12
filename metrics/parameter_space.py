import torch
import numpy as np
import pyhessian
from matplotlib import pyplot as plt

class ParameterSpaceMetrics:
    def __init__(self, config):
        self.device = config.device
        self.hessian_batch_size = config.hessian_batch_size

    def calculate_all(self, model, test_set):
        model.eval()
        input_space_results = dict()
        #input_space_results["hessian_eigenvalue_spectrum_density"] = self.hessian_eigenvalue_spectrum_density(model, test_set, self.hessian_batch_size)
        input_space_results["hessian_top_eigenvalue"] = self.hessian_top_eigenvalue(model, test_set, self.hessian_batch_size)
        input_space_results["hessian_trace"] = self.hessian_trace(model, test_set, self.hessian_batch_size)
        return input_space_results

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
