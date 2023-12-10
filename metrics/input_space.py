import torch
import torch.nn.functional as F
from torch.autograd import Variable

class InputSpaceMetrics:
    def __init__(self, config):
        self.device = config.device

    def calculate_all(self, model):
        input_space_results = dict()

        input_space_results["lipschitz_constant"] = self.lipschitz_constant_power_method(model)

        return input_space_results
        
    def lipschitz_constant_power_method(self, model, eps=1e-8, max_iter=500):
        input_size = [1, 3, 32, 32]

        model = model.to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad = False

        zeros = torch.zeros(input_size)
        zeros = zeros.to(self.device)
        bias = model(Variable(zeros))
        linear_fun = lambda x: model(x) - bias

        def norm(x, p=2):
            norms = Variable(torch.zeros(x.shape[0]))
            norms = norms.to(self.device)
            for i in range(x.shape[0]):
                norms[i] = x[i].norm(p=p)
            return norms

        v = torch.randn(input_size)
        v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_size)

        v = v.to(self.device)

        stop_criterion = False
        it = 0
        while not stop_criterion:
            previous = v
            v = self._norm_gradient_sq(linear_fun, v)
            v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_size)
            stop_criterion = (torch.norm(v - previous) < eps) or (it > max_iter)
            it += 1

        u = linear_fun(Variable(v))
        eigenvalue = norm(u).cpu().item()
        u = u.div(eigenvalue)
        return eigenvalue

    def _norm_gradient_sq(self, linear_fun, v):
        v = Variable(v, requires_grad=True)
        loss = torch.norm(linear_fun(v))**2
        loss.backward()
        return v.grad.data

