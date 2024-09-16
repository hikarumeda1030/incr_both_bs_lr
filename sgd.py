import torch
from torch.optim import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr) -> None:
        if lr < 0.0:
            raise ValueError(f"Learning rate must be non-negative, but got {lr}.")
        defaults = dict(lr=lr)
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)

    def step(self, closure=None, iteration=0):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if iteration != 0:
            gradient_list = []

        for group in self.param_groups:
            lr = group['lr']

            for param in group['params']:
                if param.grad is None:
                    continue
                gradient = param.grad.data

                if iteration != 0:
                    gradient_list.append(gradient)

                param.data.add_(gradient, alpha=-lr)

        if iteration != 0:
            return gradient_list
        else:
            return loss
