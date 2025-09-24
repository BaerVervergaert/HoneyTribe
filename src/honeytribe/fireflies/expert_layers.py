import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertSignLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, include_zero: bool = True, bias: bool = True, learning_rate: float = 0.1, alpha=.1):
        super().__init__()
        self.alpha = alpha
        self.include_zero = include_zero
        self.expert_prob = torch.ones((2 if not include_zero else 3, output_dim, input_dim))
        self.expert_prob = self.expert_prob / self.expert_prob.sum(dim=0, keepdim=True)
        self.matrix = nn.Parameter(torch.randn((2, output_dim, input_dim)))
        self.bias = nn.Parameter(torch.randn((output_dim,))) if bias else 0.
        self.transform = lambda x: torch.abs(x)
        self._last_choice = None
        self.learning_rate = learning_rate

    def forward(self, x):
        if self.training:
            cum_prob = self.expert_prob.cumsum(dim=0)
            rand = torch.rand(self.expert_prob.shape[1:])
            choice = (cum_prob >= rand[None, :]).int().argmax(dim=0)
        else:
            choice = self.expert_prob.argmax(dim=0)
        matrix = torch.where(choice == 0, self.matrix[0, :, :], self.matrix[1, :, :])
        matrix = self.transform(matrix)
        matrix = matrix * (-1)**choice * (choice != 2)
        x = F.linear(x, matrix, self.bias)
        self._last_choice = choice
        return x

    def update(self, loss):
        if self._last_choice is None:
            raise ValueError("No forward pass has been made yet.")
        if not self.training:
            raise ValueError("Update can only be called in training mode.")
        loss = loss.detach().mean()
        # Use EXP3 update rule
        with torch.no_grad():
            for i in range(self.expert_prob.shape[1]):
                for j in range(self.expert_prob.shape[2]):
                    choice = self._last_choice[i, j]
                    p = self.expert_prob[choice, i, j]
                    estimated_loss = loss / p
                    self.expert_prob[choice, i, j] *= torch.exp(-self.learning_rate * estimated_loss)
            K = (3 if self.include_zero else 2)
            beta = self.alpha * K / (K - 1)
            self.expert_prob = (1 - beta) * self.expert_prob + beta / K
            self.expert_prob = self.expert_prob / self.expert_prob.sum(dim=0, keepdim=True)
        self._last_choice = None