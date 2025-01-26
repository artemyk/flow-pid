import torch
from torch import nn
from torch.nn import functional as F
import pdb
from torch.autograd import Variable


class Concat(nn.Module):
    """Concatenation of input data on dimension 1."""

    def __init__(self):
        """Initialize Concat Module."""
        super(Concat, self).__init__()

    def forward(self, modalities):
        """
        Forward Pass of Concat.

        :param modalities: An iterable of modalities to combine
        """
        flattened = []
        # print(len(modalities), len(modalities[0]), len(modalities[0][0]))
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=1)



class Sequential2(nn.Module):
    """Implements a simpler version of sequential that handles inputs with 2 arguments."""
    
    def __init__(self, a, b):
        """Instatiate Sequential2 object.

        Args:
            a (nn.Module): First module to sequence
            b (nn.Module): Second module
        """
        super(Sequential2, self).__init__()
        self.model = nn.Sequential(a, b)

    def forward(self, x):
        """Apply Sequential2 modules to layer input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return self.model(x)