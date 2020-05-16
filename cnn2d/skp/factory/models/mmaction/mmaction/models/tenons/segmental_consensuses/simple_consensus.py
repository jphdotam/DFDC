import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import SEGMENTAL_CONSENSUSES

class _SimpleConsensus(torch.autograd.Function):
    """Simplest segmental consensus module"""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.mean(dim=1, keepdim=True)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_in = grad_output.expand(input.shape) / float(input.shape[1])
        return grad_in

@SEGMENTAL_CONSENSUSES.register_module
class SimpleConsensus(nn.Module):
    def __init__(self, consensus_type, dim=1):
        super().__init__()

        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim

    def init_weights(self):
        pass

    def forward(self, input):
        return _SimpleConsensus.apply(input)