import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.pytorch_utils import Conv1D

def pruned_gpt2_mlp_1(model, masked_matrix):
    masked_matrix = torch.tensor(masked_matrix, dtype=torch.bool)
    for i in range(len(model.transformer.h)):
        layer = model.transformer.h[i]
        if hasattr(layer, "mlp"):
            num_remaining = masked_matrix[i].sum().item()
            if num_remaining == 3072:
                continue

            mask = masked_matrix[i]

            with torch.no_grad():
                pruned_mlp_1 = Conv1D(num_remaining, 768).to(layer.mlp.c_fc.weight.device)
                pruned_mlp_2 = Conv1D(768, num_remaining).to(layer.mlp.c_proj.weight.device)

                w1 = layer.mlp.c_fc.weight[:, mask]
                pruned_mlp_1.weight.requires_grad = False
                pruned_mlp_1.weight.copy_(w1.contiguous())
                pruned_mlp_1.weight.requires_grad = True

                b1 = layer.mlp.c_fc.bias[mask]
                pruned_mlp_1.bias.requires_grad = False
                pruned_mlp_1.bias.copy_(b1.contiguous())
                pruned_mlp_1.bias.requires_grad = True

                layer.mlp.c_fc = pruned_mlp_1

                w2 = layer.mlp.c_proj.weight[mask]
                pruned_mlp_2.weight.requires_grad = False
                pruned_mlp_2.weight.copy_(w2.contiguous())
                pruned_mlp_2.weight.requires_grad = True

                layer.mlp.c_proj = pruned_mlp_2

    return model


class PrunedConv1D(Conv1D):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx, mask):
        super().__init__(nf, nx)
        self.mask = mask

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)

        batch_size = x.shape[1]
        seq_len = x.shape[0]

        output = torch.zeros(seq_len, batch_size, 768, device=x.device, dtype=x.dtype)
        indices = torch.nonzero(self.mask).squeeze()
        output[:, :, indices] = x
        return output


def pruned_gpt2_mlp_2(model, masked_matrix):
    masked_matrix = torch.tensor(masked_matrix, dtype=torch.bool)
    for i in range(len(model.transformer.h)):
        layer = model.transformer.h[i]
        if hasattr(layer, "mlp"):
            num_remaining = masked_matrix[i].sum().item()
            if num_remaining == 768:
                continue

            mask = masked_matrix[i]

            with torch.no_grad():
                pruned_mlp_2 = PrunedConv1D(num_remaining, layer.mlp.c_proj.weight.shape[0], mask).to(
                    layer.mlp.c_proj.weight.device)

                w2 = layer.mlp.c_proj.weight[:, mask]
                pruned_mlp_2.weight.requires_grad = False
                pruned_mlp_2.weight.copy_(w2.contiguous())
                pruned_mlp_2.weight.requires_grad = True

                b2 = layer.mlp.c_proj.bias[mask]
                pruned_mlp_2.bias.requires_grad = False
                pruned_mlp_2.bias.copy_(b2.contiguous())
                pruned_mlp_2.bias.requires_grad = True

                layer.mlp.c_proj = pruned_mlp_2

    return model