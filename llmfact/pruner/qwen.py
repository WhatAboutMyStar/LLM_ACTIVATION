import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

class PrunedQwen2MLP(nn.Module):
    def __init__(self, config, mask_mlp=None, device=None):
        super().__init__()
        self.config = config
        if mask_mlp is not None:
            num_remain = mask_mlp.sum().item()
            self.hidden_size = config.hidden_size
            self.intermediate_size = num_remain
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, device=device)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, device=device)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, device=device)
            self.act_fn = nn.SiLU()
        else:
            raise ValueError("Must have mlp mask.")

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def pruned_qwen_mlp(model, mask):
    mask = torch.tensor(mask, dtype=torch.bool)

    for i in range(len(model.model.layers)):
        layer = model.model.layers[i]

        mask_1 = mask[i].type(torch.bool)
        pruned_mlp = PrunedQwen2MLP(config=model.config,
                                    mask_mlp=mask_1,
                                    device=next(layer.parameters()).device)
        with torch.no_grad():
            w1 = layer.mlp.gate_proj.weight[mask_1]
            pruned_mlp.gate_proj.weight.copy_(w1.contiguous())

            w2 = layer.mlp.up_proj.weight[mask_1]
            pruned_mlp.up_proj.weight.copy_(w2.contiguous())

            w3 = layer.mlp.down_proj.weight[:, mask_1]
            pruned_mlp.down_proj.weight.copy_(w3.contiguous())

            del layer.mlp
            del w1
            del w2
            del w3
            gc.collect()
            torch.cuda.empty_cache()

            layer.mlp = pruned_mlp

    return model