import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

from llmfact.extractor import LayerOutputExtractor
from tqdm import trange

class PrunedLlamaMLP(nn.Module):
    def __init__(self, config, mask=None, device=None):
        super().__init__()
        self.config = config
        self.mask = mask

        num_remaining_mlp = self.mask.sum().item()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, num_remaining_mlp, bias=config.mlp_bias, device=device)
        self.up_proj = nn.Linear(self.hidden_size, num_remaining_mlp, bias=config.mlp_bias, device=device)
        self.down_proj = nn.Linear(num_remaining_mlp, self.hidden_size, bias=config.mlp_bias, device=device)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

def pruned_llama_mlp(model, mask):
    mask = torch.tensor(mask, dtype=torch.bool)

    for i in range(len(model.model.layers)):
        layer = model.model.layers[i]

        mask_1 = mask[i].type(torch.bool)
        # pruned_mlp = PrunedLlamaMLP(config=model.config,
        #                             mask=mask_1,
        #                             device=next(layer.parameters()).device)
        with torch.no_grad():
            # w1 = layer.mlp.gate_proj.weight[mask_1]
            layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mask_1)[0]]
            # pruned_mlp.gate_proj.weight.copy_(w1.contiguous())

            # w2 = layer.mlp.up_proj.weight[mask_1]
            layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mask_1)[0]]
            # pruned_mlp.up_proj.weight.copy_(w2.contiguous())

            layer.mlp.up_proj.out_features = mask_1.sum().item()
            layer.mlp.gate_proj.out_features = mask_1.sum().item()
            layer.mlp.intermediate_size = mask_1.sum().item()

            output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mask_1)[0]]

            layer.mlp.down_proj.weight.data = output_weight

            layer.mlp.down_proj.in_features = mask_1.sum().item()

            # w3 = layer.mlp.down_proj.weight[:, mask_1]
            # pruned_mlp.down_proj.weight.copy_(w3.contiguous())

            # # del layer.mlp
            # del w1
            # del w2
            # del w3
            gc.collect()
            torch.cuda.empty_cache()

            # layer.mlp = pruned_mlp

    return model

def pruned_llama_attention(model, mask):
    """
    :param model: llama model huggingface
    :param mask: mask shape (32, 4096)
    :return: model
    """
    mask = torch.tensor(mask, dtype=torch.bool)

    for i in range(len(model.model.layers)):
        layer = model.model.layers[i]
        # retain_heads = mask[i].sum() // 128

        layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(mask[i])[0]]
        layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(mask[i])[0]]
        layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(mask[i])[0]]

        layer.self_attn.q_proj.out_features = mask[i].sum().item()
        layer.self_attn.k_proj.out_features = mask[i].sum().item()
        layer.self_attn.v_proj.out_features = mask[i].sum().item()

        layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data[:, torch.where(mask[i])[0]]
        layer.self_attn.o_proj.in_features = mask[i].sum().item()
        # layer.self_attn.num_heads = retain_heads
        # layer.self_attn.hidden_size = retain_heads * 128

    gc.collect()
    torch.cuda.empty_cache()

    return model

class LayerBiasCompute:
    def __init__(self, model, include_layers, tokenizer, mask, dataset, total_layer_num=32):
        self.hooks = None
        self.model = model
        self.include_layers = include_layers
        self.tokenizer = tokenizer
        self.mask = mask #(32, 11008)
        self.dataset = dataset
        self.total_layer_num = total_layer_num
        self.bias_dict = {i:0 for i in range(total_layer_num)}
        self.n_samples = {i:0 for i in range(total_layer_num)}
        self.mean_dict = {i:0 for i in range(total_layer_num)}

    def hook_down(self, module, inputs, outputs, i, mask):
        device = inputs[0].device
        mask = torch.tensor(mask, dtype=torch.bool).to(device)
        new_input = inputs[0] * mask
        new_input_2 = inputs[0] * ~mask
        mean = torch.mean(new_input.reshape((-1, new_input.shape[-1])).T, dim=1)

        self.mean_dict[i] *= self.n_samples[i] / (self.n_samples[i] + 1)
        self.mean_dict[i] += mean / (self.n_samples[i] + 1)

        # bias = mean @ module.weight.data.T

        # self.bias_dict[i] *= self.n_samples[i] / (self.n_samples[i] + 1)
        # self.bias_dict[i] += bias / (self.n_samples[i] + 1)
        self.n_samples[i] += 1

        outputs = new_input_2 @ module.weight.T

        return outputs

    def register_hooks(self, mask):
        self.hooks = []
        def create_hook(i, mask):
            return lambda module, inputs, outputs: self.hook_down(module, inputs, outputs, i, mask[i:i+1, :])

        for i, layer_name in enumerate(self.include_layers):
            module = self.model
            for part in layer_name.split("."):
                module = getattr(module, part)
            hook = module.register_forward_hook(create_hook(i, mask))
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def fit(self):

        self.add_bias()

        inputs_list = [self.tokenizer(inputs, return_tensors="pt", max_length=1024, truncation=True) for inputs in
                       self.dataset]

        self.register_hooks(self.mask)

        for i in trange(len(inputs_list)):
            inputs = inputs_list[i]
            inputs = inputs.to(self.model.device)
            with torch.no_grad():
                _ = self.model(**inputs)

        self.remove_hooks()

        for i in range(3, self.total_layer_num-2):
            self.bias_dict[i] = self.mean_dict[i] @ self.model.model.layers[i].mlp.down_proj.weight.data.T
            self.model.model.layers[i].mlp.down_proj.bias.data = nn.Parameter(self.bias_dict[i])


    def add_bias(self):
        for i in range(self.total_layer_num):
            self.model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(
                torch.zeros(4096,
                            device=self.model.model.layers[i].mlp.down_proj.weight.device,
                            dtype=self.model.model.layers[i].mlp.down_proj.weight.dtype)
            )
            torch.nn.init.zeros_(self.model.model.layers[i].mlp.down_proj.bias)