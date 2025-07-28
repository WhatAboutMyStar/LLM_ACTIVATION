import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.esm.openfold_utils.tensor_utils import masked_mean
from transformers.pytorch_utils import Conv1D

from llmfact.pruner.gpt2 import pruned_gpt2_mlp_1, pruned_gpt2_mlp_2
from llmfact.pruner.chatglm import pruned_glm_mlp
from llmfact.pruner.llama import pruned_llama_mlp
from llmfact.pruner.qwen import pruned_qwen_mlp

class PrunedGPT2Model:
    def __init__(self, model, mask_mlp_1, mask_mlp_2):
        self.model = model
        self.mask_mlp_1 = mask_mlp_1
        self.mask_mlp_2 = mask_mlp_2

    def fit(self):
        total_par = 0
        for par in self.model.parameters():
            total_par += par.numel()
        print(f"total parameters before pruned: {total_par}")
        self.model = pruned_gpt2_mlp_1(self.model, self.mask_mlp_1)
        self.model = pruned_gpt2_mlp_2(self.model, self.mask_mlp_2)
        total_par_pruned = 0
        for par in self.model.parameters():
            total_par_pruned += par.numel()
        print(f"total parameters after pruned: {total_par_pruned}")
        print(f"total cut num: {total_par - total_par_pruned}")
        print(f"pruned rate: {(total_par - total_par_pruned) / total_par:.4f}")
        return self.model

class PrunedGLMModel:
    def __init__(self, model, mask_mlp):
        self.mask_mlp = mask_mlp
        self.model = model

    def fit(self):
        total_par = 0
        for par in self.model.parameters():
            total_par += par.numel()
        print(f"total parameters before pruned: {total_par}")

        self.model = pruned_glm_mlp(self.model, self.mask_mlp)

        total_par_pruned = 0
        for par in self.model.parameters():
            total_par_pruned += par.numel()
        print(f"total parameters after pruned: {total_par_pruned}")
        print(f"total cut num: {total_par - total_par_pruned}")
        print(f"pruned rate: {(total_par - total_par_pruned) / total_par:.4f}")

        return self.model

class PrunedLlamaModel:
    def __init__(self, model, mask=None):
        self.mask = mask
        self.model = model

    def fit(self):
        total_par = 0
        for par in self.model.parameters():
            total_par += par.numel()
        print(f"total parameters before pruned: {total_par}")
        self.model = pruned_llama_mlp(self.model, self.mask)

        total_par_pruned = 0
        for par in self.model.parameters():
            total_par_pruned += par.numel()
        print(f"total parameters after pruned: {total_par_pruned}")
        print(f"total cut num: {total_par - total_par_pruned}")
        print(f"pruned rate: {(total_par - total_par_pruned) / total_par:.4f}")

        return self.model

class PrunedQwenModel:
    def __init__(self, model, mask=None):
        self.mask = mask
        self.model = model

    def fit(self):
        total_par = 0
        for par in self.model.parameters():
            total_par += par.numel()
        print(f"total parameters before pruned: {total_par}")
        self.model = pruned_qwen_mlp(self.model, self.mask)

        total_par_pruned = 0
        for par in self.model.parameters():
            total_par_pruned += par.numel()
        print(f"total parameters after pruned: {total_par_pruned}")
        print(f"total cut num: {total_par - total_par_pruned}")
        print(f"pruned rate: {(total_par - total_par_pruned) / total_par:.4f}")

        return self.model