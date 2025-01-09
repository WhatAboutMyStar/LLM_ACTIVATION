from transformers import GPT2Model, GPT2Config
import torch
import numpy as np


class MaskedGPT2LMModel:
    def __init__(self, model, include_layers=[]):
        self.mask_layers = include_layers
        self.model = model
        self.hooks = None

    def register_hooks(self, mask):
        self.hooks = []
        for i, layer_name in enumerate(self.mask_layers):
            module = self.model
            for part in layer_name.split("."):
                module = getattr(module, part)
            hook = module.register_forward_hook(lambda module, inputs, outputs: self.mask_hidden_states(module, inputs, outputs, mask[i:i+1, :]))
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def mask_hidden_states(self, module, inputs, outputs, mask_matrix):

        current_device = outputs.device

        mask_matrix_on_device = mask_matrix.to(current_device)

        with torch.no_grad():
            batch_size, seq_len, hidden_dim = outputs.shape
            mask_expanded = mask_matrix_on_device.expand(batch_size, seq_len, hidden_dim)
            outputs.masked_fill_(mask_expanded, 0.0)

        return outputs

    def forward(self, inputs, max_length=150):
        with torch.no_grad():
            generated_output = self.model.generate(
                **inputs,
                max_length=150,
                num_return_sequences=1,
                pad_token_id=50256
            )
        return generated_output

class MaskedGPT2ForSequenceClassification(MaskedGPT2LMModel):
    def __init__(self, model, include_layers=[]):
        super().__init__(model, include_layers)

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs


class MaskedGPT2AmplifiedForSequenceClassification(MaskedGPT2LMModel):
    def __init__(self, model, include_layers=[], value=1):
        super().__init__(model, include_layers)
        self.value = value

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def mask_hidden_states(self, module, inputs, outputs, mask_matrix):

        current_device = outputs.device

        mask_matrix_on_device = mask_matrix.to(current_device)

        with torch.no_grad():
            batch_size, seq_len, hidden_dim = outputs.shape
            mask_expanded = mask_matrix_on_device.expand(batch_size, seq_len, hidden_dim)
            outputs = torch.where(mask_expanded, outputs + self.value, outputs)

        return outputs







