import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions

class LayerOutputExtractor:
    def __init__(self, model, include_layers=['ln_1', '']):
        self.model = model
        self.layer_outputs = []
        self.include_layers = include_layers
    
    def _hook(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        if isinstance(outputs, BaseModelOutputWithPastAndCrossAttentions):
            outputs = outputs.last_hidden_state
        
        if isinstance(outputs, BaseModelOutputWithPoolingAndCrossAttentions):
            outputs = outputs.last_hidden_state
        outputs = outputs.squeeze(0)
        if outputs.size(-1) in [768, 3072]:
            self.layer_outputs.append(outputs)
    
    def register_hooks(self):
        self.hooks = []
        for layer_name, module in self.model.named_modules():
            if layer_name in self.include_layers:
                hook = module.register_forward_hook(self._hook)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract_layer_outputs(self, inputs):
        self.layer_outputs = []
        
        self.register_hooks()

        with torch.no_grad():
            _ = self.model(**inputs)

        self.remove_hooks()
        # for layer in self.layer_outputs:
        #     print(layer.shape)
        final_output = torch.cat(self.layer_outputs, dim=1)

        return final_output