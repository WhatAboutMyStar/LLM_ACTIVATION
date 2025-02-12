from xml.sax.handler import feature_string_interning

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions
from sklearn.decomposition import FastICA, PCA
import numpy as np

from llmfact.utils import apply_mask_and_average, apply_mask_any

class LayerOutputExtractor:
    def __init__(self, model, include_layers=["h.0.attn.c_attn"], test=False, device='cpu'):
        self.hooks = None
        self.model = model
        self.layer_outputs = []
        self.include_layers = include_layers
        self.data_details = []
        self.test = test
        self.device = device
        if self.device == model.device:
            pass
        else:
            self.model.to(self.device)
    
    def _hook(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        if isinstance(outputs, BaseModelOutputWithPastAndCrossAttentions):
            outputs = outputs.last_hidden_state
        
        if isinstance(outputs, BaseModelOutputWithPoolingAndCrossAttentions):
            outputs = outputs.last_hidden_state

        self.data_details.append(outputs.shape)
        if outputs.shape[0] == 1:
            outputs = outputs.squeeze(0).detach().cpu()
        else:
            outputs = outputs.squeeze(1).detach().cpu()

        if self.test:
            print(outputs.size())
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
        self.data_details = []

        inputs = inputs.to(self.model.device)

        self.register_hooks()

        with torch.no_grad():
            _ = self.model(**inputs)

        self.remove_hooks()
        final_output = torch.cat(self.layer_outputs, dim=1)

        return final_output

class FBNFeatureExtractor:
    def __init__(self, model, include_layers=["h.0.attn.c_attn"], test=False, device='cpu'):
        self.extractor = LayerOutputExtractor(model=model,
                                              include_layers=include_layers,
                                              test=test,
                                              device=device)

    def fit(self, inputs, n_components=10, alpha=1.96, random_state=666):
        layer_outputs = self.extractor.extract_layer_outputs(inputs)
        ica = FastICA(n_components=n_components,
                      random_state=random_state)
        ica.fit(layer_outputs)
        mixing = np.array(ica.mixing_.T)
        mean = np.mean(mixing, axis=1, keepdims=True)
        std = np.std(mixing, axis=1, keepdims=True)

        normalized_matrix = (mixing - mean) / std
        mixing = np.array(normalized_matrix)
        mixing[mixing < alpha] = 0
        mixing[mixing > 0] = 1

        feature_masked, averaged_features = apply_mask_and_average(mask=mixing, feature_matrix=layer_outputs)

        return feature_masked, averaged_features

class GroupFBNFeatureExtractor(FBNFeatureExtractor):
    def __init__(self, model, include_layers=["h.0.attn.c_attn"], test=False, device='cpu'):
        super().__init__(model, include_layers=include_layers, test=test, device=device)
        self.mixing_ = None
        self.origin_mixing_ = None
        self.normal_mixing_ = None

    def fit(self, inputs_list, n_components=10, alpha=1.96, random_state=666):
        total_layer_outputs = []
        for inputs in inputs_list:
            layer_outputs = self.extractor.extract_layer_outputs(inputs)
            total_layer_outputs.append(layer_outputs)
        group_layer_outputs = torch.cat(total_layer_outputs, dim=0)
        ica = FastICA(n_components=n_components,
                      random_state=random_state)
        ica.fit(group_layer_outputs)
        mixing = np.array(ica.mixing_.T)
        self.origin_mixing_ = mixing

        mean = np.mean(mixing, axis=1, keepdims=True)
        std = np.std(mixing, axis=1, keepdims=True)

        normalized_matrix = (mixing - mean) / std
        mixing = np.array(normalized_matrix)
        self.normal_mixing_ = np.array(mixing)

        if alpha:
            mixing[mixing < alpha] = 0
            mixing[mixing > 0] = 1
            feature_masked = apply_mask_any(mask=mixing, feature_matrix=group_layer_outputs)
        else:
            feature_masked = None

        self.mixing_ = mixing

        return feature_masked

class FBNExtractor:
    def __init__(self, model, include_layers=["h.0.attn.c_attn"], test=False, device='cpu'):
        self.extractor = LayerOutputExtractor(model=model,
                                              include_layers=include_layers,
                                              test=test,
                                              device=device)

    def fit(self, inputs, n_components=10, alpha=1.96, random_state=666):
        layer_outputs = self.extractor.extract_layer_outputs(inputs)
        ica = FastICA(n_components=n_components,
                      random_state=random_state)
        ica.fit(layer_outputs)
        mixing = np.array(ica.mixing_.T)
        mean = np.mean(mixing, axis=1, keepdims=True)
        std = np.std(mixing, axis=1, keepdims=True)

        normalized_matrix = (mixing - mean) / std
        mixing = np.array(normalized_matrix)
        mixing[mixing < alpha] = 0
        mixing[mixing > 0] = 1

        return mixing
