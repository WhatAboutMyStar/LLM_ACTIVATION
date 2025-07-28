import torch
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions
from sklearn.decomposition import FastICA
import numpy as np

from llmfact.utils import apply_mask_and_average, apply_mask_any, bf16_to_numpy, z_score_signals
from llmfact.decomposition import CanICA
from tqdm import trange

from multiprocessing import Pool
from tqdm import tqdm
from datetime import datetime
import os

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

        final_output = bf16_to_numpy(final_output)

        final_output = torch.tensor(final_output, dtype=torch.float16)

        return final_output

class FBNFeatureExtractor:
    def __init__(self, model, include_layers=["h.0.attn.c_attn"], test=False, device='cpu'):
        self.extractor = LayerOutputExtractor(model=model,
                                              include_layers=include_layers,
                                              test=test,
                                              device=device)

    def fit(self, inputs, n_components=10, alpha=1.96, random_state=666, preprocessing=False, method="fastica"):
        layer_outputs = self.extractor.extract_layer_outputs(inputs)
        if preprocessing:
            layer_outputs = z_score_signals(layer_outputs)
            layer_outputs = np.array(layer_outputs.numpy(), dtype=np.float64)

        if method == "fastica":
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
        else:
            ica = CanICA(n_components=n_components, random_state=random_state)
            ica.fit(layer_outputs)
            mixing = np.array(ica.mixing_)
            mixing[mixing > 0] = 1


        feature_masked, averaged_features = apply_mask_and_average(mask=mixing, feature_matrix=layer_outputs)

        return feature_masked, averaged_features

class GroupFBNFeatureExtractor(FBNFeatureExtractor):
    def __init__(self, model, include_layers=["h.0.attn.c_attn"], test=False, device='cpu'):
        super().__init__(model, include_layers=include_layers, test=test, device=device)
        self.mixing_ = None
        self.origin_mixing_ = None
        self.normal_mixing_ = None

    def fit(self, inputs_list, n_components=10, alpha=1.96, random_state=666, preprocessing=False, method="fastica"):
        total_layer_outputs = []
        for inputs in inputs_list:
            layer_outputs = self.extractor.extract_layer_outputs(inputs)
            total_layer_outputs.append(layer_outputs)
        group_layer_outputs = torch.cat(total_layer_outputs, dim=0)
        if preprocessing:
            group_layer_outputs = z_score_signals(group_layer_outputs)
            group_layer_outputs = np.array(group_layer_outputs.numpy(), dtype=np.float64)

        if method == "fastica":
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
        else:
            ica = CanICA(n_components=n_components, random_state=random_state)
            ica.fit(group_layer_outputs)
            self.mixing_ = np.array(ica.mixing_)
            self.mixing_[self.mixing_ > 0] = 1
            self.normal_mixing_ = np.array(ica.normal_mixing_)
            self.origin_mixing_ = np.array(ica.original_mixing_)
            feature_masked = None

        return feature_masked

class FBNExtractor:
    def __init__(self, model, include_layers=["h.0.attn.c_attn"], test=False, device='cpu'):
        self.extractor = LayerOutputExtractor(model=model,
                                              include_layers=include_layers,
                                              test=test,
                                              device=device)

    def fit(self, inputs, n_components=10, alpha=1.96, random_state=666, method="fastica"):
        layer_outputs = self.extractor.extract_layer_outputs(inputs)

        if method == "fastica":
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
        else:
            ica = CanICA(n_components=n_components, random_state=random_state)
            ica.fit(layer_outputs)
            mixing = np.array(ica.mixing_)
            mixing[mixing > 0] = 1

        return mixing


def process_layer(layer_idx, layer_outputs, n_components, alpha, random_state):
    ica = FastICA(n_components=n_components, random_state=random_state)
    ica.fit(layer_outputs[:, layer_idx, :])
    mixing = torch.tensor(ica.mixing_.T)

    origin_mixing = mixing.clone()

    mean = mixing.mean(dim=0, keepdim=True)
    std = mixing.std(dim=0, keepdim=True)
    normalized_matrix = (mixing - mean) / std
    normal_mixing = torch.tensor(normalized_matrix)

    mixing[mixing < alpha] = 0
    mixing[mixing > 0] = 1

    return mixing, origin_mixing, normal_mixing


class SingleLayerAnalysis(LayerOutputExtractor):
    def __init__(self, model, include_layers=["h.0.attn.c_attn"], test=False, device='cpu'):
        super().__init__(model, include_layers=include_layers, test=test, device=device)
        self.include_layers = include_layers
        self.mixing_ = None
        self.origin_mixing_ = None
        self.normal_mixing_ = None

    def fit(self, inputs, n_components=10, alpha=1.96, random_state=666,
            preprocessing=False, total_layer_num=None, method="fastica",
            max_iter=200, n_iter=5, norm=True):
        if total_layer_num:
            total_layer_num = total_layer_num
        else:
            total_layer_num = len(self.include_layers)

        if type(inputs) == list:
            total_layer_outputs = []
            for inp in inputs:
                layer_outputs = self.extract_layer_outputs(inp)
                total_layer_outputs.append(layer_outputs)
            layer_outputs = torch.cat(total_layer_outputs, dim=0)
        else:
            layer_outputs = self.extract_layer_outputs(inputs)

        if preprocessing:
            layer_outputs = z_score_signals(layer_outputs)
            layer_outputs = np.array(layer_outputs.numpy(), dtype=np.float64)

        token_num = layer_outputs.shape[0]

        layer_outputs = layer_outputs.reshape(token_num, total_layer_num, -1)

        # mixing_list = []
        # origin_mixing_list = []
        normal_mixing_list = []
        for i in trange(total_layer_num):
            if method == "fastica":
                ica = FastICA(n_components=n_components,
                              random_state=random_state,
                              max_iter=max_iter)
                ica.fit(layer_outputs[:, i, :])
                mixing = torch.tensor(ica.mixing_.T)
                # origin_mixing_list.append(mixing)

                mean = torch.mean(mixing, dim=1, keepdim=True)
                std = torch.std(mixing, dim=1, keepdim=True)

                normalized_matrix = (mixing - mean) / std

                normal_mixing_list.append(normalized_matrix)

            else:
                ica = CanICA(n_components=n_components, random_state=random_state)
                ica.fit(layer_outputs[:, i, :], max_iter=max_iter, n_iter=n_iter, norm=norm)

                normal_mixing_list.append(torch.tensor(ica.normal_mixing_))


        # self.mixing_ = torch.cat(mixing_list, dim=1)
        # self.origin_mixing_ = torch.cat(origin_mixing_list, dim=1)
        self.normal_mixing_ = torch.cat(normal_mixing_list, dim=1)

class MutiLayerAnalysis(SingleLayerAnalysis):
    def __init__(self, model, include_layers=["h.0.attn.c_attn"], test=False, device='cpu'):
        super().__init__(model, include_layers, test, device)

    def get_layer_outputs(self, inputs):
        if type(inputs) == list:
            total_layer_outputs = []
            for inp in inputs:
                layer_outputs = self.extract_layer_outputs(inp)
                total_layer_outputs.append(layer_outputs)
            layer_outputs = torch.cat(total_layer_outputs, dim=0)
        else:
            layer_outputs = self.extract_layer_outputs(inputs)
        return layer_outputs

    def fit(self, inputs, n_components=10, window_size=3, random_state=666,
            preprocessing=False, total_layer_num=None,
            method="fastica", max_iter=200, n_iter=5, norm=True):
        if total_layer_num:
            total_layer_num = total_layer_num
        else:
            total_layer_num = len(self.include_layers)

        layer_outputs = self.get_layer_outputs(inputs)

        if preprocessing:
            layer_outputs = z_score_signals(layer_outputs)
            layer_outputs = np.array(layer_outputs.numpy(), dtype=np.float64)

        token_num = layer_outputs.shape[0]
        layer_outputs = layer_outputs.reshape(token_num, total_layer_num, -1) # (512, 28, 4096)
        neuron_num = layer_outputs.shape[-1]

        normal_mixing_list = []
        # mixing_list = []
        for start_idx in trange(0, total_layer_num - window_size + 1):
            end_idx = start_idx + window_size

            window_inputs = layer_outputs[:, start_idx:end_idx, :]
            window_inputs = window_inputs.reshape(token_num, -1)

            if method == "fastica":
                ica = FastICA(n_components=n_components,
                              random_state=random_state,
                              max_iter=max_iter)
                ica.fit(window_inputs)
                mixing = torch.tensor(ica.mixing_.T)

                mean = torch.mean(mixing, dim=1, keepdim=True)
                std = torch.std(mixing, dim=1, keepdim=True)

                normalized_matrix = (mixing - mean) / std
                normal_mixing = torch.tensor(normalized_matrix) #(n_components, window_size * neuron_num)

                normal_mixing_list.append(torch.tensor(normal_mixing))

            else:
                ica = CanICA(n_components=n_components, random_state=random_state)
                ica.fit(window_inputs, max_iter=max_iter, n_iter=n_iter, norm=norm)
                # mixing_list.append(np.array(ica.mixing_))
                normal_mixing_list.append(torch.tensor(ica.normal_mixing_))

        self.normal_mixing_ = torch.cat(normal_mixing_list,  dim=1) #(512, 28546)

        mixing_by_layer = {i: [] for i in range(total_layer_num)}

        for win_idx, normal_mixing in enumerate(normal_mixing_list):
            start_idx = win_idx
            for rel_idx in range(window_size):
                abs_layer_idx = start_idx + rel_idx
                if abs_layer_idx < total_layer_num:
                    layer_start = rel_idx * neuron_num
                    layer_end = (rel_idx + 1) * neuron_num
                    layer_mixing = normal_mixing[:, layer_start:layer_end]
                    mixing_by_layer[abs_layer_idx].append(layer_mixing)

        aggregated_mixing_parts = []
        for layer_idx in range(total_layer_num):
            if layer_idx == 0 or layer_idx == total_layer_num - 1:
                if mixing_by_layer[layer_idx]:
                    averaged_mixing = mixing_by_layer[layer_idx][0]  # 只有一个窗口包含该层
                else:
                    averaged_mixing = torch.zeros((n_components, neuron_num))  # 如果没有分析结果，填充零向量
            else:
                # 中间层需要聚合多个窗口的结果
                if mixing_by_layer[layer_idx]:
                    averaged_mixing = torch.mean(torch.stack(mixing_by_layer[layer_idx], dim=0), dim=0)
                else:
                    averaged_mixing = torch.zeros((n_components, neuron_num))  # 如果没有分析结果，填充零向量

            aggregated_mixing_parts.append(averaged_mixing)
            self.normal_mixing_ = torch.cat(aggregated_mixing_parts, dim=1)

class MutiLayerAnalysis2(MutiLayerAnalysis):
    def __init__(self, model, include_layers=["h.0.attn.c_attn"], test=False, device='cpu'):
        super().__init__(model, include_layers, test, device)
        self.normal_mixing_dict = None

    def fit(self, inputs, n_components=10, window_size=3, random_state=666,
            preprocessing=False, total_layer_num=None,
            method="fastica", max_iter=200, n_iter=5, norm=True):

        if total_layer_num:
            total_layer_num = total_layer_num
        else:
            total_layer_num = len(self.include_layers)

        layer_outputs = self.get_layer_outputs(inputs)

        if preprocessing:
            layer_outputs = z_score_signals(layer_outputs)
            layer_outputs = np.array(layer_outputs.numpy(), dtype=np.float64)

        token_num = layer_outputs.shape[0]
        layer_outputs = layer_outputs.reshape(token_num, total_layer_num, -1) # (512, 28, 4096)
        neuron_num = layer_outputs.shape[-1]

        layer_normal_mixing_dict = {i:[] for i in range(total_layer_num)}

        for start_idx in trange(0, total_layer_num - window_size + 1):
            end_idx = start_idx + window_size

            window_inputs = layer_outputs[:, start_idx:end_idx, :]
            window_inputs = window_inputs.reshape(token_num, -1)

            if method == "fastica":
                ica = FastICA(n_components=n_components,
                              random_state=random_state,
                              max_iter=max_iter)
                ica.fit(window_inputs)
                mixing = torch.tensor(ica.mixing_.T)

                mean = torch.mean(mixing, dim=1, keepdim=True)
                std = torch.std(mixing, dim=1, keepdim=True)

                normalized_matrix = (mixing - mean) / std
                normal_mixing = torch.tensor(normalized_matrix).reshape(n_components, window_size, neuron_num) #(n_components, window_size * neuron_num)

            else:
                ica = CanICA(n_components=n_components, random_state=random_state)
                ica.fit(window_inputs, max_iter=max_iter, n_iter=n_iter, norm=norm)

                normal_mixing = torch.tensor(ica.normal_mixing_).reshape(n_components, window_size, neuron_num)

            for i in range(window_size):
                layer_normal_mixing_dict[start_idx + i].append(normal_mixing[:, i, :])

        for i in range(total_layer_num):
            layer_normal_mixing_dict[i] = torch.cat(layer_normal_mixing_dict.get(i), dim=0)
        
        self.normal_mixing_dict = layer_normal_mixing_dict


def MutiICA(model, include_layers, tokenizer, dataset, n_components=256, window_size=3, preprocessing=True, max_iter=500, n_iter=5,
            norm=False, total_layer_num=32, method="canica"):
    extractor = MutiLayerAnalysis2(model, include_layers=include_layers, device=model.device)
    inputs_list = [tokenizer(inputs, return_tensors="pt", max_length=1024, truncation=True) for inputs in dataset]
    extractor.fit(inputs=inputs_list, n_components=n_components,
                  window_size=window_size, random_state=666,
                  preprocessing=preprocessing, total_layer_num=total_layer_num, method=method, max_iter=max_iter, norm=norm,
                  n_iter=n_iter)
    return extractor


def save_fbn(data, save_dir, save_name):
    if isinstance(data, list) or isinstance(data, dict):
        save_path = os.path.join(save_dir, save_name + ".pth")
        print(f"save at {save_path}")
        torch.save(data, save_path)
    else:
        save_path = os.path.join(save_dir, save_name + ".npy")
        print(f"save at {save_path}")
        np.save(save_path, data)


def SingleICA(model, include_layers, tokenizer, dataset, n_components=256, preprocessing=True, max_iter=500, norm=False, n_iter=5, total_layer_num=32, method="canica"):
    extractor = SingleLayerAnalysis(model, include_layers=include_layers, device=model.device)
    inputs_list = [tokenizer(inputs, return_tensors="pt", max_length=1024, truncation=True) for inputs in dataset]
    extractor.fit(inputs=inputs_list, n_components=n_components,
                  random_state=327, preprocessing=preprocessing,
                  total_layer_num=total_layer_num, method=method, max_iter=max_iter, norm=norm, n_iter=n_iter)
    return extractor

def get_fbn_for_pruner(model, include_layers, tokenizer, dataset, model_name, save_dir="./data/FBN/",
                       n_components=256, preprocessing=True, max_iter=500,
                       norm=False, n_iter=5, total_layer_num=32, sample_num=40, ica_num=60, window_size=None, method="canica"):

    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    save_dir = os.path.join(save_dir, current_time)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if window_size is not None:
        normal_list = []
        for i in trange(ica_num):
            try:
                extractor = MutiICA(model=model, include_layers=include_layers, tokenizer=tokenizer,
                                    dataset=dataset[sample_num * i:sample_num * (i + 1)],
                                    n_components=n_components, preprocessing=preprocessing, window_size=window_size,
                                    max_iter=max_iter, norm=norm, n_iter=n_iter, total_layer_num=total_layer_num, method=method)
                normal = extractor.normal_mixing_dict
                normal_list.append(normal)
                # save_fbn(normal, save_dir, f"text{sample_num}-{method}-MutiICA-window-size-{window_size}-{model_name}-{n_components}-{i}")
            except:
                print(f"第{i}次算ICA出现问题，出现问题的样本在[{sample_num * i}:{sample_num * (i + 1)}]")
        save_fbn(normal_list, save_dir,
                 f"text{sample_num}-{method}-MutiICA-window-size-{window_size}-{model_name}-{n_components}")
    else:
        normal_list = []
        for i in trange(ica_num):
            try:
                extractor = SingleICA(model=model, include_layers=include_layers, tokenizer=tokenizer,
                                      dataset=dataset[sample_num * i:sample_num * (i + 1)],
                                      n_components=n_components, preprocessing=True,
                                      max_iter=max_iter, norm=norm, n_iter=n_iter,
                                      total_layer_num=total_layer_num, method=method)
                normal = extractor.normal_mixing_
                normal_list.append(normal)
                save_fbn(normal, save_dir,
                         f"text{sample_num}-{method}-SingleICA-{model_name}-{n_components}-{i}")
            except:
                print(f"第{i}次算ICA出现问题，出现问题的样本在[{sample_num * i}:{sample_num * (i + 1)}]")

def load_fbn_for_pruner(load_path, threshold=3.6):
    if os.path.isdir(load_path):
        fbn_list = os.listdir(load_path)
        any_mask_list = []
        for fbn in fbn_list:
            normal_components = np.load(os.path.join(load_path, fbn))
            any_mask = np.abs(normal_components) > threshold
            any_mask = np.any(any_mask, axis=0, keepdims=True)
            any_mask_list.append(any_mask)
        any_mask = np.concatenate(any_mask_list, axis=0)
        any_mask = np.any(any_mask, axis=0, keepdims=True)
    else:
        normal_components = np.load(os.path.join(load_path))
        any_mask = np.abs(normal_components) > threshold
        any_mask = np.any(any_mask, axis=0, keepdims=True)

    return any_mask


