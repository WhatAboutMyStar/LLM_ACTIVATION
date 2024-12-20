import torch

from llmfact.utils import thresholding
from llmfact.extractor import LayerOutputExtractor

import numpy as np
from sklearn.decomposition import FastICA
from sklearn.decomposition import DictionaryLearning

class StatICA:
    def __init__(self,
                 data,
                 n_components=64,
                 random_state=666):
        """
        :param data: (Time, Neuron signals)
        """
        self.components_ = None
        self.mixing_ = None
        self.normalized_matrix = None

        self.data = np.array(data)

        self.ica = FastICA(n_components=n_components,
                           random_state=random_state)

    def fit(self):
        self.ica.fit(self.data)
        self.components_ = np.array(self.ica.transform(self.data))
        self.mixing_ = np.array(self.ica.mixing_.T)

        mean = np.mean(self.mixing_, axis=1, keepdims=True)
        std = np.std(self.mixing_, axis=1, keepdims=True)

        self.normalized_matrix = (self.mixing_ - mean) / std

        mixing_ = self.get_activate_matrix()

        act_num = (mixing_ > 0).sum(axis=1)
        act_rate = act_num / self.data.shape[1]

        print("average activation neuron number: {}".format(act_num.mean()))
        print("average activation neuron rate: {:.4f}".format(act_rate.mean()))

        return act_num, act_rate

    def reset(self,
              data,
              n_components=64,
              random_state=666):
        self.components_ = None
        self.data = data

        self.ica = FastICA(n_components=n_components,
                           random_state=random_state)

    def get_activate_matrix(self, alpha=1.96):
        mixing = np.array(self.normalized_matrix)
        mixing[mixing < alpha] = 0
        return mixing

    def get_mixing_matrix(self, return_type="pt"):
        if return_type == "pt":
            return torch.tensor(self.mixing_)
        elif return_type == "np":
            return np.array(self.mixing_)


class StatDictionaryLearning:
    def __init__(self,
                 data,
                 alpha=1,
                 n_components=64,
                 random_state=666,
                 n_jobs=-1):

        self.dictionary_ = None
        self.mixing_ = None
        self.normalized_matrix = None
        self.data = np.array(data)

        self.dl = DictionaryLearning(alpha=alpha,
                                     n_components=n_components,
                                     random_state=random_state,
                                     n_jobs=n_jobs)

    def fit(self):
        self.dl.fit(self.data)

        self.dictionary_ = self.dl.transform(self.data)
        self.mixing_ = np.array(self.dl.components_)

        mean = np.mean(self.mixing_, axis=1, keepdims=True)
        std = np.std(self.mixing_, axis=1, keepdims=True)

        self.normalized_matrix = (self.mixing_ - mean) / std

        mixing_ = self.get_activate_matrix()

        act_num = (mixing_ > 0).sum(axis=1)
        act_rate = act_num / self.data.shape[1]

        print("average activation neuron number: {}".format(act_num.mean()))
        print("average activation neuron rate: {:.4f}".format(act_rate.mean()))

        return act_num, act_rate

    def get_activate_matrix(self, alpha=1.96):
        mixing = np.array(self.normalized_matrix)
        mixing[mixing < alpha] = 0
        return mixing

class LLMFC:
    def __init__(self, model, mask, include_layers=[], device="cuda"):
        self.extractor = LayerOutputExtractor(model=model, include_layers=include_layers, device=device)
        self.mask = torch.tensor(mask)
        self.time_series = None

    def fit(self, inputs_list):
        total_layer_outputs = []
        for inputs in inputs_list:
            layer_outputs = self.extractor.extract_layer_outputs(inputs)
            total_layer_outputs.append(layer_outputs)
        group_layer_outputs = torch.cat(total_layer_outputs, dim=0)
        time_step, neuron_num = group_layer_outputs.size()
        mask_matrix_expanded = self.mask.unsqueeze(0).expand(time_step, -1, -1)
        masked_features = group_layer_outputs.unsqueeze(1) * mask_matrix_expanded

        time_series_activations = []
        for i in range(self.mask.shape[0]):
            mask = self.mask[i]
            masked_feature = masked_features[:, i, :]

            non_zero_counts = mask.sum()  # 标量

            if non_zero_counts == 0:
                time_series_activation = torch.zeros(time_step)
            else:
                time_series_activation = masked_feature.sum(dim=1) / non_zero_counts

            time_series_activations.append(time_series_activation)

        time_series_activations = torch.stack(time_series_activations)
        time_series_activations_np = time_series_activations.numpy()
        self.time_series = time_series_activations_np
        functional_connectivity_matrix = np.corrcoef(time_series_activations_np)
        functional_connectivity_matrix = torch.tensor(functional_connectivity_matrix)
        return functional_connectivity_matrix