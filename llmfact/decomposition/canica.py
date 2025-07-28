import torch
import numpy as np
import queue
from concurrent.futures  import ThreadPoolExecutor
from tqdm import tqdm
from operator import itemgetter
import warnings
warnings.filterwarnings("ignore")

import gc
import os
import traceback
from tqdm import trange
from datetime import datetime

from llmfact.decomposition.fastica import fastica
from llmfact.utils import setup_seed, z_score_signals
from llmfact.extractor import LayerOutputExtractor


class CanICA:
    def __init__(self,
                 n_components=128,
                 random_state=666,
                 device="cuda"):
        self.n_components = n_components
        self.random_state = random_state
        self.normal_mixing_ = None
        self.components_ = None
        self.device = device

    def fit(self, signals, max_iter=300):
        if isinstance(signals, str):
            if signals.endswith(".npy"):
                signals = np.load(signals)
            elif signals.endswith(".pth"):
                signals = torch.load(signals)
            else:
                raise ValueError("signals must be numpy.ndarray or pth tensor"
                                 "You provided {}".format(signals))

        if self.random_state:
            setup_seed(self.random_state)

        signals = torch.tensor(signals, dtype=torch.float32)

        signals = signals.to(self.device)

        U, _, _ = torch.linalg.svd(signals.T, full_matrices=False)
        components_ = U[:, :self.n_components]

        self.components_ = components_.T

        results = fastica(components_.to(torch.float64),
                          n_components=self.n_components,
                          fun="cube",
                          max_iter=max_iter,
                          tol=1e-04,
                          random_state=None)

        ica_maps = results[2].T

        mean = torch.mean(ica_maps, dim=1, keepdim=True)
        std = torch.std(ica_maps, dim=1, keepdim=True)

        z_score_ica_maps = (ica_maps - mean) / std

        self.normal_mixing_ = z_score_ica_maps

        return self

class SingleLayerAnalysisGPU(LayerOutputExtractor):
    def __init__(self, model, include_layers=["h.0.attn.c_attn"],  test=False, device='cpu'):
        super().__init__(model, include_layers=include_layers, test=test, device=device)
        self.include_layers  = include_layers
        self.mixing_  = None
        self.origin_mixing_  = None
        self.normal_mixing_  = None

    def fit(self, inputs, n_components=10, random_state=666,
            preprocessing=False, total_layer_num=None,
            max_iter=200):

        # 确定总层数
        if total_layer_num:
            total_layer_num = total_layer_num
        else:
            total_layer_num = len(self.include_layers)

        # 提取层输出
        if type(inputs) == list:
            layer_outputs = torch.cat([self.extract_layer_outputs(inp) for inp in inputs], dim=0)
        else:
            layer_outputs = self.extract_layer_outputs(inputs)

        if preprocessing:
            layer_outputs = z_score_signals(layer_outputs).to(torch.float64)

        token_num = layer_outputs.shape[0]
        layer_outputs = layer_outputs.reshape(token_num, total_layer_num, -1)

        # 获取GPU设备
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPU available")
        max_workers = min(num_gpus, 10)
        device_ids = list(range(max_workers))
        device_queue = queue.Queue()
        for device_id in device_ids:
            device_queue.put(device_id)

        # 定义处理函数

        def process_layer(i):
            device = device_queue.get()
            try:
                with torch.cuda.device(device):
                    data = layer_outputs[:, i, :].to(device)
                    ica = CanICA(n_components=n_components,
                                 random_state=random_state,
                                 device=device)
                    ica.fit(data, max_iter=max_iter)
                    return ica.normal_mixing_
            finally:
                del data
                del ica
                torch.cuda.empty_cache()
                device_queue.put(device)

        # 并行执行

        normal_mixing_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_layer, i) for i in range(total_layer_num)]
            for future in tqdm(futures, total=total_layer_num, desc="Processing Layers"):
                normal_mixing_list.append(future.result().detach().cpu())

        self.normal_mixing_ = torch.cat(normal_mixing_list, dim=1)
        gc.collect()
        torch.cuda.empty_cache()
        return self


def save_fbn(data, save_dir, save_name):
    save_path = os.path.join(save_dir, save_name + ".pth")
    print(f"save at {save_path}")
    torch.save(data, save_path)


def SingleICA(model, include_layers, tokenizer, dataset, n_components=128, preprocessing=True, max_iter=300, total_layer_num=28):
    extractor = SingleLayerAnalysisGPU(model, include_layers=include_layers, device=model.device)
    inputs_list = [tokenizer(inputs, return_tensors="pt", max_length=1024, truncation=True) for inputs in dataset]
    extractor.fit(inputs=inputs_list, n_components=n_components,
                  random_state=327, preprocessing=preprocessing,
                  total_layer_num=total_layer_num, max_iter=max_iter)
    return extractor


def get_fbn_for_pruner(model, include_layers, tokenizer, dataset, model_name, save_dir="./data/FBN/",
                       n_components=256, preprocessing=True, max_iter=300,
                       total_layer_num=32, sample_num=40, ica_num=80):
    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    save_dir = os.path.join(save_dir, current_time)

    print(f"save fbn at {save_dir}")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    normal_list = []
    for i in trange(ica_num):
        try:
            extractor = SingleICA(model=model, include_layers=include_layers, tokenizer=tokenizer,
                                  dataset=dataset[sample_num * i:sample_num * (i + 1)],
                                  n_components=n_components, preprocessing=preprocessing,
                                  max_iter=max_iter, total_layer_num=total_layer_num)
            normal = extractor.normal_mixing_
            normal_list.append(normal)
            save_fbn(normal, save_dir,
                     f"text{sample_num}-canica-SingleICAGPU-{model_name}-{n_components}-{i}")
        except Exception as e:
            print(f"第{i}次算ICA出现问题，出现问题的样本在[{sample_num * i}:{sample_num * (i + 1)}]")
            print(f"捕获到异常: {e.__class__.__name__}, 错误信息: {e}")

            traceback_info = traceback.format_exc()
            print("详细错误堆栈信息:")
            print(traceback_info)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    normal_components = torch.cat(normal_list, dim=0)
    return normal_components


def load_fbn_for_pruner(load_path, threshold=3.6):
    if os.path.isdir(load_path):
        fbn_list = os.listdir(load_path)
        any_mask_list = []
        for fbn in tqdm(fbn_list):
            normal_components = torch.load(os.path.join(load_path, fbn))
            any_mask = torch.abs(normal_components) > threshold
            any_mask = torch.any(any_mask, dim=0, keepdim=True)
            any_mask_list.append(any_mask)
        any_mask = torch.cat(any_mask_list, dim=0)
        any_mask = torch.any(any_mask, dim=0, keepdim=True)
    else:
        normal_components = torch.load(os.path.join(load_path))
        any_mask = torch.abs(normal_components) > threshold
        any_mask = torch.any(any_mask, dim=0, keepdim=True)

    return any_mask