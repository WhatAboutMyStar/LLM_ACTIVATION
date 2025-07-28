import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

def pruned_glm_mlp(model, mask):
    mask = torch.tensor(mask, dtype=torch.bool)

    for i in range(len(model.transformer.encoder.layers)):
        layer = model.transformer.encoder.layers[i]

        mask_1 = mask[i].type(torch.bool)
        with torch.no_grad():
            layer.mlp.dense_h_to_4h.weight.data = layer.mlp.dense_h_to_4h.weight.data[torch.where(mask_1)[0]]

            layer.mlp.dense_h_to_4h.out_features = mask_1.sum().item()
            layer.mlp.intermediate_size = mask_1.sum().item()

            layer.mlp.dense_4h_to_h.weight.data = layer.mlp.dense_4h_to_h.weight.data[:, torch.where(mask_1[:mask_1.shape[0] // 2])[0]]

            layer.mlp.dense_4h_to_h.in_features = mask_1.sum().item() // 2

            gc.collect()
            torch.cuda.empty_cache()

    return model