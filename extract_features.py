import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from llmfact import LayerOutputExtractor
import argparse
import pandas as pd
from tqdm import tqdm
import os

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path,  padding_side='left', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
    return tokenizer, model

def extract_features(tokenizer, model, sample_num=10):
    include_layers = []
    print(model.config)
    layers_num = model.config.num_hidden_layers
    model_series = model.config.architectures[0]
    if model_series == "LlamaForCausalLM":
        for i in range(layers_num):
            include_layers.append(f'model.layers.{i}')
    else:
        pass

    wiki_dataset = load_dataset("Self-GRIT/wikitext-2-raw-v1-preprocessed", split='train')
    inputs_list = [tokenizer(inputs, return_tensors="pt") for inputs in wiki_dataset['text'][:int(sample_num)]]
    extractor = LayerOutputExtractor(model, include_layers, device=model.device)
    layer_outputs_list = [extractor.extract_layer_outputs(inputs) for inputs in inputs_list]
    torch.save(layer_outputs_list, "./data/tmp_data.pth")
    model_name = model.config._name_or_path.split("/")[1]
    hidden_size = model.config.hidden_size
    print(f"save {model_name} features, layer_num:{layers_num}, hidden_size: {hidden_size} at ./data/tmp_data.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract layer outputs from a causal language model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path or HuggingFace model name to load the model and tokenizer.")
    parser.add_argument("--sample_num", type=str, required=True)
    args = parser.parse_args()
    tokenizer, model = load_model(args.model_path)
    extract_features(tokenizer, model, args.sample_num)