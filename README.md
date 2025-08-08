# LLM_ACTIVATION
[Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models](https://arxiv.org/abs/2502.20408)

[Pruning Large Language Models by Identifying and Preserving Functional Networks](https://arxiv.org/abs/2508.05239)

## Introduction
We find functional networks (just similar to the functional networks in human brain) exist in Large Language Models (LLMs). The neurons in functional networks that are critical to the LLM's performance. Masking key functional networks significantly impairs the model's performance, while retaining just a subset of these networks is adequate to maintain effective operation. 

Structured pruning is one of the representative techniques for compressing large language models (LLMs) to reduce GPU memory consumption and accelerate inference speed. It offers significant practical value in improving the efficiency of LLMs in real-world applications. Current structured pruning methods typically rely on assessment of the importance of the structure units and pruning the units with less importance. Most of them overlooks the interaction and collaboration among artificial neurons that are crucial for the functionalities of LLMs, leading to a disruption in the macro functional architecture of LLMs and consequently a pruning performance degradation. Inspired by the inherent similarities between artificial neural networks and functional neural networks in the human brain, we alleviate this challenge and propose to prune LLMs by identifying and preserving functional networks within LLMs in this study. To achieve this, we treat an LLM as a digital brain and decompose the LLM into functional networks, analogous to  identifying functional brain networks in neuroimaging data. Afterwards, an LLM is pruned by preserving the key neurons within these functional networks. Experimental results demonstrate that the proposed method can successfully identify and locate functional networks and key neurons in LLMs, enabling efficient model pruning. 

## Citaion
```
@article{liu2025brain,
  title={Brain-inspired exploration of functional networks and key neurons in large language models},
  author={Liu, Yiheng and Gao, Xiaohui and Sun, Haiyang and Ge, Bao and Liu, Tianming and Han, Junwei and Hu, Xintao},
  journal={arXiv preprint arXiv:2502.20408},
  year={2025}
}
```
