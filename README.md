# LLM_ACTIVATION
[Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models](https://arxiv.org/abs/2502.20408)

[Pruning Large Language Models by Identifying and Preserving Functional Networks](https://arxiv.org/abs/2508.05239)

## Introduction
### Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models
In recent years, the rapid advancement of large language models (LLMs) in natural language processing has sparked significant interest among researchers to understand their mechanisms and functional characteristics. Although prior studies have attempted to explain LLM functionalities by identifying and interpreting specific neurons, these efforts mostly focus on individual neuron contributions, neglecting the fact that human brain functions are realized through intricate interaction networks. Inspired by research on functional brain networks (FBNs) in the field of neuroscience, we utilize similar methodologies estabilished in FBN analysis to explore the "functional networks" within LLMs in this study. Experimental results highlight that, much like the human brain, LLMs exhibit certain functional networks that recur frequently during their operation. Further investigation reveals that these functional networks are indispensable for LLM performance. Inhibiting key functional networks severely impairs the model's capabilities. Conversely, amplifying the activity of neurons within these networks can enhance either the model's overall performance or its performance on specific tasks. This suggests that these functional networks are strongly associated with either specific tasks or the overall performance of the LLM. Our study provides novel insights into the interpretation of LLMs.

### Pruning Large Language Models by Identifying and Preserving Functional Networks
Structured pruning is one of the representative techniques for compressing large language models (LLMs) to reduce GPU memory consumption and accelerate inference speed. It offers significant practical value in improving the efficiency of LLMs in real-world applications. Current structured pruning methods typically rely on assessment of the importance of the structure units and pruning the units with less importance. Most of them overlooks the interaction and collaboration among artificial neurons that are crucial for the functionalities of LLMs, leading to a disruption in the macro functional architecture of LLMs and consequently a pruning performance degradation. Inspired by the inherent similarities between artificial neural networks and functional neural networks in the human brain, we alleviate this challenge and propose to prune LLMs by identifying and preserving functional networks within LLMs in this study. To achieve this, we treat an LLM as a digital brain and decompose the LLM into functional networks, analogous to  identifying functional brain networks in neuroimaging data. Afterwards, an LLM is pruned by preserving the key neurons within these functional networks. Experimental results demonstrate that the proposed method can successfully identify and locate functional networks and key neurons in LLMs, enabling efficient model pruning. 

### FNF: Functional Network Fingerprint for Large Language Models
The development of large language models (LLMs) is costly and has significant commercial value. Consequently, preventing unauthorized appropriation of open-source LLMs and protecting developers’ intellectual property rights have become critical challenges. In this work, we propose the Functional Network Fingerprint (FNF), a training-free, sample-efficient method for detecting whether a suspect LLM is derived from a victim model, based on the consistency between their functional network activity. We demonstrate that models that share a common origin, even with differences in scale or architecture, exhibit highly consistent patterns of neuronal activity within their functional networks across diverse input samples. In contrast, models trained independently on distinct data or with different objectives fail to preserve such activity alignment. Unlike conventional approaches, our method requires only a few samples for verification, preserves model utility, and remains robust to common model modifications (such as fine-tuning, pruning, and parameter permutation), as well as to comparisons across diverse architectures and dimensionalities. FNF thus provides model owners and third parties with a simple, non-invasive, and effective tool for protecting LLM intellectual property.

## Pruning LLMs
- [This is an example of pruning Vicuna-7B-v1.5 (This is an early version of CanICA that runs on CPU, lacking parallel processing capabilities and thus being relatively slow.)](CanICA-Vicuna-7B.ipynb)
- [This is an example of pruning Vicuna-7B-v1.5 (In this example, GPU acceleration is used, and we have implemented CanICA in a form that supports parallel processing on the GPU, resulting in faster speeds.)](CanICA-GPU-Vicuna-7B-test.ipynb)
- [This is an example referenced from FLAP, which provides bias compensation for the pruned model.](CanICA-Vicuna-7B+bias.ipynb)

## LLM Fingerprint
- [This is an example of obtaining dataset for LLM Fingerprint](Model-features-extract.ipynb)
- [This is an example of functional networks fingerprint for LLMs](LLM-Fingerprint.ipynb)

## Citaion
```
@article{liu2025brain,
  title={Brain-inspired exploration of functional networks and key neurons in large language models},
  author={Liu, Yiheng and Gao, Xiaohui and Sun, Haiyang and Ge, Bao and Liu, Tianming and Han, Junwei and Hu, Xintao},
  journal={arXiv preprint arXiv:2502.20408},
  year={2025}
}

@article{liu2025pruning,
  title={Pruning Large Language Models by Identifying and Preserving Functional Networks},
  author={Liu, Yiheng and Ning, Junhao and Xia, Sichen and Gao, Xiaohui and Qiang, Ning and Ge, Bao and Han, Junwei and Hu, Xintao},
  journal={arXiv preprint arXiv:2508.05239},
  year={2025}
}
```
