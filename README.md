# NVIDIA NeMo-Aligner

# Latest News
- We released Nemotron-4-340B [Base](https://huggingface.co/nvidia/Nemotron-4-340B-Base), [Instruct](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct), [Reward](https://huggingface.co/nvidia/Nemotron-4-340B-Reward). The Instruct and Reward variants are trained in Nemo-Aligner. Please see the [Helpsteer2](https://arxiv.org/abs/2406.08673) paper for more details on the reward model training.
- We are excited to announce the release of accelerated generation support in our RLHF pipeline using [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). For more information, please refer to our [RLHF documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/rlhf.html).
- [NeMo-Aligner Paper](https://arxiv.org/abs/2405.01481) is now out on arxiv!

## Introduction

NeMo-Aligner is a scalable toolkit for efficient model alignment. The toolkit has support for state-of-the- art model alignment algorithms such as SteerLM, DPO, and Reinforcement Learning from Human Feedback (RLHF). These algorithms enable users to align language models to be more safe, harmless, and helpful. Users can perform end-to-end model alignment on a wide range of model sizes and take advantage of all the parallelism techniques to ensure their model alignment is done in a performant and resource-efficient manner. For more technical details, please refer to our [paper](https://arxiv.org/abs/2405.01481).

The NeMo-Aligner toolkit is built using the NeMo Framework, which enables scalable training across thousands of GPUs using tensor, data, and pipeline parallelism for all alignment components. Additionally, our checkpoints are cross-compatible with the NeMo ecosystem, facilitating inference deployment and further customization (https://github.com/NVIDIA/NeMo-Aligner).

The toolkit is currently in it's early stages. We are committed to improving the toolkit to make it easier for developers to pick and choose different alignment algorithms to build safe, helpful, and reliable models.

## Key Features

* **SteerLM: Attribute Conditioned SFT as an (User-Steerable) alternative to RLHF** 
    * [Llama3-70B-SteerLM-Chat](https://huggingface.co/nvidia/Llama3-70B-SteerLM-Chat) aligned with NeMo-Aligner.
    * Corresponding reward model [Llama3-70B-SteerLM-RM](https://huggingface.co/nvidia/Llama3-70B-SteerLM-RM).
    * Learn more at our [SteerLM](https://arxiv.org/abs/2310.05344) and [HelpSteer2](https://arxiv.org/abs/2406.08673) papers.
* **Supervised Fine Tuning**
* **Reward Model Training**
* **Reinforcement Learning from Human Feedback using the [PPO](https://arxiv.org/pdf/1707.06347.pdf) Algorithm**
    * [Llama3-70B-PPO-Chat](https://huggingface.co/nvidia/Llama3-70B-PPO-Chat) aligned with NeMo-Aligner using TRT-LLM.
* **Direct Preference Optimization** as described in [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290)
    * [Llama3-70B-DPO-Chat](https://huggingface.co/nvidia/Llama3-70B-DPO-Chat) aligned with NeMo Aligner.
* **Self-Play Fine-Tuning (SPIN)** as described in [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/pdf/2401.01335)

## Learn More
* [Documentation](https://github.com/NVIDIA/NeMo-Aligner/blob/main/docs/README.md)
* [Examples](https://github.com/NVIDIA/NeMo-Aligner/tree/main/examples/nlp/gpt)
* [Tutorials](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/index.html)
* [Paper](https://arxiv.org/abs/2405.01481)

## Latest Release

For the latest stable release, please see the [releases page](https://github.com/NVIDIA/NeMo-Aligner/releases). All releases come with a pre-built container. Changes within each release will be documented in [CHANGELOG](https://github.com/NVIDIA/NeMo-Aligner/blob/main/CHANGELOG.md).

## Install Your Own Environment

### Requirements
NeMo-Aligner has the same requirements as the [NeMo Toolkit Requirements](https://github.com/NVIDIA/NeMo#requirements) with the addition of [PyTriton](https://github.com/triton-inference-server/pytriton).

### Install NeMo-Aligner
Please follow the same steps as outlined in the [NeMo Toolkit Installation Guide](https://github.com/NVIDIA/NeMo#installation).  After installing NeMo, execute the following additional command:
```bash
pip install nemo-aligner
```
Alternatively, if you prefer to install the latest commit:
```bash
pip install .
```

### Docker Containers

We provide an official NeMo-Aligner Dockerfile which is based on stable, tested versions of NeMo, Megatron-LM, and TransformerEngine. The primary objective of this Dockerfile is to ensure stability, although it might not always reflect the very latest versions of those three packages. You can access our Dockerfile [here](https://github.com/NVIDIA/NeMo-Aligner/blob/main/Dockerfile).

Alternatively, you can build the NeMo Dockerfile here [NeMo Dockerfile](https://github.com/NVIDIA/NeMo/blob/main/Dockerfile) and add `RUN pip install nemo-aligner` at the end.

## Future work
- We will continue improving the stability of the PPO learning phase.
- Improve the performance of RLHF.
- Add TRT-LLM inference support for Rejection Sampling.

## Contribute to NeMo-Aligner
We welcome community contributions! Please refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo-Aligner/blob/main/CONTRIBUTING.md) for guidelines.

## Cite NeMo-Aligner in Your Work
```
@misc{shen2024nemoaligner,
      title={NeMo-Aligner: Scalable Toolkit for Efficient Model Alignment},
      author={Gerald Shen and Zhilin Wang and Olivier Delalleau and Jiaqi Zeng and Yi Dong and Daniel Egert and Shengyang Sun and Jimmy Zhang and Sahil Jain and Ali Taghibakhshi and Markel Sanz Ausin and Ashwath Aithal and Oleksii Kuchaiev},
      year={2024},
      eprint={2405.01481},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
This toolkit is licensed under the [Apache License, Version 2.0.](https://github.com/NVIDIA/NeMo-Aligner/blob/main/LICENSE)
