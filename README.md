# NVIDIA NeMo-Aligner

## Introduction

NeMo-Aligner is a scalable toolkit for efficient model alignment. The toolkit has support for fine tuning large language models using Reinforcement Learning from Human Feedback (RLHF). RLHF is a technique to align language models to be more safe, harmless and helpful. The toolkit supports the entire RLHF pipeline, starting from Supervised Fine Tuning(SFT) to training a reward model on comparison data, followed by the [Proximal Policy Optimization algorithm](https://arxiv.org/abs/1707.06347) for RL training.

NeMo-Aligner toolkit is built using the [NeMo Toolkit](https://github.com/NVIDIA/NeMo) which allows for scaling training up to 1000s of GPUs using tensor, data and pipeline parallelism for all components of alignment. All of our checkpoints are cross compatible with the NeMo ecosystem; allowing for inference deployment and further customization.

The toolkit is currently in it's early stages, and we are committed to improving the toolkit to make it easier for developers to pick and choose different alignment algorithms to build safe, helpful and reliable models.

## Key features

* **SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF.** Learn more at our [SteerLM](https://arxiv.org/abs/2310.05344) and [HelpSteer](https://arxiv.org/abs/2311.09528) papers. Try it instantly for free on [NVIDIA AI Playground](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/llama2-70b-steerlm)
* **Supervised Fine Tuning**
* **Reward Model Training**
* **Reinforcement Learning from Human Feedback using the PPO Algorithm**
* **[Direct Preference Optimization](https://arxiv.org/pdf/2305.18290.pdf)**

## Learn More
* [Documentation](./docs/README.md)
* [Examples](./examples/nlp/gpt/)
* [Tutorials](https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html): Please refer to the alignment section for step by step tutorials

## Requirements
NeMo-Aligner has the same requirements as the [NeMo Toolkit Requirements](https://github.com/NVIDIA/NeMo#requirements) and is built into the NeMo Training container.

## Installation
Please follow the same steps as the [NeMo Toolkit Installation Guide](https://github.com/NVIDIA/NeMo#installation) but add the following at the end to install NeMo-Aligner
```bash
pip install nemo-aligner
```
or if you prefer to install the latest commit
```bash
pip install .
```

### Docker Containers

We release along with the NeMo toolkit containers. You may find more details about released containers in [releases page](https://github.com/NVIDIA/NeMo-Aligner/releases).

To build your own, refer to the [Dockerfile](https://github.com/NVIDIA/NeMo/blob/main/Dockerfile) and add `RUN pip install nemo-aligner` at the end.

## Release history

For the list of changes within each release please see the [Changelog](CHANGELOG.md).

## Future work
- Add Rejection Sampling support
- We will continue improving the stability of the PPO learning phase.
- Improve the performance of RLHF

## Contributing
We welcome community contributions! Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
This toolkit is licensed under the [Apache License, Version 2.0.](LICENSE)
