# Accelerated Reinforcement Learning From Human Feedback

For more details beyond the usage guide please see the NeMo-Aligner [paper](https://arxiv.org/abs/2405.01481).

## Description
Response generation during the RLHF PPO rollout phase constitutes a majority of the RLHF step time, taking up as much as 90% of total train time if not optimized. To address these bottlenecks, we use [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and their fast inference kernels to accelerate the generation stage. In our ablation experiments we observed a 6.96x speedup with our TRT-LLM integration, and we are working on making this speedup even better.

## Environment

We're working on adding all our dependencies into the [NeMo-FW-Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo), in the meantime we provide a [Dockerfile](Dockerfile) that can be built with all our dependencies.

## How it works
At the start of RLHF training, we compile the engine with TRT-LLM. This first compilation will take more time than other steps, in other steps we simply take the existing compiled engine and push updated model weights to it. Training is still done using the [NeMo-FW](https://github.com/NVIDIA/NeMo) which contain efficient training kernels.

## Usage Guide

To begin please follow the usage guide in the [Tutorials](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/index.html) page for RLHF. All the other configurations work just as before, but with TRT-LLM we have now added the [trainer.ppo.trt_llm](examples/nlp/gpt/conf/gpt_ppo_actor.yaml#L39) subconfig in the PPO actor.

## Performance tuning guide
There are a few configurations to consider when using TRT-LLM with RLHF.

* `trainer.ppo.trt_llm.enable`: Turns on and off TRT-LLM 
* `trainer.ppo.trt_llm.reshard`: If this flag is on and TRT-LLM is enabled, we will reshard the model to go from pipeline parallelism to tensor parallelism only during inference. NeMo training will still be with pipeline parallelism. When this option is activated, distributed groups within the TRT-LLM inference context treat pipeline parallel groups as data parallel groups. Caution must be used to handle data sharding.
* `trainer.ppo.trt_llm.unload_engine_train`: If this flag is enabled, then we will unload the engine when training. The benefit of unloading the engine when training is that it frees up more memory but comes at a cost of taking time doing this onloading. For the most optimal configuration, we reduce the rollout microbatch size but keep the engine while training(i.e set this boolean to false). 

During the TRT-LLM optimization phase, we also noticed that data parallel workers can have significantly different generation times. To balance it out we have a flask server hosted on rank 0 that acts as a distributed queue and distributes work to the other workers. This can be set with `trainer.flask_server.enable=True`.

## Performance
We are working on improving the performance of our TRT-LLM and will post the most up to date numbers in this README as we keep improving. The current performance numbers are as follows:

| Actor + Critic Node Count | Time per PPO Step in seconds | Estimated Time to train [hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) | Scaling from Base |
|---------------------------|----------------------------- |------------------------------------------------------------------------------------|-------------------|
| 8 + 8                     | 253.8                        | 11.1 hours                                                                         | 1                 |
| 16 + 16                   | 143.4                        | 6.3 hours                                                                          | **1.77x**         |
| 32 + 32                   | 81.2                         | 3.5 hours                                                                          | **3.13x**         |

Time per PPO Step on LLaMa2 70B Actor and Critic. Number of rollout samples is 1024, and the training global batch size is 128.
