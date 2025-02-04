# Experimental GRPO implementation for LLM RL reasoning

This experimental sub-package contains an implementation of GRPO with user-specifiable 'Environments' that provide feedback for the LLM to learn from. 

## GRPO Sturcture:
```
NeMo-Aligner/nemo_aligner/experimental/grpo/
├── algorithms/
│   └── grpo.py                     <----- main GRPO training loop
├── data/
│   ├── datasets.py                 <----- Mixed-task RL datasets
│   └── builders.py
├─- examples/
│   └── grpo_math_llama_8b.sh       <----- a basic example that trains Llama8B with MATH rewards
├─- experience/
│   ├── environments                <----- where all reward-providers live
│   └── rollout_generator.py        <----- a handler for sampling from the LLM, calling environments, and managing state
├─- models/nlp/gpt/
│   └── megatron_gpt_grpo_actor.py  <----- GRPO model with loss definition
└── utils/
```

## Environment Abstraction
Contrary to RLHF, we may want to use a large mixture of diverse tasks and reward-gathering schemes for RL Reasoning training. We may want a Generative LLM Reward model, or a simpler math symbolic checker. 

To support this, we create an [```EnvironmentInterface```](experience/interfaces.py) that defines ```start_step```, ```finish_step```, and ```global_post_process_and_metrics``` functions. This allows you to define arbitrary asynchronously executing environments and have everything you need to calculate rewards and metrics.

### Example: [```MathEnvironment```](experience/environments/math_environment.py)
This provided example sets up a client to a remote ```flask``` server that gets called on ```start_step``` and returns a ```future``` that is consumed on ```finish_step```.  

## Dataset Structure
To integrate with the ```EnvironmentInterface``` abstraction, each sample of your dataset should contain a ```task_name``` field that gets mapped to an Environment Object in the training script ([example](../../../examples/nlp/gpt/train_gpt_grpo.py#L165))

