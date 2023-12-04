# Documentation

## Custom Trainers

NeMo-Aligner uses custom trainers to coordinate all aspects of training. There are currently 3 custom trainers:
1. [SupervisedTrainer](/nemo_aligner/algorithms/supervised.py): for SFT, SteerLM and Reward modeling.
2. [CriticServerTrainer](/nemo_aligner/algorithms/critic_server_trainer.py): trains the RL critic via PyTriton requests. It will also run the reward model depending on the configuration.
3. [PPOTrainer](/nemo_aligner/algorithms/ppo.py): performs the RLHF PPO training, since PPO has components such as the Critic, this trainer will send inference and train requests via [PyTriton](https://github.com/triton-inference-server/pytriton) to the CriticServerTrainer to train and run inference on the critic.

## Configuration guide

See the example configurations in the [conf folder](/examples/nlp/gpt/conf/) for an explanation of different configurations we support. Note that all specified configurations in the `.yaml` file will overwrite the loaded model configuration from the pretrained checkpoint.


## APIs
Our custom trainers will only call predefined APIs on the model passed in. These APIs are defined in [alignable_interface.py](/nemo_aligner/models/alignable_interface.py).

## Launching scripts and their description
* Supervised Fine Tuning Training: [train_gpt_sft.py](/examples/nlp/gpt/train_gpt_sft.py) with [gpt_sft.yaml](/examples/nlp/gpt/conf/gpt_sft.yaml).
* Reward Model Training: [train_reward_model.py](/examples/nlp/gpt/train_reward_model.py) with [training_rm.yaml](/examples/nlp/gpt/conf/training_rm.yaml).
* Reward Model Inference: [serve_reward_model.py](/examples/nlp/gpt/serve_reward_model.py) with [inference_rm.yaml](/examples/nlp/gpt/conf/inference_rm.yaml).
* PPO Critic Server: [serve_ppo_critic.py](/examples/nlp/gpt/serve_ppo_critic.py) with [gpt_ppo_critic.yaml](/examples/nlp/gpt/conf/gpt_ppo_critic.yaml).
* PPO Actor Training: [train_gpt_ppo_actor.py](/examples/nlp/gpt/train_gpt_ppo_actor.py) with [gpt_ppo_actor.yaml](/examples/nlp/gpt/conf/gpt_ppo_actor.yaml).

To run a full RLHF PPO job, we need to start both the CriticServerTrainer and PPOTrainer.

## Training architecture and details
Please see [Training.md](./training.md).