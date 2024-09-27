.. include:: /content/nemo.rsts

.. _model-aligner-rlhf:

Model Alignment by RLHF
@@@@@@@@@@@@@@@@@@@@@@@

For the purposes of this tutorial, we will go through the entire Reinforcement Learning from Human Feedback (RLHF) pipeline using models from the NeMo Framework. These models can include LLaMa2 or Mistral, and our scripts will function consistently across them.

RLHF is usually preceded by a Supervised Fine-Tuning (SFT). We should first follow the :ref:`Prerequisite guide <prerequisite>` and the :ref:`SFT guide <sft>`. After obtaining the SFT model, we will use this to start the RLHF process. We will use the `PPO <https://arxiv.org/abs/1707.06347>`__ algorithm for reinforcement learning on the `Anthropic-HH-RLHF <https://huggingface.co/datasets/Anthropic/hh-rlhf>`__ dataset.

Data Processing for RLHF
#########################

We have a script ready to use for processing the Anthropic-HH dataset into a jsonlines format. Run the following command on the `download_and_process.py <https://github.com/NVIDIA/NeMo-Megatron-Launcher/blob/8d4f34c9da6b3254ca316b2c43ee88b77a894529/launcher_scripts/nemo_launcher/collections/dataprep_scripts/anthropichh_dataprep/download_and_process.py#L1>`__ script for anthropic HH.

   .. code-block:: bash 

      python download_and_process.py

After running this script, you should have the files ``{train,test}_comparisons.jsonl`` and ``{train,test}_prompts.jsonl``. Comparison files are used for reward model training, whereas prompts files are used for the reinforcement learning training.

Reward Model Training
#####################

The reward model is used to score the quality of the response. It is trained using a pairwise comparison loss, which means it requires a dataset of response pairs. In each pair, one response is ranked higher than the other. A well-constructed reward model is essential for the success of the Proximal Policy Optimization (PPO) training process.

Data Preprocessing
%%%%%%%%%%%%%%%%%%

You can also bring your own data for the reward model training phase. The reward model datasets require the following format::

   {"text": prompt1 || good_response_1}
   {"text": prompt1 || bad_response_1}
   {"text": prompt2 || good_response_2}
   {"text": prompt2 || bad_response_2}
   ...

The || denotes a string concatenation, whereas *prompt1* and *prompt2* are different prompts. Note that for the same prompt, *prompt || good_response* must come before *prompt || bad_response* in the dataset.

An example JSONL file can look like the following::

  {"text": "User: When did Virgin Australia start operating?\nAssistant: 31 August 2000"}
  {"text": "User: When did Virgin Australia start operating?\nAssistant: I refuse to answer this question."}
  {"text": "User: What is 6*10?\nAssistant: 60"}
  {"text": "User: What is 6*10?\nAssistant: 90"}
  ...


To launch reward model training, you must start with a pretrained or SFT-trained model. For this section, we will use the SFT model trained in the previous step to train the reward model.

.. tab-set::

    .. tab-item:: Terminal
        :sync: key3

         To run reward model training on the terminal directly:

         .. code-block:: bash 

            GPFS="/path/to/nemo-aligner-repo"
            TRAIN_DATA_PATH="/path/to/train_comparisons.jsonl"
            VALID_DATA_PATH="/path/to/test_comparisons.jsonl"

            python -u ${GPFS}/examples/nlp/gpt/train_reward_model.py \
               trainer.num_nodes=1 \
               trainer.devices=8 \
               ++model.micro_batch_size=1 \
               ++model.global_batch_size=512 \
               ++model.data.data_impl=jsonl \
               pretrained_checkpoint.restore_from_path=/path/to/megatron_gpt_sft.nemo \
               "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
               exp_manager.create_wandb_logger=False \
               exp_manager.wandb_logger_kwargs.project=rm_training \
               exp_manager.wandb_logger_kwargs.name=rm_training \
               exp_manager.explicit_log_dir=/results

    .. tab-item:: Slurm
        :sync: key4

         To run reward model training with Slurm, use the following script. The script below uses 4 nodes, but you can change the node count to something smaller.

         .. code-block:: bash 

            #!/bin/bash
            #SBATCH -A <<ACCOUNT NAME>>
            #SBATCH -p <<PARTITION NAME>>
            #SBATCH -N 4
            #SBATCH -t 4:00:00
            #SBATCH -J <<JOB NAME>>
            #SBATCH --ntasks-per-node=8
            #SBATCH --exclusive
            #SBATCH --overcommit

            GPFS="/path/to/nemo-aligner-repo"
            PRETRAINED_CHECKPOINT_NEMO_FILE="/path/to/megatron_gpt_sft.nemo"

            TRAIN_DATA_PATH="/path/to/train_comparisons.jsonl"
            VALID_DATA_PATH="/path/to/test_comparisons.jsonl"

            PROJECT="<<WANDB PROJECT>>"

            CONTAINER=<<<CONTAINER>>> # use the latest NeMo Training container, Aligner will work there

            MOUNTS="--container-mounts=MOUNTS" # mounts

            RESULTS_DIR="/path/to/result_dir"

            OUTFILE="${RESULTS_DIR}/rm-%j_%t.out"
            ERRFILE="${RESULTS_DIR}/rm-%j_%t.err"
            mkdir -p ${RESULTS_DIR}

            MOUNTS="--container-mounts=MOUNTS" # mounts

            read -r -d '' cmd <<EOF
            echo "*******STARTING********" \
            && echo "---------------" \
            && echo "Starting training" \
            && cd ${GPFS} \
            && export PYTHONPATH="${GPFS}:${PYTHONPATH}" \
            && export HYDRA_FULL_ERROR=1 \
            && python -u ${GPFS}/examples/nlp/gpt/train_reward_model.py \
               trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
               trainer.devices=8 \
               pretrained_checkpoint.restore_from_path='${PRETRAINED_CHECKPOINT_NEMO_FILE}' \
               "++model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
               ++model.micro_batch_size=1 \
               ++model.global_batch_size=512 \
               ++model.data.data_impl=jsonl \
               exp_manager.explicit_log_dir=${RESULTS_DIR} \
               exp_manager.create_wandb_logger=True \
               exp_manager.wandb_logger_kwargs.name=${NAME} \
               trainer.rm.save_interval=500 \
               trainer.rm.val_check_interval=100 \
               trainer.rm.limit_val_batches=100000 \
               exp_manager.wandb_logger_kwargs.project=${PROJECT}
            EOF

            srun -o $OUTFILE -e $ERRFILE --container-image=$CONTAINER $MOUNTS bash -c "${cmd}"
            set +x


*Remark: Currently, the example training script does not automatically run evaluation on the provided test set. This may change in a future release.* 

During reward model training, it’s expected that the validation accuracy improves as the training progresses. In the example provided above using Slurm, we achieved a validation accuracy of 69.57%. 

With the finished training, NeMo-Aligner will save a ``megatron_gpt.nemo`` which is the reward model we need for the RL stage.

PPO Training
############

After you have fine-tuned a GPT model using SFT and trained a reward model as explained in the preceding section, you can proceed with RLHF using PPO.

During PPO training, we conceptually have four models that interact with each other:

#. The PPO Actor Network (also known as the Policy Network): This is the model we are training. It should start from an SFT model.
#. The Reward Model (RM) Network (also known as a Preference Model (PM)): This model takes a prompt concatenated with a response as input and outputs a single scalar value, the reward, which the PPO algorithm will try to maximize.
#. The PPO Critic Network (also known as the Value Network): Since PPO is an Actor-Critic algorithm, we need a Critic to guide the Actor during training. The Critic will provide value estimates for each token in the responses provided by the Actor. These values can be seen as an estimate of the total reward the Actor will receive after generating all the remaining tokens. The Critic should be initialized from the RM so as to provide useful feedback in the early stages of training. Note: The RM generates a single reward for the entire sequence, whereas the Critic generates a value for each token.
#. The Initial Policy Network (also known as the Reference Model): We use this model to compute a KL Divergence penalty term that ensures that the PPO Actor does not diverge too much from the Initial Policy. This way, we prevent the PPO Actor from overfitting to the rewards given by the RM, and ensure it does not forget the knowledge it acquired during pretraining and SFT. This model should be the one used to initialize the PPO Actor Network.

In the most optimized configuration, NeMo-Aligner will run the Actor and initial policy within the same job as well as the Critic and reward model within the same job. It will then use CPU offloading to load back the corresponding model when needed.
   
The next section discusses how to launch each of these two jobs.

Launching the Reward Model and Critic Server
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

To launch the server:

.. code-block:: bash 

   #!/bin/bash
   # Example: If using the reward model trained from the above configuration, you can find
              the trained reward model checkpoint here: "/results/checkpoints/megatron_gpt.nemo"
   CHECKPOINT_NEMO_FILE="/path/to/trained_rm.nemo"
   GPFS="/path/to/nemo-aligner-repo"

   RESULTS_DIR="critic_results_dir"

   export PYTHONPATH="${GPFS}:${PYTHONPATH}" \
   && export HYDRA_FULL_ERROR=1 \
   && python -u ${GPFS}/examples/nlp/gpt/serve_ppo_critic.py \
      trainer.devices=8 \
      trainer.num_nodes=1 \
      ++model.tensor_model_parallel_size=1 \
      ++model.pipeline_model_parallel_size=1 \
      exp_manager.create_wandb_logger=False \
      exp_manager.wandb_logger_kwargs.name=critic_training \
      exp_manager.wandb_logger_kwargs.project=nemo_aligner_ppo \
      exp_manager.explicit_log_dir=${RESULTS_DIR} \
      trainer.ppo.inference_micro_batch_size=4 \
      ++pretrained_checkpoint.restore_from_path=${CHECKPOINT_NEMO_FILE} \
      ++model.megatron_amp_O2=True \
      ++model.activations_checkpoint_granularity=null \
      ++trainer.ppo.combine_rm_and_critic_server=True \
      ++model.offload_adam_states=True \
      ++model.mcore_gpt=True

The above example launches the reward model Critic server on 8 GPUs and 1 node. Please make sure to change ``trainer.devices``, ``trainer.num_nodes`` depending on your model size and scale. NeMo-Aligner will work on any scale. In addition, make sure to tune the `trainer.ppo.inference_micro_batch_size` argument as this determines the batch size the PPO Actor is allowed to send to the Critic per DP rank.

Launching the Initial Policy and PPO Actor Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The PPO Actor training job contains the master controller that makes the HTTP calls to all servers when needed. To launch the PPO Actor and Initial Policy server:

.. code-block:: bash 

   GPFS="/path/to/nemo-aligner-repo"
   TRAIN_DATA_PATH="/path/to/train_prompts.jsonl"
   VALID_DATA_PATH="/path/to/test_prompts.jsonl"

   PRETRAINED_ACTOR_NEMO_FILE="/path/to/sft_checkpoint.nemo"

   RESULTS_DIR="actor_results_dir"

   export PYTHONPATH="${GPFS}:${PYTHONPATH}" \
   && export HYDRA_FULL_ERROR=1 \
   && python -u ${GPFS}/examples/nlp/gpt/train_gpt_ppo_actor.py \
      "++model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
      ++model.data.data_impl=jsonl \
      pretrained_checkpoint.restore_from_path=${PRETRAINED_ACTOR_NEMO_FILE} \
      trainer.num_nodes=1 \
      trainer.devices=8 \
      ++model.pipeline_model_parallel_size=1 \
      ++model.tensor_model_parallel_size=1 \
      ++model.ppo.combine_rm_and_critic_server=True \
      ++model.ppo.offload_adam_states=True \
      ++model.megatron_amp_O2=True \
      ++trainer.ppo.normalize_advantages=True \
      ++model.mcore_gpt=True \
      exp_manager.create_wandb_logger=False \
      exp_manager.wandb_logger_kwargs.name=ppo_actor_training \
      exp_manager.wandb_logger_kwargs.project=nemo_aligner_ppo \
      exp_manager.explicit_log_dir=/rlhf/actor_test \
      ++model.ppo.entropy_bonus=0.0 \
      remote_critic_rm.pad_to_length=2048

The above script launches the initial and Actor server on 1 node with 8 GPUs.

.. note::
   For more info on PPO hyperparameters, see `PPO Hparams <https://github.com/NVIDIA/NeMo-Aligner/blob/main/docs/RLHFTraining.md#ppo-hyperparameters>`__.

Launching Both Servers for RLHF Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

You can use Slurm to launch both jobs and coordinate them together in a full RLHF job using the following script:

.. code-block:: bash 

   #!/bin/bash
   #SBATCH -N 1 --ntasks-per-node 8 -A <<ACCOUNT>> -p <<PARTITION>> --job-name <<JOBNAME>> -t 4:00:00 --exclusive
   #SBATCH hetjob
   #SBATCH -N 1 --ntasks-per-node 8 -A <<ACCOUNT>> -p <<PARTITION>> --job-name <<JOBNAME>> -t 4:00:00 --exclusive

   NAME="2p_ppo"

   # PARAMETERS
   RM_NEMO_FILE="/path/to/trained_rm.nemo"

   ACTOR_NEMO_FILE="/path/to/sft_model.nemo"

   TRAIN_DATA_PATH="/path/to/train_prompts.jsonl"
   VALID_DATA_PATH="/path/to/test_prompts.jsonl"

   RESULTS_DIR="/path/to/results_dir"
   mkdir -p $RESULTS_DIR

   GPFS="/path/to/nemo-aligner-repo"
   MOUNTS="--container-mounts=MOUNTS" # mounts

   CONTAINER=<<<CONTAINER>>> # use the latest NeMo Training container, Aligner will work there

   PROJECT=ppo_run

   CRITIC_LOG_DIR="${RESULTS_DIR}/critic_results"
   CRITIC_OUTFILE="${CRITIC_LOG_DIR}/critic_output_%j_%t.log"
   CRITIC_ERRFILE="${CRITIC_LOG_DIR}/critic_error_%j_%t.err"
   CRITIC_PORT=5567

   mkdir -p $CRITIC_LOG_DIR

   CRITIC_NAME="${NAME}_critic"

   read -r -d '' cmd_critic_inference <<EOF
   cd ${GPFS} \
   && export PYTHONPATH="${GPFS}:${PYTHONPATH}" \
   && export HYDRA_FULL_ERROR=1 \
   && python -u ${GPFS}/examples/nlp/gpt/serve_ppo_critic.py \
      trainer.ppo.inference_micro_batch_size=4 \
      trainer.devices=8 \
      trainer.num_nodes=${SLURM_JOB_NUM_NODES_HET_GROUP_0} \
      exp_manager.explicit_log_dir=${CRITIC_LOG_DIR} \
      exp_manager.create_wandb_logger=True \
      exp_manager.wandb_logger_kwargs.name=${CRITIC_NAME} \
      exp_manager.wandb_logger_kwargs.project=${PROJECT} \
      trainer.ppo.port=${CRITIC_PORT} \
      ++model.offload_adam_states=True \
      ++model.micro_batch_size=1 \
      ++model.global_batch_size=64 \
      pretrained_checkpoint.restore_from_path=${RM_NEMO_FILE}
   EOF

   srun --het-group=0 -o $CRITIC_OUTFILE -e $CRITIC_ERRFILE --container-image=${CONTAINER} $MOUNTS bash -c "${cmd_critic_inference}" &

   sleep 30

   ACTOR_LOG_DIR="${RESULTS_DIR}/actor_results"
   CHECKPOINT_DIR="${ACTOR_LOG_DIR}/checkpoints"
   TENSOBOARD_DIR="${ACTOR_LOG_DIR}/tensorboard"

   PPO_ERRFILE="${ACTOR_LOG_DIR}/actor_error_%j_%t.err"
   PPO_OUTFILE="${ACTOR_LOG_DIR}/actor_output_%j_%t.log"

   mkdir -p $ACTOR_LOG_DIR
   mkdir -p $TENSOBOARD_DIR
   mkdir -p $CHECKPOINT_DIR

   ACTOR_NAME="${NAME}_actor"

   host_critic="$(scontrol show hostnames=$SLURM_JOB_NODELIST_HET_GROUP_0 | head -n1)"

   read -r -d '' cmd_ppo <<EOF
   cd ${GPFS} \
   && export PYTHONPATH="${GPFS}:${PYTHONPATH}" \
   && export HYDRA_FULL_ERROR=1 \
   && python -u ${GPFS}/examples/nlp/gpt/train_gpt_ppo_actor.py \
      trainer.devices=8 \
      trainer.num_nodes=${SLURM_JOB_NUM_NODES_HET_GROUP_1} \
      trainer.ppo.max_steps=15 \
      ++model.data.data_impl=jsonl \
      "++model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
      pretrained_checkpoint.restore_from_path=${ACTOR_NEMO_FILE} \
      exp_manager.explicit_log_dir=${ACTOR_LOG_DIR} \
      exp_manager.create_wandb_logger=True \
      exp_manager.wandb_logger_kwargs.name=${ACTOR_NAME} \
      exp_manager.wandb_logger_kwargs.project=${PROJECT} \
      ++model.micro_batch_size=1 \
      ++model.global_batch_size=64 \
      ++model.activations_checkpoint_granularity=selective \
      ++model.activations_checkpoint_method=uniform \
      ++model.optim.lr=9e-7 \
      trainer.ppo.val_check_interval=3 \
      ++model.optim.sched.min_lr=9e-8 \
      ++model.ppo.entropy_bonus=0.0 \
      ++model.ppo.ratio_eps=0.2 \
      ++model.ppo.num_rollout_samples=512 \
      ++model.ppo.rollout_micro_batch_size=8 \
      ++model.ppo.length_params.max_length=1024 \
      trainer.ppo.initial_policy_kl_penalty=0.02 \
      remote_critic_rm.critic.ip=${host_critic} \
      remote_critic_rm.critic.port=${CRITIC_PORT}
   EOF

   srun --het-group=1 -o $PPO_OUTFILE -e $PPO_ERRFILE --container-image=${CONTAINER} $MOUNTS bash -c "${cmd_ppo}" &

   wait

The above script runs the reward model Critic server on 1 node and the Actor on 1 node.

It is important to launch all jobs with ``&`` after the srun command, to ensure they do not block each other. 

.. note::
   Make sure to change the Critic arg ``trainer.ppo.inference_micro_batch_size`` such that ``trainer.ppo.inference_micro_batch_size * DP size <= model.ppo.rollout_micro_batch_size``.

Speeding up PPO with TensorRT-LLM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NeMo-Aligner has support for accelerating RLHF with `TensorRT-LLM <https://github.com/NVIDIA/TensorRT-LLM>`__. This can provide a significant speedup to the PPO training time when enabled. There are a few crucial flags to set when using TensorRT-LLM with Aligner.

#. `trainer.ppo.trt_llm.enable=True` enables TensorRT-LLM.
#. `trainer.ppo.trt_llm.reshard=True` enables on the fly pipeline parallelism resharding during inference. This essentially runs training with pipeline parallelism but inference with tensor parallelism only.
#. `trainer.ppo.trt_llm.unload_engine_train=False`: may be set to True to unload the engine during training, which frees up more memory for training since the engine is on GPU but comes at a cost of loading and offloading the engine. It's recommended to keep this False.
#. `trainer.trt_llm.model_type=llama` tells TensorRT-LLM which model type we want to run inference on. Must be changed for other model types, as an example for nemotron this should be gptnext.
#. `trainer.ppo.batch_iterator.use_flask=True` enables a flask server to balance work across different DP workers.

For more information please see the aligner `paper <https://arxiv.org/abs/2405.01481>`__.

PPO Results with TensorRT-LLM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

We used the TensorRT-LLM PPO integration to train the `LLaMa3-70B-PPO-Chat <https://huggingface.co/nvidia/Llama3-70B-PPO-Chat>`__ model. This model was trained using a global batch size of 128, num_rollout_samples of 128, constant learning rate of 1e-7 and KL penalty of 3e-3. For more details please see the Helpsteer2 `paper <https://arxiv.org/abs/2406.08673>`__.

PPO Performance Results with TensorRT-LLM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

We test the scaling of our TRT-LLM integration by running Llama3 70B Actor and Llama3 70B Reward model on the Helpsteer2 prompts with ``num_rollout_samples=1024``, ``global_batch_size=128``, reshard enabled and engine offloading set to False.
+------------------+-------------------+-----------------------------+----------------------+-------------------+
| Actor Node Count | Critic Node Count | Number of Tokens Generated  | Step Time(seconds)  | Scaling Efficiency |
+==================+===================+=============================+======================+===================+
| 8                | 4                 | 350.7                       | 366.7                | 1                 |
+------------------+-------------------+-----------------------------+----------------------+-------------------+
| 16               | 8                 | 323.3                       | 190.4                | 1.93              |
+------------------+-------------------+-----------------------------+----------------------+-------------------+
| 32               | 16                | 331.7                       | 108                  | 3.40              |
+------------------+-------------------+-----------------------------+----------------------+-------------------+
| 64               | 32                | 334                         | 56.9                 | 6.44              |
+------------------+-------------------+-----------------------------+----------------------+-------------------+

NOTE: for 64x32 config we used a rollout_micro_batch_size of 16 instead of 8 since we have more memory coming from the distributed optimizer.

We also support running RLHF on Llama3.1 405B Actor and Reward Model. The following numbers are generated with ``num_rollout_samples=128``, ``global_batch_size=128``, reshard turned off, engine offloading set to False.

+------------------+-------------------+----------------------------+--------------------+
| Actor Node Count | Critic Node Count | Number of Tokens Generated | Step Time(seconds) |
+==================+===================+============================+====================+
| 84               | 42                | 915.6                      | 164.6              |
+------------------+-------------------+----------------------------+--------------------+

In the future we aim to improve the performance of generation with large models that have high pipeline parallelism size.

PPO Results
%%%%%%%%%%%

Once you've completed RLHF training, you can serve your model using the `megatron_gpt_eval.py <https://github.com/NVIDIA/NeMo/blob/8cd5f1c8e7d4fed9f4f946028cd02047c5d2296f/examples/nlp/language_modeling/megatron_gpt_eval.py#L4>`__ script from the NeMo codebase to run more rigorous evaluation of your trained model.
   
Scaling the Tutorial to Bigger Models
#####################################

While the tutorial above provides a way to get started with RLHF, it doesn’t represent the most optimal performance or convergence configuration. When running RLHF fully, we anticipate achieving an MT-bench score improvement of approximately +0.4 to +0.5. It’s essential to begin with a high-quality SFT model and closely monitor the response length.
