.. include:: /content/nemo.rsts

.. _model-aligner-reinforce:

Model Alignment by REINFORCE
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

In this tutorial, we will guide you through the process of aligning a NeMo Framework model using REINFORCE. This method can be applied to various models, including LLaMa2 and Mistral, with our scripts functioning consistently across different models.

REINFORCE is usually preceded by a Supervised Fine-Tuning (SFT). We should first follow the :ref:`Prerequisite guide <prerequisite>` and the :ref:`SFT guide <sft>`. After obtaining the SFT model, we will also need to train a reward model as in :ref:`PPO guide <ppo>`. We will use the REINFORCE algorithm on the `Anthropic-HH-RLHF <https://huggingface.co/datasets/Anthropic/hh-rlhf>`__ dataset.

REINFORCE Training
############

After you have fine-tuned a GPT model using Supervised Fine-Tuning (SFT), and trained a reward model as explained in the preceding section, you can start aligning the policy using REINFORCE.

During REINFORCE training, we have three models interacting with each other, which Aligner runs in two separate jobs:

#. The Policy Network: This is the model we are training and it should start from an SFT model.
#. The Reward Model (RM): This model accepts a prompt combined with a response as input and produces a single scalar value, known as the reward. The REINFORCE algorithm aims to maximize this reward.
#. The Initial Policy Network (also known as the Reference Model): We use this model to compute a KL Divergence penalty term that ensures that the PPO Actor does not diverge too much from the Initial Policy. This way, we prevent the REINFORCE Actor from overfitting to the rewards given by the RM, and ensure it does not forget the knowledge it acquired during pretraining and SFT. This model should be the one used to initialize the REINFORCE Actor Network.

The next section discusses how to launch each of these two jobs.

Launching the Reward Model and Critic Server
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

To launch the server:

.. code-block:: bash 

   #!/bin/bash
   CHECKPOINT_NEMO_FILE="/path/to/trained_rm.nemo"
   GPFS="/path/to/nemo-aligner-repo"

   RESULTS_DIR="critic_results_dir"

   cd ${GPFS}
   export PYTHONPATH="${GPFS}:${PYTHONPATH}" \
   && export HYDRA_FULL_ERROR=1 \
   && python -u examples/nlp/gpt/serve_reward_model.py \
      trainer.num_nodes=1 \
      trainer.devices=8 \
      ++model.tensor_model_parallel_size=4 \
      rm_model_file=${RM_NEMO_FILE}


The above example launches the reward model server on eight GPUs and one node. Make sure to change trainer.devices, trainer.num_nodes depending on your model size and scale. Aligner will work on any scale. Also, make sure to tune the trainer.reinforce.inference_micro_batch_size argument. This argument sets the size of the batch the REINFORCE actor is allowed to send to the reward per DP rank.

Launch the Initial Policy and REINFORCE Actor Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The REINFORCE Actor training job contains the master controller that makes the HTTP calls to all servers when needed. To launch the REINFORCE Actor and Initial Policy server:

.. code-block:: bash 

   GPFS="/path/to/nemo-aligner-repo"
   TRAIN_DATA_PATH="/path/to/train_prompts.jsonl"
   VALID_DATA_PATH="/path/to/test_prompts.jsonl"

   PRETRAINED_ACTOR_NEMO_FILE="/path/to/sft_checkpoint.nemo"
   RESULTS_DIR="/path/to/actor_results_dir"

   USE_FLASK=False
   ACTOR_LR=1e-6
   KL=0.01
   NUM_ROLLOUTS=32
   ACTOR_GBS=32
   REWARD_PORT=5555
   # Change this to the hostname of server hosting the reward model 
   host_reward="localhost"

   cd ${GPFS}
   export PYTHONPATH="${GPFS}:${PYTHONPATH}" \
   && export HYDRA_FULL_ERROR=1 \
   && python -u examples/nlp/gpt/train_gpt_reinforce_actor.py \
      "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
      pretrained_checkpoint.restore_from_path=\"${ACTOR_NEMO_FILE}\" \
      exp_manager.checkpoint_callback_params.save_top_k=1 \
      exp_manager.explicit_log_dir=\"${RESULTS_DIR}\" \
      trainer.reinforce.max_epochs=1 \
      trainer.reinforce.max_steps=313 \
      trainer.reinforce.val_check_interval=4 \
      trainer.num_nodes=1 \
      trainer.devices=8 \
      trainer.reinforce.trt_llm.enable=True \
      trainer.reinforce.trt_llm.reshard=True \
      trainer.reinforce.trt_llm.unload_engine_train=False \
      ++model.tensor_model_parallel_size=4 \
      ++model.reinforce.num_rollout_samples=${NUM_ROLLOUTS} \
      model.global_batch_size=${ACTOR_GBS} \
      model.micro_batch_size=1 \
      model.optim.lr=\"\\\$\{multiply:${ACTOR_LR},1.001\}\" \
      model.optim.sched.warmup_steps=0 \
      model.optim.sched.constant_steps=312 \
      model.optim.sched.min_lr=${ACTOR_LR} \
      model.optim.weight_decay=0.01 \
      model.reinforce.rollout_micro_batch_size=16 \
      model.reinforce.forward_micro_batch_size=16 \
      model.reinforce.val_rollout_micro_batch_size=8 \
      model.data.data_impl=jsonl \
      remote_rm.reward_model.ip=${host_reward} \
      remote_rm.reward_model.port=${REWARD_PORT} \
      ++model.reinforce.length_params.max_length=2048 \
      trainer.reinforce.initial_policy_kl_penalty="${KL}" \
      ++model.optim.bucket_cap_mb=200 \
      ++model.dist_ckpt_format=zarr \
      ++model.optim.overlap_grad_sync=False \
      ++model.optim.contiguous_grad_buffer=True \
      ++model.enable_nge=True \
      trainer.reinforce.batch_iterator.use_flask=${USE_FLASK} \
      trainer.reinforce.rollout_batch_seq_length=4096

The above command launches the initial and actor server on one node with eight GPUs.

Launching Both Servers for REINFORCE training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

You can use slurm to launch the two jobs and get them to coordinate together in a full REINFORCE job through the following:

.. code-block:: bash 

   #!/bin/bash
   #SBATCH -N 1 --ntasks-per-node 8 -A <<ACCOUNT>> -p <<PARTITION>> --job-name <<JOBNAME>> -t 4:00:00 --exclusive
   #SBATCH hetjob
   #SBATCH -N 1 --ntasks-per-node 8 -A <<ACCOUNT>> -p <<PARTITION>> --job-name <<JOBNAME>> -t 4:00:00 --exclusive

   NAME="reinforce"

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

   PROJECT=reinforce_run

   CRITIC_LOG_DIR="${RESULTS_DIR}/critic_results"
   CRITIC_OUTFILE="${CRITIC_LOG_DIR}/critic_output_%j_%t.log"
   CRITIC_ERRFILE="${CRITIC_LOG_DIR}/critic_error_%j_%t.err"
   REWARD_PORT=5567
   CRITIC_CONFIG_PATH="${GPFS}/examples/nlp/gpt/conf"
   CRITIC_CONFIG_NAME="inference_rm"

   CONF_DIR="${GPFS}/examples/nlp/gpt/conf"
   CONFIG_NAME="gpt_reinforce_actor"

   mkdir -p $CRITIC_LOG_DIR

   CRITIC_NAME="${NAME}_critic"

   read -r -d '' cmd_critic_inference <<EOF
   cd ${GPFS} \
   && export PYTHONPATH="${GPFS}:${PYTHONPATH}" \
   && export HYDRA_FULL_ERROR=1 \
   && python -u examples/nlp/gpt/serve_reward_model.py \
      --config-path=${CRITIC_CONFIG_PATH} \
      --config-name=${CRITIC_CONFIG_NAME} \
      trainer.num_nodes=1 \
      trainer.devices=8 \
      ++model.tensor_model_parallel_size=4 \
      rm_model_file=${RM_NEMO_FILE} \
      inference.port=${REWARD_PORT}
   EOF

   srun --het-group=0 -o $CRITIC_OUTFILE -e $CRITIC_ERRFILE --container-image=${CONTAINER} $MOUNTS bash -c "${cmd_critic_inference}" &

   sleep 30

   ACTOR_LOG_DIR="${RESULTS_DIR}/actor_results"
   CHECKPOINT_DIR="${ACTOR_LOG_DIR}/checkpoints"
   TENSOBOARD_DIR="${ACTOR_LOG_DIR}/tensorboard"

   NUM_ROLLOUTS=32
   NORMALIZE="True"
   ACTOR_LR="1e-6"
   ACTOR_GBS=32
   KL=0.01
   USE_FLASK=False

   mkdir -p $ACTOR_LOG_DIR
   mkdir -p $TENSOBOARD_DIR
   mkdir -p $CHECKPOINT_DIR

   ACTOR_NAME="${NAME}_actor"

   host_reward="$(scontrol show hostnames=$SLURM_JOB_NODELIST_HET_GROUP_0 | head -n1)"

   read -r -d '' cmd_reinforce <<EOF
   cd ${GPFS}
   export PYTHONPATH="${GPFS}:${PYTHONPATH}" \
   && export HYDRA_FULL_ERROR=1 \
   && python -u examples/nlp/gpt/train_gpt_reinforce_actor.py \
      "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
      pretrained_checkpoint.restore_from_path=\"${ACTOR_NEMO_FILE}\" \
      exp_manager.checkpoint_callback_params.save_top_k=1 \
      exp_manager.explicit_log_dir=\"${RESULTS_DIR}\" \
      trainer.reinforce.max_epochs=1 \
      trainer.reinforce.max_steps=313 \
      trainer.reinforce.val_check_interval=4 \
      trainer.num_nodes=1 \
      trainer.devices=8 \
      trainer.reinforce.trt_llm.enable=True \
      trainer.reinforce.trt_llm.reshard=True \
      trainer.reinforce.trt_llm.unload_engine_train=False \
      ++model.tensor_model_parallel_size=4 \
      ++model.reinforce.num_rollout_samples=${NUM_ROLLOUTS} \
      model.global_batch_size=${ACTOR_GBS} \
      model.micro_batch_size=1 \
      model.optim.lr=\"\\\$\{multiply:${ACTOR_LR},1.001\}\" \
      model.optim.sched.warmup_steps=0 \
      model.optim.sched.constant_steps=312 \
      model.optim.sched.min_lr=${ACTOR_LR} \
      model.optim.weight_decay=0.01 \
      model.reinforce.rollout_micro_batch_size=16 \
      model.reinforce.forward_micro_batch_size=16 \
      model.reinforce.val_rollout_micro_batch_size=8 \
      model.data.data_impl=jsonl \
      remote_rm.reward_model.ip=${host_reward} \
      remote_rm.reward_model.port=${REWARD_PORT} \
      ++model.reinforce.length_params.max_length=2048 \
      trainer.reinforce.initial_policy_kl_penalty="${KL}" \
      ++model.optim.bucket_cap_mb=200 \
      ++model.dist_ckpt_format=zarr \
      ++model.optim.overlap_grad_sync=False \
      ++model.optim.contiguous_grad_buffer=True \
      ++model.enable_nge=True \
      trainer.reinforce.batch_iterator.use_flask=${USE_FLASK} \
      trainer.reinforce.rollout_batch_seq_length=4096
   EOF

   srun --het-group=1 -o $PPO_OUTFILE -e $PPO_ERRFILE --container-image=${CONTAINER} $MOUNTS bash -c "${cmd_reinforce}" &

   wait

The above script runs the reward model server on one node and the actor on one node.

It is important to launch all jobs with ``&`` after the srun command to ensure they do not block each other. 

.. Note::
   Make sure to change the reward model arg ``trainer.reinforce.inference_micro_batch_size`` such that ``trainer.reinforce.inference_micro_batch_size * DP size <= model.reinforce.rollout_micro_batch_size``.

REINFORCE Results
%%%%%%%%%%%%%%%%%%%%%%%%%%

After you've completed reinforce training, you can serve your model using the `megatron_gpt_eval.py <https://github.com/NVIDIA/NeMo/blob/8cd5f1c8e7d4fed9f4f946028cd02047c5d2296f/examples/nlp/language_modeling/megatron_gpt_eval.py#L4>`__ script from the NeMo codebase to run more rigorous evaluation of your trained model.