.. include:: /content/nemo.rsts

Model Alignment by RLHF
@@@@@@@@@@@@@@@@@@@@@@@

The NeMo framework supports efficient model alignment via the NeMo Aligner codebase.

All algorithms in NeMo Aligner will work with any GPT based model that is from mcore(i.e in the config it has ``mcore_gpt=True``). For the purposes of this tutorial, we will go through the entire RLHF pipeline using the newly released `2B GPT model with 4096 sequence length <https://huggingface.co/nvidia/GPT-2B-001>`__.  The same tutorial also works for GPT models(such as LLaMa2) of any size.

Obtaining a pretrained model
############################
To start, we must first get a pretrained model to align. There are 2 models we recommend to get started. The rest of the tutorial will work with either model, but for demonstration purposes we will use the smaller 2B model. 

.. tab-set::

    .. tab-item:: 2B GPT
        :sync: key1

        #. Get the 2B checkpoint via ``wget https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo``
        #. Extract the NeMo File to a folder with ``mkdir model_checkpoint && tar -xvf GPT-2B-001_bf16_tp1.nemo -C model_checkpoint``
        #. And then run the script to convert from old NeMo checkpoint to Megatron-Core checkpoint. The script is located `here <https://github.com/NVIDIA/NeMo/blob/86b198ff93438d454f9c7f3550bcfb7d4e59feab/scripts/nlp_language_modeling/convert_nemo_gpt_to_mcore.py>`__.
            .. code-block:: bash 

               python convert_nemo_gpt_to_mcore.py \
                  --in-folder ./model_checkpoint \
                  --out-file ./mcore_gpt.nemo

    .. tab-item:: LLaMa2 7B
        :sync: key2

        #. Download the `Llama 2 7B LLM model and tokenizer <https://huggingface.co/meta-llama/Llama-2-7b>`__ into the models folder.
        #. Convert the LLaMa2 LLM into ``.nemo`` format
            .. code-block:: bash 

               python NeMo/scripts/nlp_language_modeling/convert_hf_llama_to_nemo.py \
                   --in-file /path/to/llama --out-file /output_path/mcore_gpt.nemo

After these steps you should have a file ``mcore_gpt.nemo`` to use in NeMo-Aligner.

.. note::
   Mcore models use Transformer engine as a backend, and it tries to find efficient kernels. But depending on the GPU you have it may not find them. If you ever face errors that relate to kernel finding set these variables on top of your script.

   .. code-block:: bash

      export NVTE_MASKED_SOFTMAX_FUSION=0
      export NVTE_FLASH_ATTN=0
      export NVTE_FUSED_ATTN=0

.. _sft:

Instruction Following Taught by Supervised Fine-Tuning (SFT)
############################################################
For best RLHF training performance, it is recommended to start with a supervised fine tuned model rather than the base model. In this section, we will describe how to do SFT using NeMo-Aligner.

**Supervised Fine-Tuning** (SFT) is the process of fine-tuning all of a model's parameters on supervised data of inputs and outputs. It teaches the model how to follow user specified instructions. It is typically done after model pre-training. This section describes the steps involved in fine-tuning a GPT model for instruction following. Subsequent sections describe how to format your data and run training.

SFT Data Formatting
%%%%%%%%%%%%%%%%%%%

This section uses the `Dolly dataset <https://github.com/databrickslabs/dolly>`__ as an example to demonstrate how to format your SFT data. This dataset consists of 15,000 instruction-context-response triples.

First, to download the data enter the following command::

    wget https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl

The downloaded data, stored at ``databricks-dolly-15k.jsonl``, is a JSONL file with each line formatted like this::

   {
       "instruction": "When did Virgin Australia start operating?",
       "context": "Virgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based airline. It is the largest airline by fleet size to use the Virgin brand. It commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.[3] It suddenly found itself as a major airline in Australia's domestic market after the collapse of Ansett Australia in September 2001. The airline has since grown to directly serve 32 cities in Australia, from hubs in Brisbane, Melbourne and Sydney.[4]",
       "response": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.",
       "category": "closed_qa"
   }

As this example shows, there are no clear "input" and "output" fields that SFT requires.

For an example of how to process this data format into a JSONL file with "input" and "output" fields, see `preprocess.py <https://github.com/NVIDIA/NeMo-Megatron-Launcher/blob/8779c6ef3d8f90356aefdab3ce69d5262d660d6f/launcher_scripts/nemo_launcher/collections/dataprep_scripts/dolly_dataprep/preprocess.py>`__::

   python preprocess.py --input databricks-dolly-15k.jsonl
   
This script converts the *Instruction*, *Context*, and *Response* fields into *Input* and *Output*. It also concatenates the *Instruction* and *Context* fields with a ``\n\n`` separator and randomizes the order in which they appear in the input to generate a new JSONL file. This generates an output file called `databricks-dolly-15k-output.jsonl`.

Now that we have the data we will use NeMo-Aligner to do the supervised fine tuning.

.. tab-set::

    .. tab-item:: Terminal
        :sync: key3

         To run SFT on the terminal directly. Note that the working directory must be at the NeMo Aligner repo for this to work.

         .. code-block:: bash 

            python examples/nlp/gpt/train_gpt_sft.py \
               trainer.precision=bf16 \
               trainer.num_nodes=1 \
               trainer.devices=8 \
               trainer.sft.max_steps=-1 \
               trainer.sft.limit_val_batches=40 \
               trainer.sft.val_check_interval=1000 \
               model.megatron_amp_O2=True \
               model.restore_from_path=/path/to/your/mcore_gpt.nemo \
               model.optim.lr=5e-6 \
               model.answer_only_loss=True \
               model.data.num_workers=0 \
               model.data.train_ds.micro_batch_size=1 \
               model.data.train_ds.global_batch_size=128 \
               model.data.train_ds.file_path=/path/to/databricks-dolly-15k.jsonl \
               model.data.validation_ds.micro_batch_size=1 \
               model.data.validation_ds.global_batch_size=128 \
               model.data.validation_ds.file_path=/path/to/databricks-dolly-15k.jsonl \
               exp_manager.create_wandb_logger=True \
               exp_manager.explicit_log_dir=/results \
               exp_manager.wandb_logger_kwargs.project=sft_run \
               exp_manager.wandb_logger_kwargs.name=dolly_sft_run \
               exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
               exp_manager.resume_if_exists=True \
               exp_manager.resume_ignore_no_checkpoint=True \
               exp_manager.create_checkpoint_callback=True \
               exp_manager.checkpoint_callback_params.monitor=validation_loss

    .. tab-item:: Slurm
        :sync: key4

         To run SFT via slurm do the following:

         .. code-block:: bash 

            #!/bin/bash
            #SBATCH -A <<YOUR ACCOUNT>>
            #SBATCH -p <<<YOUR PARTITION>>>
            #SBATCH -N 1
            #SBATCH -t 4:00:00
            #SBATCH -J <<<JOB NAME>>>
            #SBATCH --ntasks-per-node=8
            #SBATCH --exclusive
            #SBATCH --overcommit

            GPFS="/path/to/nemo-aligner-repo"

            TRAIN_DATA_PATH="/path/to/databricks-dolly-15k.jsonl"
            VALID_DATA_PATH="/path/to/databricks-dolly-15k.jsonl"

            PRETRAINED_ACTOR_NEMO_FILE="/path/to/your/mcore_gpt.nemo"

            PROJECT=WANDB_PROJECT # if you want to use wandb

            RESULTS_DIR="/path/to/result_dir"

            OUTFILE="${RESULTS_DIR}/sft-%j_%t.out"
            ERRFILE="${RESULTS_DIR}/sft-%j_%t.err"
            mkdir -p ${RESULTS_DIR}

            CONTAINER=<<<CONTAINER>>> # use the latest NeMo Training container, Aligner will work there

            MOUNTS="--container-mounts=MOUNTS" # mounts

            read -r -d '' cmd <<EOF
            echo "*******STARTING********" \
            && echo "---------------" \
            && echo "Starting training" \
            && cd ${GPFS} \
            && export PYTHONPATH="${GPFS}:${PYTHONPATH}" \
            && export HYDRA_FULL_ERROR=1 \
            && python -u ${GPFS}/examples/nlp/gpt/train_gpt_sft.py \
               trainer.precision=bf16 \
               trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
               trainer.devices=8 \
               trainer.sft.max_steps=-1 \
               trainer.sft.limit_val_batches=40 \
               trainer.sft.val_check_interval=1000 \
               trainer.sft.save_interval=50 \
               model.megatron_amp_O2=True \
               model.restore_from_path=${PRETRAINED_ACTOR_NEMO_FILE} \
               model.optim.lr=5e-6 \
               model.answer_only_loss=True \
               model.data.num_workers=0 \
               model.data.train_ds.micro_batch_size=1 \
               model.data.train_ds.global_batch_size=128 \
               model.data.train_ds.file_path=${TRAIN_DATA_PATH} \
               model.data.validation_ds.micro_batch_size=1 \
               model.data.validation_ds.global_batch_size=128 \
               model.data.validation_ds.file_path=${VALID_DATA_PATH} \
               exp_manager.create_wandb_logger=True \
               exp_manager.explicit_log_dir=${RESULTS_DIR} \
               exp_manager.wandb_logger_kwargs.project=${PROJECT} \
               exp_manager.wandb_logger_kwargs.name=dolly_sft_run \
               exp_manager.resume_if_exists=True \
               exp_manager.resume_ignore_no_checkpoint=True \
               exp_manager.create_checkpoint_callback=True \
               exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
               exp_manager.checkpoint_callback_params.monitor=validation_loss
            EOF

            srun -o $OUTFILE -e $ERRFILE --container-image=$CONTAINER $MOUNTS bash -c "${cmd}"
            set +x

If you have a smaller or bigger machine make sure to change the ``trainer.num_nodes`` and ``trainer.devices`` correspondingly. This allows scaling to thousands of GPUs.

For this particular run on 2B model, the final train loss is around 1.536. When the training finishes there will be a file called ``megatron_gpt_sft.nemo`` for use in the RLHf process.

.. note::
   NeMo FW has support for wandb logging. To get started with wandb please see their `quick start guide <https://docs.wandb.ai/quickstart>`__. You can turn on wandb logging with ``exp_manager.create_wandb_logger=True`` and it will log the job results to wandb.

   For the slurm scripts we provide, they all use the `pyxis <https://github.com/NVIDIA/pyxis>`__ slurm extension which require ``--container-image=`` ``--container-mounts=`` to be provided. It is not necessary to use this extension, NeMo Aligner will work on regular python environments as well.

Now that we have obtained a SFT model, we will use this to start the RLHF process. We will use the `PPO <https://arxiv.org/abs/1707.06347>`__ algorithm for the reinforcement learning on the `Anthropic-HH-RLHF <https://huggingface.co/datasets/Anthropic/hh-rlhf>`__ dataset.

Data processing for RLHF
#########################

We have a script ready to use for processing the Anthropic-HH dataset into a jsonlines format. Run the following command on the `download_and_process.py <https://github.com/NVIDIA/NeMo-Megatron-Launcher/blob/8d4f34c9da6b3254ca316b2c43ee88b77a894529/launcher_scripts/nemo_launcher/collections/dataprep_scripts/anthropichh_dataprep/download_and_process.py#L1>`__ script for anthropic HH.

   .. code-block:: bash 

      python download_and_process.py

After running this script you should have the files ``{train,test}_comparisons.jsonl`` and ``{train,test}_prompts.jsonl``. Comparison files are used for reward model training, whereas prompts file are used for the reinforcement learning training.

Reward Model Training
#####################

The reward model is used to score how good a response is. It is trained using a pairwise comparison loss and therefore requires a dataset of response pairs, where one response in the pair is ranked higher than the other. A good reward model is cruical for the success of the PPO training.

Data Preprocessing
%%%%%%%%%%%%%%%%%%

You can also bring your own data for the reward model training phase. The reward model datasets require the following format::

   {"text": prompt1 || good_response_1}
   {"text": prompt1 || bad_response_1}
   {"text": prompt2 || good_response_2}
   {"text": prompt2 || bad_response_2}
   ...

where || denotes string concatenation and *prompt1* and *prompt2* are different prompts. Note that for the same prompt, *prompt || good_response* must come before *prompt || bad_response* in the dataset.

An example JSONL file can look like the following::

  {"text": User: When did Virgin Australia start operating?\nAssistant: 31 August 2000}
  {"text": User: When did Virgin Australia start operating?\nAssistant: I refuse to answer this question.}
  {"text": User: What is 6*10?\nAssistant: 60}
  {"text": User: What is 6*10?\nAssistant: 90}
  ...


To launch reward model training, you must start with a pretrained or SFT trained model. For this section we will use the SFT model trained in the previous step to train the reward model.

.. tab-set::

    .. tab-item:: Terminal
        :sync: key3

         To run reward model training on the terminal directly

         .. code-block:: bash 

            GPFS="/path/to/nemo-aligner-repo"
            TRAIN_DATA_PATH="/path/to/train_comparisons.jsonl"
            VALID_DATA_PATH="/path/to/test_comparisons.jsonl"

            GPFS="/path/to/nemo-aligner-repo"

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

         To run reward model training using Slurm. The script below uses 4 nodes, but you can change the node count to something smaller.

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


*Remark: currently, the example training script does not automatically run evaluation on the provided test set. This may change in a future release.* 

A good reward model training will have validation accuracy improve as the training goes on. In the above slurm example we achieve 69.57% validation accuracy. 

With the finished training NeMo-Aligner will save a ``megatron_gpt.nemo`` which is the reward model we need for the RL stage.

PPO Training
############

After you have fine-tuned a GPT model using Supervised Fine-Tuning (SFT), and trained a reward model as explained in the preceding section, you can start doing RLHF with PPO.

During PPO training, we conceptually have 4 models interacting with each other:

#. The PPO Actor Network (also known as the Policy Network): This is the model we are training, and it should start from an SFT model.
#. The Reward Model (RM) Network (also known as a Preference Model (PM)): This model takes a prompt concatenated with a response as input, and outputs a single scalar value: the reward, which the PPO algorithm will try to maximize.
#. The PPO Critic Network (also known as the Value Network): Since PPO is an Actor-Critic algorithm, we need a Critic to guide the Actor during training. The Critic will provide value estimates for each token in the responses provided by the Actor. These values can be seen as an estimate of the total reward the Actor will receive after generating all the remaining tokens. The Critic should be initialized from the RM so as to provide useful feedback in the early stages of training. Note: The RM generates a single reward for the entire sequence, whereas the Critic generates a value for each token.
#. The Initial Policy Network (also known as the Reference Model): We use this model to compute a KL Divergence penalty term that ensures that the PPO Actor does not diverge too much from the Initial Policy. This way, we prevent the PPO Actor from overfitting to the rewards given by the RM, and ensure it does not forget the knowledge it acquired during pretraining and SFT. This model should be the one used to initialize the PPO Actor Network.

In the most optimized configuration, Aligner will run the actor and initial policy within the same job and the critic and reward model within the same job. It will then use cpu offloading to load back the corresponding model when needed.
   
The next section discusses how to launch each of these two jobs.

Launching the Reward Model and Critic Server
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

To launch the server:

.. code-block:: bash 

   #!/bin/bash
   CHECKPOINT_NEMO_FILE="/path/to/trained_rm.nemo"
   GPFS="/path/to/nemo-aligner-repo"

   RESULTS_DIR="critic_results_dir"

   export PYTHONPATH="${GPFS}:${PYTHONPATH}" \
   && export HYDRA_FULL_ERROR=1 \
   && python -u ${GPFS}/examples/nlp/gpt/serve_ppo_critic.py \
      trainer.devices=1 \
      trainer.num_nodes=8 \
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

The above example launches the reward model critic server on 8 gpus and 1 node. Please make sure to change ``trainer.devices``, ``trainer.num_nodes`` depending on your model size and scale. Aligner will work on any scale. Also make sure to tune the `trainer.ppo.inference_micro_batch_size` argument, this sets how big of a batch the PPO actor is allowed to send to the critic per DP rank.

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

The above launches the initial and actor server on 1 node with 8 GPUs

.. note::
   Fore more info on PPO hyperparameters please see `PPO Hparams <https://github.com/NVIDIA/NeMo-Aligner/blob/main/docs/RLHFTraining.md#ppo-hyperparameters>`__.

Launching Both Servers for RLHF training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

You can use slurm to launch the 2 jobs and get them to coordinate together in a full RLHF job via the following:

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

The above script runs the reward model critic server on 1 node and the actor on 1 node.

It is important to launch all jobs with ``&`` after the srun command, to ensure they do not block each other. 

.. note::
   Make sure to change the critic arg ``trainer.ppo.inference_micro_batch_size`` such that ``trainer.ppo.inference_micro_batch_size * DP size <= model.ppo.rollout_micro_batch_size``.

PPO Results
%%%%%%%%%%%

Once you've completed RLHF training, you can serve your model using the `megatron_gpt_eval.py <https://github.com/NVIDIA/NeMo/blob/8cd5f1c8e7d4fed9f4f946028cd02047c5d2296f/examples/nlp/language_modeling/megatron_gpt_eval.py#L4>`__ script from the NeMo codebase to run more rigorous evaluation of your trained model.
   
Scaling the tutorial to bigger models
#####################################

The above tutorial is a way to get started with RLHF but is not the most optimal performant or convergence configuration. When running RLHF fully, we expect around +0.4 to +0.5 on the MT-bench score. It is cruical to start with a good SFT model and monitor the response length.
