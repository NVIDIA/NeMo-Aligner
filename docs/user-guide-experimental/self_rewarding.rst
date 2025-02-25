.. include:: /content/nemo.rsts

Model Alignment by Self-Rewarding Language Models
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Original paper: https://arxiv.org/abs/2401.10020
Meta Self-Rewarding paper: https://arxiv.org/abs/2407.19594

The NeMo framework supports efficient model alignment via the NeMo Aligner codebase.

All algorithms in NeMo Aligner are compatible with any GPT-based model from Megatron Core (i.e., those with mcore_gpt=True in the configuration). For this tutorial, we will demonstrate the entire self-rewarding pipeline using a 2B GPT model with 4096 sequence length <https://huggingface.co/nvidia/GPT-2B-001>__. This tutorial is also applicable to other GPT models, such as Llama models, regardless of their size.

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

               python /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py \
                   --input_name_or_path /path/to/llama --output_path /output_path/mcore_gpt.nemo

After these steps you should have a file ``mcore_gpt.nemo`` to use in NeMo-Aligner.

.. note::
   Mcore models use TransformerEngine as a backend, and it tries to find efficient kernels. But depending on the GPU you have it may not find them. If you ever face errors that relate to kernel finding set these variables on top of your script.

   .. code-block:: bash

      export NVTE_MASKED_SOFTMAX_FUSION=0
      export NVTE_FLASH_ATTN=0
      export NVTE_FUSED_ATTN=0

Additionally, TransformerEngine is non-deterministic by default, meaning subsequent runs of Self-Rewarding using identical parameters will produce different results, which is not ideal for parameter perturbation.
Helpfully, TransformerEngine exposes a flag to set if you want to guarantee deterministic training runs:

.. code-block:: bash

   export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
   export NVTE_MASKED_SOFTMAX_FUSION=0

SFT vs Foundational (base) model for Self-Rewarding Training
############################################################
Self-Rewarding can be run on either base/foundational models, that is, models which have only been trained on autoregressive language prediction tasks and not on instruction following tasks,
or, you can also run Self-Rewarding on models which have been SFTed on instruction-based datasets as well, similar to DPO/PPO. Either type of model will work well with Self-Rewarding. If you would like to start with a supervised fine tuned model instead of a base model, please see our full guide on how to perform SFT on a Megatron GPT model :ref:`SFT guide <sft>`.

Self-Rewarding Model Training
#############################

Self-Rewarding training uses the exact same dataset formatting and files as the NeMo-Aligner SFT trainer. Please see the data formatting section of SFT to understand the data format necessary for Self-Rewarding :ref:`SFT guide <sft>`

Once your data is processed into the correct format you are ready to begin Self-Rewarding training. You must start with a pretrained or SFT trained model. For this section we will use the SFT model trained in the previous step to train the Self-Rewarding model.
For the purposes of the following sections, we'll assume your training jsonl file is located in ``/path/to/train_sft_format.jsonl`` and your validation jsonl file is located in ``/path/to/valid_sft_format.jsonl``.

Due to some limitations of the Nemo Aligner system and reusing code files, the parameters for Self-Rewarding share the same parameter namespace as SPIN, so these parameters are labelled as ``spin``, but they apply to the self-rewarding algorithm.

For the parameters below, the ``model.spin.ref_policy_kl_penalty`` corresponds to the beta parameter in the Self-Rewarding paper, and ``trainer.self_rewarding.max_iterations`` corresponds to number of iterations.

Self-Rewarding is a very generation-heavy algorithm, with N*k generations per sample in the training data. As such, it is highly advisable to enable TRTLLM in order to vastly speedup training generation times (5-7X speedup).
You can enable TRT by setting ``trainer.self_rewarding.trt_llm.enable=true`` along with ``trainer.self_rewarding.trt_llm.model_type``. Set this parameter to ``gptnext`` for Nemotron models and ``llama`` for the Llama family of models.
If you want to train using Meta-Rewarding instead of the original Self-Rewarding, you need to set ``model.spin.use_meta_judge=true``. When using meta mode, you also need to set ``model.spin.meta_judge_pcnt`` which controls the maximum percent of any GBS which can be populated by meta-judge training samples.
If you want to use Length Control (Meta-Self-Rewarding paper, section 2.1, last paragraph), you can set that with ``model.spin.length_control``. This parameter accepts either a scalar or a list of size == number of iterations, where
each iteration will apply its corresponding length control value. This allows you to create a schedule of different length control values for each iteration. This logic will work for both Self-Rewarding and Meta Self-Rewarding.
You can also control which variant of DPO loss is used for training using the ``model.spin.preference_loss`` parameter. Valid entries are: ``dpo``, ``scale``, ``rpo_bwd_kl``, ``rpo_fwd_kl``, ``ipo``, and ``rpo_sq``. Default is ``dpo``.


.. tab-set::

    .. tab-item:: Terminal
        :sync: key3

         To run Self-Rewarding model training on the terminal directly:

         .. code-block:: bash 

            export GPFS="/path/to/nemo-aligner-repo"
            export TRAIN_DATA_PATH="/path/to/train_sft_format.jsonl"
            export VALID_DATA_PATH="/path/to/valid_sft_format.jsonl"

            python -u ${GPFS}/examples/nlp/gpt/train_gpt_self_rewarding.py \
               trainer.num_nodes=1 \
               trainer.devices=8 \
               model.micro_batch_size=1 \
               model.global_batch_size=64 \
               pretrained_checkpoint.restore_from_path=/path/to/megatron_gpt_sft.nemo \
               "model.data.train_ds.file_path=${TRAIN_DATA_PATH}" \
               "model.data.validation_ds.file_path=${VALID_DATA_PATH}" \
               exp_manager.create_wandb_logger=false \
               exp_manager.wandb_logger_kwargs.project=self_rewarding_training \
               exp_manager.wandb_logger_kwargs.name=self_rewarding_training \
               exp_manager.explicit_log_dir=/results \
               ++model.sequence_parallel=false \
               ++model.apply_rope_fusion=false \
               trainer.self_rewarding.max_iterations=3 \
               trainer.self_rewarding.max_epochs=1 \
               model.spin.ref_policy_kl_penalty=0.1 \
               model.spin.use_meta_judge=false \
               model.spin.length_params.max_length=2048 \
               model.data.train_ds.max_seq_length=4096

    .. tab-item:: Slurm
        :sync: key4

         To run Self-Rewarding model training with Slurm, use the script below. The script uses 4 nodes, but you can change the node count to something different:

         .. code-block:: bash 

            #!/bin/bash
            #SBATCH -A <<ACCOUNT NAME>>
            #SBATCH -p <<PARTITION NAME>>
            #SBATCH -N 4
            #SBATCH -t 4:00:00
            #SBATCH -J <<JOB NAME>>
            #SBATCH --ntasks-per-node=8
            #SBATCH --gpus-per-node 8
            #SBATCH --exclusive
            #SBATCH --overcommit

            GPFS="/path/to/nemo-aligner-repo"
            PRETRAINED_CHECKPOINT_NEMO_FILE="/path/to/megatron_gpt_sft.nemo"

            TRAIN_DATA_PATH="/path/to/train_sft_format.jsonl"
            VALID_DATA_PATH="/path/to/valid_sft_format.jsonl"

            PROJECT="<<WANDB PROJECT>>"

            CONTAINER=<<<CONTAINER>>> # use the latest NeMo Training container, Aligner will work there
            MOUNTS="--container-mounts=${GPFS}:${GPFS},${TRAIN_DATA_PATH}:${TRAIN_DATA_PATH},${VALID_DATA_PATH}:${VALID_DATA_PATH},${PRETRAINED_CHECKPOINT_NEMO_FILE}:${PRETRAINED_CHECKPOINT_NEMO_FILE}"

            RESULTS_DIR="/path/to/result_dir"

            OUTFILE="${RESULTS_DIR}/rm-%j_%t.out"
            ERRFILE="${RESULTS_DIR}/rm-%j_%t.err"
            mkdir -p ${RESULTS_DIR}

            read -r -d '' cmd <<EOF
            echo "*******STARTING********" \
            && echo "---------------" \
            && echo "Starting training" \
            && cd ${GPFS} \
            && export PYTHONPATH="${GPFS}:${PYTHONPATH}" \
            && export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
            && export NVTE_MASKED_SOFTMAX_FUSION=0 \
            && export HYDRA_FULL_ERROR=1 \
            && python -u ${GPFS}/examples/nlp/gpt/train_gpt_self_rewarding.py \
               trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
               trainer.devices=8 \
               pretrained_checkpoint.restore_from_path='${PRETRAINED_CHECKPOINT_NEMO_FILE}' \
               "model.data.train_ds.file_path=${TRAIN_DATA_PATH}" \
               "model.data.validation_ds.file_path=${VALID_DATA_PATH}" \
               model.micro_batch_size=1 \
               model.global_batch_size=64 \
               exp_manager.explicit_log_dir=${RESULTS_DIR} \
               exp_manager.create_wandb_logger=True \
               exp_manager.wandb_logger_kwargs.name=${NAME} \
               exp_manager.wandb_logger_kwargs.project=${PROJECT} \
               trainer.self_rewarding.max_iterations=3 \
               trainer.self_rewarding.max_epochs=1 \
               model.spin.ref_policy_kl_penalty=0.1 \
               model.spin.use_meta_judge=false \
               model.spin.length_params.max_length=2048 \
               model.data.train_ds.max_seq_length=4096
            EOF

            srun -o $OUTFILE -e $ERRFILE --container-image=$CONTAINER $MOUNTS bash -c "${cmd}"
            set +x

During Self-Rewarding training, there will be several metrics recorded to WandB which you can monitor, the following of which are specific to Self-Rewarding:

- chosen_lengths: average token length of chosen responses (average taken across GBS)
- reject_lengths: as above but for rejected responses
- chosen_generated_rewards: the average reward (across GBS) generated by the LLM-as-a-judge for chosen responses
- rejected_generated_rewards: as above but for rejected responses
- rewards_chosen_mean: see below for a definition of what reward means in this context
- rewards_rejected_mean: as above but for rejected responses
- bad_samples_per_GBS: the percentage of samples in a GBS which are excluded from training because of bad output from the LLM-as-a-judge (could be caused by parse errors, or all responses being judge with the same score, etc)
- bad_ends_per_GBS: only valid if using TRT, this tracks the percentage of each GBS where TRT generates incorrect stop tokens (should be really low, < 1%)
- preference_loss: the raw DPO variant loss
- sft_loss: if adding an SFT loss (categorical cross-entropy loss) for the chosen response, then you can see that raw loss here

The ``reward`` in this case is calculated as the difference between model log probs and the reference log probs, multiplied by the KL penalty (beta in the original paper), for the ground truth and generated responses.
During training, the acc should generally be increasing, but don't worry if its absolute value remains low, as it doesn't correlate to finalised MTBench or MMLU scores. It should just be generally increasing.
All metrics will be grouped by either ``train/`` or ``val/`` in WandB, representing whether that metric is from the training or validation set, respectively.
You can also see a table which will print out the prompt, chosen response, and rejected response for each validation step. This allows you to keep track of response quality and hallucinations.

When it comes to ideal hyperparameters for Self-Rewarding training, much will depend on the characteristics of your SFT (or base/foundational) model and your training data, so there is no one-size-fits-all parameter set which will work in all cases.
Additionally, Self-Rewarding (with or without meta) is a complex algorithm with a lot of moving pieces and a lot of parameters, so finding what works well for your model and data can be difficult.
Below are some of observations from the Nvidia Alignment team as to what parameters we have seen work well:

* global_batch_size: we recommend using 64, and going up to 128 only for large models (70B+) that are also training with large datasets
* iterations/epochs: the original paper uses 3 iterations with 1 epoch per iteration, and we find this to be sufficient for most use cases
* learning rate: for SFT/aligned models, we recommend a smaller LR, between 3e-7 and 1e-7. If training a foundational model, then something between 3e-6 to 9e-7.
* ref_policy_kl_penalty: we did not see large changes from perturbations to this value; we recommend 0.1 - 0.001
* length_control: depends very much on model size and data, but we found good results with [0,0,0.1]
* use_meta_judge: we have found stronger results when settings this to true, which is in line with the paper's results
* meta_judge_pcnt: we recommend you do not set this higher than 0.15 (15%). Any higher, and we have observed that the llm-as-a-judge model starts to output identical scores for every response (always a 5)
