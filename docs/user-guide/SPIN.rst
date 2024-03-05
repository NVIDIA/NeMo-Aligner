.. include:: /content/nemo.rsts

Model Alignment by Self-Play Fine-Tuning (SPIN)
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Original paper: https://arxiv.org/abs/2401.01335

The NeMo framework supports efficient model alignment via the NeMo Aligner codebase.

All algorithms in NeMo Aligner will work with any GPT based model that is from mcore(i.e in the config it has ``mcore_gpt=True``). For the purposes of this tutorial, we will go through the entire SPIN pipeline using the newly released `2B GPT model with 4096 sequence length <https://huggingface.co/nvidia/GPT-2B-001>`__.  This same tutorial also works for GPT models(such as LLaMa2) of any size.

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
   Mcore models use TransformerEngine as a backend, and it tries to find efficient kernels. But depending on the GPU you have it may not find them. If you ever face errors that relate to kernel finding set these variables on top of your script.

   .. code-block:: bash

      export NVTE_MASKED_SOFTMAX_FUSION=0
      export NVTE_FLASH_ATTN=0
      export NVTE_FUSED_ATTN=0

Additionally, TransformerEngine is non-deterministic by default, meaning subsequent runs of SPIN using identical parameters will produce different results, which is not ideal for parameter perturbation.
Helpfully, TransformerEngine exposes a flag to set if you want to guarantee deterministic training runs:

.. code-block:: bash

   export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

SFT vs Foundational (base) model for SPIN Training
##################################################
Unlike DPO and PPO, SPIN was designed to be run on foundational (base) models, that is, models which have only been trained on autoregressive language prediction tasks and not on instruction following tasks.
However, you can also run SPIN on models which have been SFTed on instruction-based datasets as well, similar to DPO/PPO. Either type of model will work well with SPIN. If you would like to start with a supervised fine tuned model instead of a base model, please see our full guide on how to perform SFT on a Megatron GPT model :ref:`SFT guide <sft>`.

SPIN Model Training
###################

SPIN training uses the exact same dataset formatting and files as the NeMo-Aligner SFT trainer. Please see the data formatting section of SFT to understand the data format necessary for SPIN :ref:`SFT guide <sft>`

Once your data is processed into the correct format you are ready to begin SPIN training. You must start with a pretrained or SFT trained model. For this section we will use the SFT model trained in the previous step to train the SPIN model.
For the purposes of the following sections, we'll assume your training jsonl file is located in ``/path/to/train_spin_format.jsonl`` and your validation jsonl file is located in ``/path/to/valid_spin_format.jsonl``.

For the below parameters, the ``model.spin.ref_policy_kl_penalty`` corresponds to the beta parameter in the SPIN paper, and ``trainer.spin.max_iterations`` corresponds to T (with ``trainer.spin.max_epochs`` epochs per iteration)

.. tab-set::

    .. tab-item:: Terminal
        :sync: key3

         To run SPIN model training on the terminal directly

         .. code-block:: bash 

            export GPFS="/path/to/nemo-aligner-repo"
            export TRAIN_DATA_PATH="/path/to/train_spin_format.jsonl"
            export VALID_DATA_PATH="/path/to/valid_spin_format.jsonl"

            python -u ${GPFS}/examples/nlp/gpt/train_gpt_spin.py \
               trainer.num_nodes=1 \
               trainer.devices=8 \
               model.micro_batch_size=1 \
               model.global_batch_size=64 \
               pretrained_checkpoint.restore_from_path=/path/to/megatron_gpt_sft.nemo \
               "model.data.train_ds.file_path=${TRAIN_DATA_PATH}" \
               "model.data.validation_ds.file_path=${VALID_DATA_PATH}" \
               exp_manager.create_wandb_logger=false \
               exp_manager.wandb_logger_kwargs.project=spin_training \
               exp_manager.wandb_logger_kwargs.name=spin_training \
               exp_manager.explicit_log_dir=/results \
               trainer.spin.max_iterations=1 \
               trainer.spin.max_epochs=1 \
               model.spin.ref_policy_kl_penalty=0.1 \
               model.spin.length_params.max_length=2048 \
               model.data.train_ds.max_seq_length=4096

    .. tab-item:: Slurm
        :sync: key4

         To run SPIN model training using Slurm. The script below uses 4 nodes, but you can change the node count to something different.

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

            TRAIN_DATA_PATH="/path/to/train_spin_format.jsonl"
            VALID_DATA_PATH="/path/to/valid_spin_format.jsonl"

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
            && export HYDRA_FULL_ERROR=1 \
            && python -u ${GPFS}/examples/nlp/gpt/train_gpt_spin.py \
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
               trainer.spin.max_iterations=1 \
               trainer.spin.max_epochs=1 \
               model.spin.ref_policy_kl_penalty=0.1 \
               model.spin.length_params.max_length=2048 \
               model.data.train_ds.max_seq_length=4096
            EOF

            srun -o $OUTFILE -e $ERRFILE --container-image=$CONTAINER $MOUNTS bash -c "${cmd}"
            set +x

During SPIN training, there will be several metrics recorded to WandB which you can monitor, chiefly acc (representing the percentage amount whereby the model's implicit reward for the ground truth response is greater than for the response generated by the reference policy).
The ``reward`` in this case is calculated as the difference between model log probs and the reference log probs, multiplied by the KL penalty (beta in the original paper), for the ground truth and generated responses.
During training, the acc should generally be increasing, but don't worry if its absolute value remains low, as it doesn't correlate to finalised MTBench or MMLU scores. It should just be generally increasing.
Other metrics to keep an eye on are the rewards_actual_mean and rewards_generated_mean, which represent the average of the ``rewards`` as defined above. Again, the absolute values aren't necessarily so important as the fact that the actual_mean should be greater than the generated_mean over time, and the greater that difference, the better.
All metrics will be grouped by either ``train/`` or ``val/`` in WandB, representing whether that metric is from the training or validation set, respectively.
NOTE: for validation we only calculate a vanilla SFT negative log-likelihood loss instead of using the formal SPIN loss, so for validation metrics there will only be the SFT NLL loss. We do this to speed up the validation aspect of training, as doing SPIN generation is time consuming, and not really necessary for validation.

When it comes to ideal hyperparameters for SPIN training, much will depend on the characteristics of your SFT (or base/foundational) model and your training data, so there is no one-size-fits-all parameter set which will work in all cases.
However, the following is a brief overview of which hyperparameters we have perturbed for various model sizes and their effects:

* global_batch_size: the SPIN paper recommends 64 for a 7B model, which we have found holds true, in that higher GBS for 7B models performs much worse. For larger models, you can increase to 128 or 256 as needed, but we recommend you start with 64 as a baseline
* iterations/epochs: the SPIN paper used iterations=3 and epochs=2 for their training on a 7B model with a training dataset size of 200k. Using the same foundational model as the authors, we found better results with iterations=1, epochs=1 using a 50k subset of their 200k data. We therefore recommend starting with iterations=1, and increasing to 2 as needed by testing on MT-Bench/MMLU.
                     additionally, unlike the SPIN paper, our implementation does not currently inject the generated samples from iteration t-1 into t, and this may be a reason why we do not see any performance increases with iterations > 1.
* learning rate: the SPIN paper recommends starting with 5e-7 and annealing down to 1e-7 for the final iteration. We found that this generally works well, however, we also saw good resutls from a constant learning rate of 4e-7 or 3e-7.
* ref_policy_kl_penalty: this is an area of ongoing research. The SPIN paper recommends startings at 0.1 and increasing up to 5.0 for the final iteration. We find that a beta of 0.1 works well for the first iteration, but subsequent iterations tend to overfit quickly, which raising the KL penalty seems to help with, but not enough that T > 1 checkpoints perform better than T <= 1. For now, we recommend leaving KL at 0.1 and training for a single iteration only.
