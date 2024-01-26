.. include:: /content/nemo.rsts

Model Alignment by Direct Preference Optimisation (DPO)
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

The NeMo framework supports efficient model alignment via the NeMo Aligner codebase.

All algorithms in NeMo Aligner will work with any GPT based model that is from mcore(i.e in the config it has ``mcore_gpt=True``). For the purposes of this tutorial, we will go through the entire DPO pipeline using the newly released `2B GPT model with 4096 sequence length <https://huggingface.co/nvidia/GPT-2B-001>`__.  The same tutorial also works for GPT models(such as LLaMa2) of any size.

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

Additionally, TransformerEngine is non-deterministic by default, meaning subsequent runs of DPO using identical parameters will produce different results, which is not ideal for parameter perturbation.
Helpfully, TransformerEngine exposes a flag to set if you want to guarantee deterministic training runs:

.. code-block:: bash

   export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

Instruction Following Taught by Supervised Fine-Tuning (SFT)
############################################################
For best DPO training performance, it is recommended to start with a supervised fine tuned model rather than the base model. For a full guide on how to perform SFT on a megatron GPT model, please refer to the :ref:`SFT guide <sft>`.

DPO Model Training
#####################

Before running the core DPO training, you must prepare you training and validation data to format required for DPO training. DPO expects .jsonl files where each line is a JSON dict corresponding to a single, complete sample, as shown below::

   {"prompt": "Which year was the Magna Carta signed?", "chosen_response": "1215", "rejected_response": "I refuse to answer this question."}
   {"prompt": "Please give me the name of a famous medieval painter.", "chosen_response": "Hieronymus Bosch", "rejected_response": "David Hockney"}

However, please be aware that most Megatron GPT models adhere to a strict formatting template which needs to be followed, and will depend on the template used during SFT training. For example, many GPT models use the extra_id template, which, when applied, would necessitate that your data looks like this::

   {"prompt": "<extra_id_0>System\n\n<extra_id_1>User\nWhich year was the Magna Carta signed?\n<extra_id_1>Assistant\n", "chosen_response": "1215\n<extra_id_1>", "rejected_response": "I refuse to answer this question.\n<extra_id_1>"}
   {"prompt": "<extra_id_0>System\n\n<extra_id_1>User\nPlease give me the name of a famous medieval painter.\n<extra_id_1>Assistant\n", "chosen_response": "Hieronymus Bosch\n<extra_id_1>", "rejected_response": "David Hockney\n<extra_id_1>"}

Always follow the prompt-response template format used during your SFT training for DPO, as failure to do so will produce a model which outputs garbage text. You should create one jsonl file in the format above for your training data, and one jsonl for your validation data.

Once your data is processed into the correct format you are ready to begin DPO training. You must start with a pretrained or SFT trained model. For this section we will use the SFT model trained in the previous step to train the DPO model.
For the purposes of the following sections, we'll assuming your training jsonl file is located in ``/path/to/train_dpo_format.jsonl`` and your validation jsonl file is located in ``/path/to/valid_dpo_format.jsonl``.

For the below parameters, the ``model.dpo.ref_policy_kl_penalty`` corresponds to the beta parameter in the DPO paper.

.. tab-set::

    .. tab-item:: Terminal
        :sync: key3

         To run DPO model training on the terminal directly

         .. code-block:: bash 

            export GPFS="/path/to/nemo-aligner-repo"
            export TRAIN_DATA_PATH="/path/to/train_dpo_format.jsonl"
            export VALID_DATA_PATH="/path/to/valid_dpo_format.jsonl"

            python -u ${GPFS}/examples/nlp/gpt/train_gpt_dpo.py \
               trainer.num_nodes=1 \
               trainer.devices=8 \
               ++model.micro_batch_size=1 \
               ++model.global_batch_size=512 \
               pretrained_checkpoint.restore_from_path=/path/to/megatron_gpt_sft.nemo \
               "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
               exp_manager.create_wandb_logger=false \
               exp_manager.wandb_logger_kwargs.project=dpo_training \
               exp_manager.wandb_logger_kwargs.name=dpo_training \
               exp_manager.explicit_log_dir=/results \
               ++trainer.dpo.max_epochs=1 \
               ++model.dpo.ref_policy_kl_penalty=0.1

    .. tab-item:: Slurm
        :sync: key4

         To run DPO model training using Slurm. The script below uses 4 nodes, but you can change the node count to something smaller.

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

            TRAIN_DATA_PATH="/path/to/train_comparisons.jsonl"
            VALID_DATA_PATH="/path/to/test_comparisons.jsonl"

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
            && python -u ${GPFS}/examples/nlp/gpt/train_gpt_dpo.py \
               trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
               trainer.devices=8 \
               pretrained_checkpoint.restore_from_path='${PRETRAINED_CHECKPOINT_NEMO_FILE}' \
               "++model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
               ++model.micro_batch_size=1 \
               ++model.global_batch_size=512 \
               exp_manager.explicit_log_dir=${RESULTS_DIR} \
               exp_manager.create_wandb_logger=True \
               exp_manager.wandb_logger_kwargs.name=${NAME} \
               exp_manager.wandb_logger_kwargs.project=${PROJECT} \
               ++trainer.dpo.max_epochs=1 \
               ++model.dpo.ref_policy_kl_penalty=0.1
            EOF

            srun -o $OUTFILE -e $ERRFILE --container-image=$CONTAINER $MOUNTS bash -c "${cmd}"
            set +x

During DPO training, there will be several metrics recorded to WandB which you can monitor, chiefly acc (representing the percentage amount whereby the model's chosen rewards are greater than the rejected rewards).
The ``reward`` in this case is calculated as the difference between model log probs and the reference log probs, multiplied by the KL penalty (beta in the original paper), for the chosen and rejected responses.
During training, the acc should generally be increasing, but don't worry if its absolute value remains low, as it doesn't correlate to finalised MTBench or MMLU scores. It should just be generally increasing.
Other metrics to keep an eye on are the rewards_chosen_mean and rewards_rejected_mean, which represent the average of the ``rewards`` as defined above. Again, the absolute values aren't necessarily so important as the face the the chosen_mean should be greater than the rejected_mean over time, and the greater that difference, the better.
All metrics will be grouped by either ``train/`` or ``val/`` in WandB, representing whether that metric is from the training or validation set, respectively.

When it comes to ideal hyperparameters for DPO training, much will depend on the characteristics of your SFT (or base/foundation) model, so there are no one-size-fits-all parameters which will work in all cases.
However, the following the following is a brief overview of which hyperparameters we have perturbed for various model sizes and their effects:

* global_batch_size: generally, we have found that, all other parameters held equal, lower GBS performs worse. GBS of 256 or 512 seems to be the sweet spot for most models we trained
* epochs: highly sensitive to training data size. We recommend you start with 1 epoch and then add on from there. We did not see any improvements beyond 3 epochs.
* learning rate: we tested cosine annealing with a warmup of 10 steps, followed by a slow decay to a constant rate. That constant rate should be fairly low, we saw best performance with 9e-7 and 5-e7
* ref_policy_kl_penalty: we generally saw better performance with lower values of 0.1, 0.2, 0.5, and 1.0. Occassionally values as high as 5.0 worked too.
