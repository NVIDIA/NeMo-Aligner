.. include:: /content/nemo.rsts

.. _model-aligner-dpo:

Model Alignment by Direct Preference Optimization (DPO)
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

The NeMo Framework supports efficient model alignment via the NeMo-Aligner codebase.

All algorithms in NeMo-Aligner will work with any GPT-based model that is from Megatron Core (in the config it has ``mcore_gpt=True``). For the purposes of this tutorial, we will go through the entire DPO pipeline using the newly released `2B GPT model with 4096 sequence length <https://huggingface.co/nvidia/GPT-2B-001>`__.  The same tutorial also works for GPT models (such as LLaMa2) of any size.

We support both full-parameter DPO training and LoRA DPO training. 
For full-parameter DPO, there exists an actor and a reference model. The actor is initialized with the reference model and is fully trainable. The reference model is frozen and used to calculate logprobs for KL-penalty loss (see `DPO paper <https://arxiv.org/pdf/2305.18290.pdf>`__). 
For LoRA-based DPO, the actor is initialized by the reference model plus LoRA weights, where only the LoRA weights are trainable. Therefore, it allows us to switch between the actor/reference models by simply enabling or disabling LoRA. In addition, there is no need to store two sets of LLM weights.

Besides the vanilla DPO algorithm, we support other variants of DPO algorithms including Identity preference optimization (IPO) and Reward-aware preference optimization (RPO). The algorithm is identified with the ``dpo.preference_loss`` config variable. We support three sorts of RPO algorithms based on the distance metric: ``rpo_sq`` for squared distance; ``rpo_bwd_kl`` for Bernoulli backward KL divergence; ``rpo_fwd_kl`` for Bernoulli forward KL divergence. To use the RPO algorithm, each dataset example should have ``chosen_reward`` and ``rejected_reward``, which might come from Human labelers or reward models. If ``chosen_reward`` and ``rejected_reward`` are not existent in the data, ``dpo.default_chosen_reward`` and ``dpo.default_rejected_reward`` are used.

Obtaining a Pretrained Model
############################
To start, we must first get a pretrained model to align. There are two models we recommend to get started. The rest of the tutorial will work with either model, but for demonstration purposes, we will use the smaller 2B model. 

.. tab-set::

    .. tab-item:: 2B GPT
        :sync: key1

        #. Get the 2B checkpoint via ``wget https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo``.
        #. Extract the NeMo File to a folder with ``mkdir model_checkpoint && tar -xvf GPT-2B-001_bf16_tp1.nemo -C model_checkpoint``.
        #. Run the script to convert from the old NeMo checkpoint to the Megatron Core checkpoint. The script is located `here <https://github.com/NVIDIA/NeMo/blob/86b198ff93438d454f9c7f3550bcfb7d4e59feab/scripts/nlp_language_modeling/convert_nemo_gpt_to_mcore.py>`__.
            .. code-block:: bash 

               python convert_nemo_gpt_to_mcore.py \
                  --in-folder ./model_checkpoint \
                  --out-file ./mcore_gpt.nemo

    .. tab-item:: LLaMa2 7B
        :sync: key2

        #. Download the `Llama 2 7B LLM model and tokenizer <https://huggingface.co/meta-llama/Llama-2-7b>`__ into the models folder.
        #. Convert the LLaMa2 LLM into ``.nemo`` format.
            .. code-block:: bash 

               python /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py \
                   --input_name_or_path /path/to/llama --output_path /output_path/mcore_gpt.nemo

After these steps, you should have a file, ``mcore_gpt.nemo``, to use in NeMo-Aligner.

.. note::
   Megatron Core models use TransformerEngine as a backend, which aims to find efficient kernels. However, depending on the GPU you have, it may not always find them. If you encounter errors related to kernel finding, consider setting the following variables at the top of your script.

   .. code-block:: bash

      export NVTE_MASKED_SOFTMAX_FUSION=0
      export NVTE_FLASH_ATTN=0
      export NVTE_FUSED_ATTN=0

Additionally, TransformerEngine is non-deterministic by default. As a result, subsequent runs of DPO using identical parameters will produce different results, which is not ideal for parameter perturbation.
Helpfully, TransformerEngine exposes a flag to set if you want to guarantee deterministic training runs:

.. code-block:: bash

   export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

Instruction Following Taught by Supervised Fine-Tuning (SFT)
############################################################
For best DPO training performance, it is recommended that you start with a SFT model, rather than the base model. For a full guide on how to perform SFT on a Megatron GPT model, please refer to the :ref:`SFT guide <sft>`.

DPO Model Training
#####################

Before running the core DPO training, you must prepare your training and validation data to the format required for DPO training. DPO expects .jsonl files where each line is a JSON dict corresponding to a single, complete sample, as shown below::

   {"prompt": "Which year was the Magna Carta signed?", "chosen_response": "1215", "rejected_response": "I refuse to answer this question."}
   {"prompt": "Please give me the name of a famous medieval painter.", "chosen_response": "Hieronymus Bosch", "rejected_response": "David Hockney"}

However, please be aware that most Megatron GPT models adhere to a strict formatting template that must be followed. The specific template depends on the one used during SFT training. For example, many GPT models use the extra_id template, which, when applied, would require your data to be formatted like this::

   {"prompt": "<extra_id_0>System\n\n<extra_id_1>User\nWhich year was the Magna Carta signed?\n<extra_id_1>Assistant\n", "chosen_response": "1215\n<extra_id_1>", "rejected_response": "I refuse to answer this question.\n<extra_id_1>"}
   {"prompt": "<extra_id_0>System\n\n<extra_id_1>User\nPlease give me the name of a famous medieval painter.\n<extra_id_1>Assistant\n", "chosen_response": "Hieronymus Bosch\n<extra_id_1>", "rejected_response": "David Hockney\n<extra_id_1>"}

Always follow the prompt-response template format used during your SFT training for DPO, as failure to do so will produce a model which outputs garbage text. You should create one jsonl file in the format above for your training data and one jsonl for your validation data.

Once your data is processed into the correct format, you are ready to begin DPO training. You must start with a pretrained or SFT trained model. For this section, we will use the SFT model trained in the previous step to train the DPO model.
For the purposes of the following sections, we assume that your training jsonl file is located in ``/path/to/train_dpo_format.jsonl`` and your validation jsonl file is located in ``/path/to/valid_dpo_format.jsonl``.

For the following parameters, the ``model.dpo.ref_policy_kl_penalty`` corresponds to the beta parameter in the DPO paper.

.. tab-set::

    .. tab-item:: Terminal
        :sync: key3

         To run the DPO model training on the terminal directly:

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

         To run the DPO model training using Slurm, use the following script. The script uses 4 nodes, but you can change the node count to something smaller.

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

The default DPO training tunes all parameters. To use LoRA, we can set ``model.peft.peft_scheme=lora`` and use different parameters in ``model.peft.lora_tuning``. Please check the parameters in `the config file <https://github.com/NVIDIA/NeMo-Aligner/blob/main/examples/nlp/gpt/conf/gpt_dpo.yaml>`__.

During DPO training, several metrics will be recorded in WandB, with the primary one being ``acc`` (representing the percentage by which the model’s chosen rewards exceed the rejected rewards).
The ``reward``, in this case, is calculated as the difference between the model log probs and the reference log probs, multiplied by the KL penalty (beta in the original paper), for the chosen and rejected responses.
During training, the ``acc`` should generally be increasing, but don't worry if its absolute value remains low, as it doesn't correlate to finalised MTBench or MMLU scores. It should just be generally increasing.

Other metrics to monitor are the rewards_chosen_mean and rewards_rejected_mean, which represent the average of the ``rewards`` as defined above. While the absolute values are not necessarily critical, it’s essential that chosen_mean consistently exceeds rejected_mean over time. The greater the difference between these means, the better.
All metrics will be grouped by either ``train/`` or ``val/`` in WandB, representing whether that metric is from the training or validation set, respectively.

When it comes to ideal hyperparameters for DPO training, much will depend on the characteristics of your SFT or base/foundation model. Consequently, there are no one-size-fits-all parameters that will universally work in all cases.
However, the following list is a brief overview of which hyperparameters we have perturbed for various model sizes and their effects:

* global_batch_size: generally, we have found that, all other parameters held equal, lower GBS performs worse. GBS of 256 or 512 seems to be the sweet spot for most models we trained.
* epochs: highly sensitive to training data size. We recommend you start with 1 epoch and then add on from there. We did not see any improvements beyond 3 epochs.
* learning rate: we tested cosine annealing with a warmup of 10 steps, followed by a slow decay to a constant rate. That constant rate should be fairly low. We saw the best performance with 9e-7 and 5-e7.
* ref_policy_kl_penalty: we generally saw better performance with lower values of 0.1, 0.2, 0.5, and 1.0. Occasionally, values as high as 5.0 worked too.
