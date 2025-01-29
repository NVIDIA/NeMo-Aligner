.. include:: /content/nemo.rsts

Model Generation with Data Parallelism and TRT
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

The NeMo framework supports efficient model generation via the NeMo Aligner codebase.

All algorithms in NeMo Aligner are compatible with any GPT-based model from Megatron Core (i.e., those with mcore_gpt=True in the configuration). For this tutorial, we will demonstrate the generation pipeline using a 2B GPT model with 4096 sequence length <https://huggingface.co/nvidia/GPT-2B-001>__. This tutorial is also applicable to other GPT models, such as Llama models, regardless of their size.

Obtaining a pretrained model
############################
To start, we must first get an aligned model to generate responses from. There are 2 models we recommend to get started. The rest of the tutorial will work with either model, but for demonstration purposes we will use the smaller 2B model. 

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

Additionally, TransformerEngine is non-deterministic by default, meaning subsequent runs of generation using identical parameters will produce different results, which is not ideal for generation.
Helpfully, TransformerEngine exposes a flag to set if you want to guarantee deterministic generation runs:

.. code-block:: bash

   export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
   export NVTE_MASKED_SOFTMAX_FUSION=0

Aligned vs Foundational (base) model for Generation
###################################################
Generation can be run on either base/foundational models, that is, models which have only been trained on autoregressive language prediction tasks and not on instruction following tasks,
or, you can also run generation on models which have been aligned on instruction-based or preference-based datasets as well, similar to DPO/PPO. Either model will work, but you will get much higher quality
responses (generations) from an aligned model, and we highly recommend using an aligned model for generation if you want high quality responses.

Data Format for Generation
##########################

The input files for generation in Aligner use the exact same format of .jsonl files as used by SFT in Nemo and Aligner. Please see the data formatting section of SFT to understand the data format necessary for Self-Rewarding :ref:`SFT guide <sft>`
Please note that Aligner generation does not support the use of mmap or binary files, only .jsonl files in the SFT format.    

Running Generation in Aligner
#############################

Once your data is processed into the correct format you are ready to begin generation. You must start with a pretrained or aligned model. For this section we will use the aligned model from the previous section for generation.
For the purposes of the following sections, we'll assume your generation jsonl file is located in ``/path/to/generation_sft_format.jsonl``.

The key parameters for generation are located under ``model.generation`` and include the following:

``model.generation.num_responses_to_gen`` - controls how many responses you want the model to generate per prompt

The following block shows the standard Nemo sampling params for generating responses, which are the same as we use across all Nemo and Aligner codebases:

.. code-block:: yaml
    sampling_params:
      use_greedy: False
      temperature: 1.0
      top_k: 0
      top_p: 1.0
      repetition_penalty: 1.0
      add_BOS: False
      all_probs: False
      compute_logprob: False
      end_strings: ["<|endoftext|>", "<extra_id_1>"]

    # length argument for autoregressive sampling
    # max length means max amount of tokens to generate
    length_params:
      max_length: ${int_div:${model.encoder_seq_length}, 2}
      min_length: 1

Finally, we have the TRT parameters, which allows for faster TRTLLM-based response generation:

.. code-block:: yaml
    trt_llm:
      enable: True  # use this to turn TRT on/off
      # reshard: False # reshard is not supported in generation

      # TRTLLM preallocates activation memory according to the number of input tokens
      max_input_len: ${subtract:${model.encoder_seq_length}, ${model.generation.length_params.max_length}}

      model_type: gptnext # can be gptj, gptnext, llama, gemma, falcon

      # Generation does not have a training stage, so there is no need to unload the engine.
      unload_engine_train: False


Keep in mind that Aligner generation utilises data parallelism to speed up generation. This means that your input data file will be divided by GBS, and data which is
not cleanly divisible by GBS will be dropped starting from the end of the file. For example, if your data file has 11639 samples with a GBS of 32, this means that
11639 mod 32 = 23 samples will be dropped and not generated. To avoid this, you can either reduce your data parallelism to 1, or you can pad your data file up to the nearest
multiple of your GBS (you can pad with basic prompts like "how are you"). Additionally, if you truncate your input data using the ``model.data.train_ds.max_seq_length`` parameter,
then your data will be reduced even further. Truncation applies before the DP truncation.

With your data prepared, you can now run generation. We demonstrate two techniques below, one using cmdline inputs directly, and another demonstrating the use of SLURM.


.. tab-set::

    .. tab-item:: Terminal
        :sync: key3

         To run Self-Rewarding model training on the terminal directly:

         .. code-block:: bash 

            export GPFS="/path/to/nemo-aligner-repo"
            export TRAIN_DATA_PATH="/path/to/generation_sft_format.jsonl"

            python -u ${GPFS}/examples/nlp/gpt/run_generation.py \
               trainer.num_nodes=1 \
               trainer.devices=8 \
               model.micro_batch_size=1 \
               model.global_batch_size=32 \
               pretrained_checkpoint.restore_from_path=/path/to/megatron_gpt_sft.nemo \
               "model.data.train_ds.file_path=${TRAIN_DATA_PATH}" \
               exp_manager.create_wandb_logger=false \
               exp_manager.wandb_logger_kwargs.project=null \
               exp_manager.wandb_logger_kwargs.name=null \
               exp_manager.explicit_log_dir=/results \
               ++model.sequence_parallel=false \
               ++model.apply_rope_fusion=false \
               trainer.generation.max_epochs=1 \
               model.generation.num_responses_to_gen=1 \
               trainer.generation.trt_llm.enable=true

    .. tab-item:: Slurm
        :sync: key4

         To run generation with Slurm, use the script below. The script uses 4 nodes, but you can change the node count to something different:

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

            TRAIN_DATA_PATH="/path/to/generation_sft_format.jsonl"

            PROJECT="<<WANDB PROJECT>>"

            CONTAINER=<<<CONTAINER>>> # use the latest NeMo Training container, Aligner will work there
            MOUNTS="--container-mounts=${GPFS}:${GPFS},${TRAIN_DATA_PATH}:${TRAIN_DATA_PATH},${PRETRAINED_CHECKPOINT_NEMO_FILE}:${PRETRAINED_CHECKPOINT_NEMO_FILE}"

            RESULTS_DIR="/path/to/result_dir"

            OUTFILE="${RESULTS_DIR}/rm-%j_%t.out"
            ERRFILE="${RESULTS_DIR}/rm-%j_%t.err"
            mkdir -p ${RESULTS_DIR}

            read -r -d '' cmd <<EOF
            echo "*******STARTING********" \
            && echo "---------------" \
            && echo "Starting generation" \
            && cd ${GPFS} \
            && export PYTHONPATH="${GPFS}:${PYTHONPATH}" \
            && export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
            && export NVTE_MASKED_SOFTMAX_FUSION=0 \
            && export HYDRA_FULL_ERROR=1 \
            && python -u ${GPFS}/examples/nlp/gpt/run_generation.py \
               trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
               trainer.devices=8 \
               pretrained_checkpoint.restore_from_path='${PRETRAINED_CHECKPOINT_NEMO_FILE}' \
               "model.data.train_ds.file_path=${TRAIN_DATA_PATH}" \
               model.micro_batch_size=1 \
               model.global_batch_size=32 \
               ++model.sequence_parallel=false \
               ++model.apply_rope_fusion=false \
               exp_manager.explicit_log_dir=${RESULTS_DIR} \
               exp_manager.create_wandb_logger=False \
               exp_manager.wandb_logger_kwargs.name=null \
               exp_manager.wandb_logger_kwargs.project=null \
               trainer.generation.max_epochs=1 \
               model.generation.num_responses_to_gen=1 \
               trainer.generation.trt_llm.enable=true
            EOF

            srun -o $OUTFILE -e $ERRFILE --container-image=$CONTAINER $MOUNTS bash -c "${cmd}"
            set +x

The output file containing the responses will be located in ``${RESULTS_DIR}/generations/generations.jsonl``. All responses will be stored to this file as they
are generated, and even if your generation process abruptly terminates, it will resume where it left off once restarted. Once generation is complete all of your
responses will be located in this file.

The structure of this file is a .jsonl file, where each line represents a JSON object of the following form:

.. code-block:: json
    { step: the step number in the epoch,
	  consumed_samples: the number of samples consumed so far of the input dataset,
	  prompt: the prompt passed to the model,
	  responses: a list of length ``model.generation.num_responses_to_gen`` which contains all of the responses to the input prompt
	}

The step and consumed_samples fields are not needed by the end user, but they're there so that the process can correctly resume if it unexpectedly goes down in the middle
of a generation run.

Please note that the responses will contain all raw tokens which the model generated, this includes all special headers, turn starts/ends, and BOS/EOS tokens. To get a "clean" output
the end user must filter this out themselves via some sort of post-processing step (which is not currently provided).

