.. include:: /content/nemo.rsts

Supervised Fine-Tuning (SFT) with Knowledge Distillation
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Knowledge distillation is a technique in which a smaller (student) model learns from a larger (teacher) model. The goal is to "distill" information from the teacher to the student.
Compared to standard SFT which trains the model to predict the next token, knowledge distillation allows more calibrated information passing from the teacher to the student.
There are two primary benefits of knowledge distillation compared to standard supervised fine-tuning: (1) convergence in fewer training tokens, and (2) improved accuracy.

There are many variants of knowledge distillation. NeMo Aligner supports training the student model to match the top-K logits of the teacher model. In this tutorial, we will go through fine-tuning a 2B student using a fine-tuned Nemotron 8B chat model.

.. note::
   Before starting this tutorial, be sure to review the :ref:`introduction <model-aligner-intro>` for tips on setting up your NeMo-Aligner environment.

   If you run into any problems, refer to NeMo's `Known Issues page <https://docs.nvidia.com/nemo-framework/user-guide/latest/knownissues.html>`__. The page enumerates known issues and provides suggested workarounds where appropriate.


Obtain the fine-tuned teacher and pre-trained student models
############################################################
To start, we must first download both the pre-trained student and fine-tuned teacher models

.. tab-set::

    .. tab-item:: 2B GPT Student
        :sync: key1

        #. Get the 2B checkpoint: ``wget https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo``.
        #. Extract the NeMo File to a folder with ``mkdir student_checkpoint && tar -xvf GPT-2B-001_bf16_tp1.nemo -C student_checkpoint``.
        #. And then run the script to convert from old NeMo checkpoint to Megatron-Core checkpoint. The script is located `here <https://github.com/NVIDIA/NeMo/blob/66646b83737d9a0facb1d8d714e0424fc86ec21a/scripts/checkpoint_converters/convert_gpt_nemo_to_mcore.py>`__.
            
            .. code-block:: bash 

               python convert_gpt_nemo_to_mcore.py \
                  --input_name_or_path ./student_checkpoint \
                  --output_path ./2b_student.nemo

    .. tab-item:: Nemotron 8B Teacher
        :sync: key2

        #. Download the `Llama3-8B LLM model and tokenizer <https://huggingface.co/meta-llama/Meta-Llama-3-8B>`__ into the model's folder. You can use the Hugging Face CLI for this:
            
            .. code-block:: bash
               huggingface-cli download nvidia/nemotron-3-8b-chat-4k-sft --local-dir teacher_checkpoint

After these steps you should have files ``2b_student.nemo`` and ``teacher_checkpoint/Nemotron-3-8B-Chat-4k-SFT.nemo`` to use in NeMo-Aligner.

.. note::
   Megatron Core models use TransformerEngine as a backend, which attempts to find efficient kernels. However, depending on your GPU, it may not always succeed. If you encounter errors related to kernel finding, set these variables at the top of your script.

   .. code-block:: bash

      export NVTE_MASKED_SOFTMAX_FUSION=0
      export NVTE_FLASH_ATTN=0
      export NVTE_FUSED_ATTN=0


Download the Data
#################

In this example, we will use the `OpenAssistant dataset <https://huggingface.co/datasets/OpenAssistant/oasst1>`__. Download and convert the dataset into the chat format by using the following script:

.. code-block:: bash

   python /opt/NeMo-Aligner/examples/nlp/data/steerlm/preprocess_openassistant_data.py --output_directory=data/oasst

Cache the Teacher's Logits
##########################

Next, we augment the dataset with the logits from the teacher. Note that this code will generate the top-K teacher logits for each example in descending order. For the purposes of this tutorial, we save the teacher's top four logits by setting:

.. code-block:: bash

   top_k=4

In practice, ``k`` is usually set to something larger, such as 100.

.. important::
   Failing to save the teacher's logits in descending order may affect convergence. If you choose to compute the teacher's logits using a different script than the one provided in this example, make sure the resulting dataset still has the teacher's logits in descending order.


This step takes around 20 minutes on 8 H100 80G GPUs.

.. tab-set::

    .. tab-item:: Terminal
        :sync: key3

         To run fine-tuning on the terminal directly, use the following command. For successful execution, ensure that the NeMo-Aligner repository is set as your current working directory.

         .. code-block:: bash 

            python examples/nlp/synthetic_data_gen/compute_topk_logits.py \
               trainer.num_nodes=1 \
               trainer.devices=8 \
               trainer.precision=bf16 \
               pretrained_checkpoint.restore_from_path=teacher_checkpoint/Nemotron-3-8B-Chat-4k-SFT.nemo \
               model.megatron_amp_O2=True \
               model.tensor_model_parallel_size=1 \
               data.chat=True \
               data.sample=True \
               data.num_workers=0 \
               data.data.max_seq_length=4096 \
               data.data.file_path=data/oasst/train.jsonl \
               data.data.add_eos=False \
               data.data.hf_dataset=True \
               top_k=4 \
               model.global_batch_size=16 \
               model.micro_batch_size=2 \
               start_from_idx=0 \
               end_at_idx=56439 \
               output_path=data/oasst/train_with_logits_0.jsonl


    .. tab-item:: Slurm
        :sync: key4

         To generate the teacher logits via Slurm, run the following command:

         .. code-block:: bash 

            #!/bin/bash
            #SBATCH -A <<YOUR ACCOUNT>>
            #SBATCH -p <<<YOUR PARTITION>>>
            #SBATCH -N 1
            #SBATCH -t 2:00:00
            #SBATCH -J <<<JOB NAME>>>
            #SBATCH --ntasks-per-node=8
            #SBATCH --gpus-per-node=8
            #SBATCH --exclusive
            #SBATCH --overcommit

            GPFS="/path/to/nemo-aligner-repo"

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
            && echo "Starting..." \
            && cd ${GPFS} \
            && export PYTHONPATH="${GPFS}:${PYTHONPATH}" \
            && export HYDRA_FULL_ERROR=1 \
            && python examples/nlp/synthetic_data_gen/compute_topk_logits.py \
                  trainer.num_nodes=\${SLURM_JOB_NUM_NODES} \
                  trainer.devices=\${SLURM_NTASKS_PER_NODE} \
                  trainer.precision=bf16 \
                  pretrained_checkpoint.restore_from_path=teacher_checkpoint/Nemotron-3-8B-Chat-4k-SFT.nemo \
                  model.megatron_amp_O2=True \
                  model.tensor_model_parallel_size=1 \
                  data.chat=True \
                  data.sample=True \
                  data.num_workers=0 \
                  data.data.max_seq_length=4096 \
                  data.data.file_path=data/oasst/train.jsonl \
                  data.data.add_eos=False \
                  data.data.hf_dataset=True \
                  top_k=4 \
                  model.global_batch_size=16 \
                  model.micro_batch_size=2 \
                  start_from_idx=0 \
                  end_at_idx=56439 \
                  output_path=data/oasst/train_with_logits_0.jsonl
            EOF

            srun -o $OUTFILE -e $ERRFILE --container-image=$CONTAINER $MOUNTS bash -c "${cmd}"
            set +x


You can also generate the teacher logits for the validation dataset by replacing these lines:

.. code-block:: bash

   data.data.file_path=data/oasst/train.jsonl \
   end_at_idx=56399 \
   output_path=data/oasst/train_with_logits_0.jsonl

with

.. code-block:: bash

   data.data.file_path=data/oasst/val.jsonl \
   end_at_idx=2937 \
   output_path=data/oasst/val_with_logits_0.jsonl

.. note::
   Storing the teacher's logits can be quite memory intensive. To avoid going out of memory when loading the data, the data is loaded into memory in chunks. The example above uses a single chunk.
   
   To use multiple chunks, run the code multiple times, changing the ``start_from_idx`` and ``end_at_idx`` indices to exhaust the entire dataset:

   .. code-block:: bash

      start_from_idx=${START_FROM_IDX} \
      end_at_idx=${END_AT_IDX} \
      output_path=data/oasst/train_with_logits_${CHUNK_INDEX}.jsonl

   Each time the code is run, a single chunk will be produced. Note that the output path should be suffixed with the chunk index. The index is expected to range from 0 to num_chunks - 1.

Fine-Tune the Student
#####################

Once the data has been prepared, you are ready to fine-tune the student model.  This step takes around 50 minutes on 8 H100 80G GPUs.

.. tab-set::

    .. tab-item:: Terminal
        :sync: key3

         To run fine-tuning on the terminal directly, use the following command. For successful execution, ensure that the NeMo-Aligner repository is set as your current working directory.

         .. code-block:: bash 

            python -u examples/nlp/gpt/train_gpt_knowledge_distillation.py \
               trainer.num_nodes=1 \
               trainer.devices=8 \
               trainer.precision=bf16 \
               trainer.knowledge_distillation.limit_val_batches=5 \
               trainer.knowledge_distillation.max_steps=100 \
               trainer.knowledge_distillation.val_check_interval=1000 \
               trainer.knowledge_distillation.save_interval=1000 \
               pretrained_checkpoint.restore_from_path=2b_student.nemo \
               model.megatron_amp_O2=True \
               ++model.tensor_model_parallel_size=1 \
               ++model.pipeline_model_parallel_size=1 \
               model.micro_batch_size=1 \
               model.global_batch_size=128 \
               model.optim.lr=1e-5 \
               model.optim.name=distributed_fused_adam \
               model.optim.weight_decay=0.01 \
               model.optim.sched.constant_steps=0 \
               model.knowledge_distillation.target_logits_scale=1.0 \
               model.knowledge_distillation.logits_scale=1.0 \
               model.knowledge_distillation.sft_loss_weight=0.4 \
               model.knowledge_distillation.kd_loss_weight=1 \
               model.knowledge_distillation.kd_loss=fwd_kl \
               "model.data.data_prefix={train: [data/oasst/train_with_logits_CHUNK_ID.jsonl], validation: [data/oasst/val_with_logits_CHUNK_ID.jsonl], test: [data/oasst/val_with_logits_CHUNK_ID.jsonl]}" \
               ++model.data.data_impl=chunked_jsonl \
               ++model.data.n_chunks=1 \
               ++"model.data.n_examples_per_chunk={train: 56440, validation: 2938, test: 2938}" \
               ++model.data.seq_length=4096 \
               model.data.splits_string=\'98,1,1\' \
               exp_manager.create_wandb_logger=True \
               exp_manager.wandb_logger_kwargs.name=sft_knowledge_distillation_70b_chat \
               exp_manager.wandb_logger_kwargs.project=sft_knowledge_distillation \
               exp_manager.explicit_log_dir=results/kd_log_dir \
               exp_manager.resume_if_exists=True \
               exp_manager.resume_ignore_no_checkpoint=True \
               exp_manager.checkpoint_callback_params.monitor=val_loss \
               exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True


    .. tab-item:: Slurm
        :sync: key4

         To generate the teacher logits via Slurm, run the following command:

         .. code-block:: bash 

            #!/bin/bash
            #SBATCH -A <<YOUR ACCOUNT>>
            #SBATCH -p <<<YOUR PARTITION>>>
            #SBATCH -N 1
            #SBATCH -t 4:00:00
            #SBATCH -J <<<JOB NAME>>>
            #SBATCH --ntasks-per-node=8
            #SBATCH --gpus-per-node=8
            #SBATCH --exclusive
            #SBATCH --overcommit

            GPFS="/path/to/nemo-aligner-repo"

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
            && python -u /opt/NeMo-Aligner/examples/nlp/gpt/train_gpt_knowledge_distillation.py \
                  trainer.num_nodes=\${SLURM_JOB_NUM_NODES} \
                  trainer.devices=\${SLURM_NTASKS_PER_NODE} \
                  trainer.precision=bf16 \
                  trainer.knowledge_distillation.limit_val_batches=5 \
                  trainer.knowledge_distillation.max_steps=100 \
                  trainer.knowledge_distillation.val_check_interval=1000 \
                  trainer.knowledge_distillation.save_interval=1000 \
                  pretrained_checkpoint.restore_from_path=2b_student.nemo \
                  model.megatron_amp_O2=True \
                  ++model.tensor_model_parallel_size=1 \
                  ++model.pipeline_model_parallel_size=1 \
                  model.micro_batch_size=1 \
                  model.global_batch_size=128 \
                  model.optim.lr=1e-5 \
                  model.optim.name=distributed_fused_adam \
                  model.optim.weight_decay=0.01 \
                  model.optim.sched.constant_steps=0 \
                  model.knowledge_distillation.target_logits_scale=1.0 \
                  model.knowledge_distillation.logits_scale=1.0 \
                  model.knowledge_distillation.sft_loss_weight=0.4 \
                  model.knowledge_distillation.kd_loss_weight=1 \
                  model.knowledge_distillation.kd_loss=fwd_kl \
                  "model.data.data_prefix={train: [data/oasst/train_with_logits_CHUNK_ID.jsonl], validation: [data/oasst/val_with_logits_CHUNK_ID.jsonl], test: [data/oasst/val_with_logits_CHUNK_ID.jsonl]}" \
                  ++model.data.data_impl=chunked_jsonl \
                  ++model.data.n_chunks=1 \
                  ++"model.data.n_examples_per_chunk={train: 56440, validation: 2938, test: 2938}" \
                  ++model.data.seq_length=4096 \
                  model.data.splits_string=\'98,1,1\' \
                  exp_manager.create_wandb_logger=True \
                  exp_manager.wandb_logger_kwargs.name=sft_knowledge_distillation_70b_chat \
                  exp_manager.wandb_logger_kwargs.project=sft_knowledge_distillation \
                  exp_manager.explicit_log_dir=results/kd_log_dir \
                  exp_manager.resume_if_exists=True \
                  exp_manager.resume_ignore_no_checkpoint=True \
                  exp_manager.checkpoint_callback_params.monitor=val_loss \
                  exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True
            EOF

            srun -o $OUTFILE -e $ERRFILE --container-image=$CONTAINER $MOUNTS bash -c "${cmd}"
            set +x

If running with multiple chunks, modify ``data.n_chunks`` and ``data.n_examples_per_chunk`` accordingly. The data prefixes (for example, ``data/oasst/train_with_logits_CHUNK_ID.jsonl``) should remain unchanged.
For example, if ``data.n_chunks=10`` and ``data.n_examples_per_chunk=100``, we should have 10 files with names ``data/oasst/train_with_logits_0.jsonl, ..., data/oasst/train_with_logits_9.jsonl``.
Each of the files should have 100 examples. The file template ``data/oasst/train_with_logits_CHUNK_ID.jsonl`` should be passed to ``modle.data.data_prefix`` as shown in the example above.
The ``CHUNK_ID`` will be replaced with ``0`` to ``data.n_chunks-1`` at data load time.

Results
#######

The following table exemplifies the advantage of knowledge distillation. A pre-trained `Nemotron-4 15B model <https://arxiv.org/pdf/2402.16819>`__ was fine-tuned using vanilla SFT loss and using a combination
of SFT loss and knowledge distillation loss. Knowledge distillation was performed using a Nemotron-4 340B SFT model. Fine-tuning was performed using a math/code dataset.

With only a minimal drop in throughput, knowledge distillation yields better accuracy than SFT while requiring fewer training steps to converge.

=============== ================== ============ ============= ================== ======================= ============= =================== ==================
Base Model      Training objective Train steps  MMLU (5-shot) MMLU (0-shot)      HumanEval (0-shot)      MBPP (0-shot) GSM8K (0-shot)      MATH (0-shot)
=============== ================== ============ ============= ================== ======================= ============= =================== ==================
Nemotron 15B    SFT loss           600,000      65.3          56.9               64.6                    71.7          84.2                30.12
Nemotron 15B    KD + SFT loss      420,000      65.3          57.3               70.1                    73.3          85.2                35.84
Nemotron 15B    KD + SFT loss      600,000      65.3          57.6               72                      73.8          84.8                36.6
=============== ================== ============ ============= ================== ======================= ============= =================== ==================
