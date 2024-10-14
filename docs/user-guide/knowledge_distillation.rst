.. include:: /content/nemo.rsts

SFT with Knowledge Distillation
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Knowledge distillation is a technique in which a smaller (student) model learns from a larger (teacher) model. The goal is to "distill" information from the teacher to the student,
resulting in a small model with comparable capabilities to the large model. There are many variants of knowledge distillation, see `<here> TODO: ADD LINK`__ for an overview.

In this tutorial, we will go through fine-tuning a 2B student using a fine-tuned LLaMa3 70B chat model. We train the 2B student to match the logits of the 70B teacher. Compared to standard SFT which trains the model to predict the next token,
this approach allows more calibrated information passing from the teacher to the student.

Step 1: Obtain the fine0tuned teacher and pre-trained student models
####################################################################
To start, we must first download the fine-tuned teacher and pre-trained student models 

.. tab-set::

    .. tab-item:: 2B GPT Student
        :sync: key1

        #. Get the 2B checkpoint via ``wget https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo``
        #. Extract the NeMo File to a folder with ``mkdir student_checkpoint && tar -xvf GPT-2B-001_bf16_tp1.nemo -C student_checkpoint``
        #. And then run the script to convert from old NeMo checkpoint to Megatron-Core checkpoint. The script is located `here <https://github.com/NVIDIA/NeMo/blob/86b198ff93438d454f9c7f3550bcfb7d4e59feab/scripts/nlp_language_modeling/convert_nemo_gpt_to_mcore.py>`__.
            .. code-block:: bash 

               python convert_nemo_gpt_to_mcore.py \
                  --in-folder ./student_checkpoint \
                  --out-file ./2b_student.nemo

    .. tab-item:: LLaMa3 70B Teacher
        :sync: key2

        #. Download the `Llama 3 70B LLM model and tokenizer <https://huggingface.co/nvidia/Llama3-ChatQA-2-70B >`__ into the models folder. ##  TODO: extract .nemo !!! figure out how to do this once hf is back up
        #. Extract the NeMo File to a folder with ``mkdir teacher_checkpoint && tar -xvf GPT-2B-001_bf16_tp1.nemo -C teacher_checkpoint`` ## TODO: update for 70B
        #. And then run the script to convert from old NeMo checkpoint to Megatron-Core checkpoint. The script is located `here <https://github.com/NVIDIA/NeMo/blob/86b198ff93438d454f9c7f3550bcfb7d4e59feab/scripts/nlp_language_modeling/convert_nemo_gpt_to_mcore.py>`__.
            .. code-block:: bash 

               python convert_nemo_gpt_to_mcore.py \
                  --in-folder ./teacher_checkpoint \
                  --out-file ./70b_teacher.nemo

After these steps you should have a files ``2b_student.nemo`` and ``70b_teacher.nemo`` to use in NeMo-Aligner.

.. note::
   Mcore models use TransformerEngine as a backend, and it tries to find efficient kernels. But depending on the GPU you have it may not find them. If you ever face errors that relate to kernel finding set these variables on top of your script.

   .. code-block:: bash

      export NVTE_MASKED_SOFTMAX_FUSION=0
      export NVTE_FLASH_ATTN=0
      export NVTE_FUSED_ATTN=0


Step 2: Download the data
#########################

In this example, you use the `OpenAssistant dataset <https://huggingface.co/datasets/OpenAssistant/oasst1>`__. Download and convert the dataset into the chat format by using the following script:

.. code-block:: bash

   python /opt/NeMo-Aligner/examples/nlp/data/steerlm/preprocess_openassistant_data.py --output_directory=data/oasst

Step 3: Cache the teacher's logits
##################################

Next, we augment the dataset with the logits from the teacher. 

.. tab-set::

    .. tab-item:: Terminal
        :sync: key3

         To run SFT on the terminal directly, use the following command. For successful execution, ensure that the NeMo-Aligner repository is set as your current working directory.

         .. code-block:: bash 

            python examples/nlp/synthetic_data_gen/compute_topk_logits.py \
               trainer.num_nodes=1 \
               trainer.devices=8 \
               trainer.precision=bf16 \
               pretrained_checkpoint.restore_from_path=teacher_checkpoint/70b_teacher.nemo \ ## TODO: unzip?
               model.megatron_amp_O2=True \
               model.tensor_model_parallel_size=8 \ ## TODO: test
               data.chat=True \
               data.sample=True \
               data.num_workers=0 \
               data.data.max_seq_length=4096 \
               data.data.file_path=data/oasst/train.jsonl \
               data.data.add_eos=False \
               data.data.hf_dataset=True \
               top_k=25 \
               batch_size=8 \ ## TODO: tune!
               forward_micro_batch_size=1 \
               start_from_idx=0 \
               output_path=data/oasst/train_with_logits.jsonl

               ## TODO: remove this? should default to oasst dir
               data.data.index_mapping_dir=/indexmap_dir \


    .. tab-item:: Slurm
        :sync: key4

         To generate the teacher logits via slurm, run the following command: 

         ### TODO: make the mounts more explicit

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
            && python examples/nlp/synthetic_data_gen/compute_topk_logits.py \
                  trainer.num_nodes=\${SLURM_JOB_NUM_NODES} \
                  trainer.devices=\${SLURM_NTASKS_PER_NODE} \
                  trainer.precision=bf16 \
                  pretrained_checkpoint.restore_from_path=teacher_checkpoint/70b_teacher.nemo \ ## TODO: unzip?
                  model.megatron_amp_O2=True \
                  model.tensor_model_parallel_size=8 \ ## TODO: test
                  data.chat=True \
                  data.sample=True \
                  data.num_workers=0 \
                  data.data.max_seq_length=4096 \
                  data.data.file_path=data/oasst/train.jsonl \
                  data.data.add_eos=False \
                  data.data.hf_dataset=True \
                  top_k=25 \
                  batch_size=8 \ ## TODO: tune! make this depend on the number of devices
                  forward_micro_batch_size=1 \
                  start_from_idx=0 \
                  output_path=data/oasst/train_with_logits.jsonl
            EOF

            srun -o $OUTFILE -e $ERRFILE --container-image=$CONTAINER $MOUNTS bash -c "${cmd}"
            set +x

## TODO: check below!     
Note that storing the teacher's logits cam be quite memory intensive. To avoid going out of memory when loading the data, a chunked approach is used in which
data is loaded into memory in chunks. The example above uses a single chunk. To use multiple chunks, run the code multiple times, changing the ``start_from_idx`` and ``end_at_idx`` indices to
exhaust the entire dataset:

.. code-block:: bash

   start_from_idx=${START_FROM_IDX} \
   end_at_idx=${END_AT_IDX}

Each time the code is run, a single chunk will be produced.

## TODO: run the same code for validation and test datasets

Step 4: Fine-tune the student
#############################

Once the data has been prepared, you are ready to fine-tune the student model:

.. tab-set::

    .. tab-item:: Terminal
        :sync: key3

         To run SFT on the terminal directly, use the following command. For successful execution, ensure that the NeMo-Aligner repository is set as your current working directory.

         .. code-block:: bash 

            python /opt/NeMo-Aligner/examples/nlp/gpt/train_gpt_knowledge_distillation.py \
               trainer.num_nodes=1 \
               trainer.devices=8 \
               trainer.precision=bf16 \
               trainer.knowledge_distillation.limit_val_batches=40 \
               trainer.knowledge_distillation.max_steps=-1 \
               trainer.knowledge_distillation.val_check_interval=1000 \
               trainer.knowledge_distillation.save_interval=1000 \
               pretrained_checkpoint.restore_from_path=student_checkpont/2b_student.nemo \ ## TODO: check
               model.megatron_amp_O2=True \
               ++model.tensor_model_parallel_size=1 \
               ++model.pipeline_model_parallel_size=1 \
               model.micro_batch_size=1 \
               model.global_batch_size=128 \
               model.optim.lr=5e-6 \
               model.optim.name=distributed_fused_adam \
               model.optim.weight_decay=0.01 \
               model.knowledge_distillation.target_logits_scale=1.0 \
               model.knowledge_distillation.logits_scale=1.0 \
               model.knowledge_distillation.sft_loss_weight=0.1 \
               model.knowledge_distillation.kd_loss_weight=1.0 \
               model.knowledge_distillation.kd_loss=bwd_kl \
               "model.data.data_prefix={train: [data/oasst/train_with_logits.jsonl], validation: [data/oasst/val_with_logits.jsonl], test: [data/oasst/test_with_logits.jsonl]}" \
               ++model.data.data_impl=chunked_jsonl \
               ++model.data.n_chunks=1 \
               ++model.data.n_examples_per_chunk=${N_EXAMPLES_PER_CHUNK} \ ## TODO: set for this example!! should we just use 100 examples per chunk?
               ++model.data.seq_length=4096 \
               exp_manager.create_wandb_logger=True \
               exp_manager.wandb_logger_kwargs.name=sft_knowledge_distillation_70b_chat \
               exp_manager.wandb_logger_kwargs.project=sft_knowledge_distillation \
               exp_manager.explicit_log_dir=/results/log_dir \ ## TODO: set up mount
               exp_manager.resume_if_exists=True \
               exp_manager.resume_ignore_no_checkpoint=True \
               exp_manager.checkpoint_callback_params.monitor=val_loss \
               exp_manager.checkpoint_callback_params.save_nemo_on_train_end=False


    .. tab-item:: Slurm
        :sync: key4

         To generate the teacher logits via slurm, run the following command: 

         ### TODO: make the mounts more explicit

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
            && python /opt/NeMo-Aligner/examples/nlp/gpt/train_gpt_knowledge_distillation.py \
               trainer.num_nodes=1 \
               trainer.devices=8 \
               trainer.precision=bf16 \
               trainer.knowledge_distillation.limit_val_batches=40 \
               trainer.knowledge_distillation.max_steps=-1 \
               trainer.knowledge_distillation.val_check_interval=1000 \
               trainer.knowledge_distillation.save_interval=1000 \
               pretrained_checkpoint.restore_from_path=student_checkpont/2b_student.nemo \ ## TODO: check
               model.megatron_amp_O2=True \
               ++model.tensor_model_parallel_size=1 \
               ++model.pipeline_model_parallel_size=1 \
               model.micro_batch_size=1 \
               model.global_batch_size=128 \
               model.optim.lr=5e-6 \
               model.optim.name=distributed_fused_adam \
               model.optim.weight_decay=0.01 \
               model.knowledge_distillation.target_logits_scale=1.0 \
               model.knowledge_distillation.logits_scale=1.0 \
               model.knowledge_distillation.sft_loss_weight=0.1 \
               model.knowledge_distillation.kd_loss_weight=1.0 \
               model.knowledge_distillation.kd_loss=bwd_kl \
               "model.data.data_prefix={train: [data/oasst/train_with_logits.jsonl], validation: [data/oasst/val_with_logits.jsonl], test: [data/oasst/test_with_logits.jsonl]}" \
               ++model.data.data_impl=chunked_jsonl \
               ++model.data.n_chunks=1 \
               ++model.data.n_examples_per_chunk=${N_EXAMPLES_PER_CHUNK} \ ## TODO: set for this example!! should we just use 100 examples per chunk?
               ++model.data.seq_length=4096 \
               exp_manager.create_wandb_logger=True \
               exp_manager.wandb_logger_kwargs.name=sft_knowledge_distillation_70b_chat \
               exp_manager.wandb_logger_kwargs.project=sft_knowledge_distillation \
               exp_manager.explicit_log_dir=/results/log_dir \ ## TODO: set up mount
               exp_manager.resume_if_exists=True \
               exp_manager.resume_ignore_no_checkpoint=True \
               exp_manager.checkpoint_callback_params.monitor=val_loss \
               exp_manager.checkpoint_callback_params.save_nemo_on_train_end=False

            srun -o $OUTFILE -e $ERRFILE --container-image=$CONTAINER $MOUNTS bash -c "${cmd}"
            set +x

If running with multiple chunks, modify ``data.n_chunks`` and ``data.n_examples_per_chunk`` accordingly. ### TODO: add details

### TODO: add details on how to evaluate model following training