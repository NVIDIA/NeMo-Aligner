.. include:: /content/nemo.rsts

.. _prerequisite:

Obtain a Pretrained Model
#########################

The NeMo Framework supports efficient model alignment using the NeMo-Aligner codebase. All algorithms in NeMo-Aligner will work with any NeMo GPT-based model. To see a collection of scripts that convert popular models from Hugging Face to ``.nemo`` format, go `here <https://github.com/NVIDIA/NeMo/tree/main/scripts/nlp_language_modeling>`__.

To get started, you need to obtain a pretrained model to align. Three models are recommended: 2B GPT, LLama2-7B, or Nemotron-340B. For demonstration purposes, the smaller 2B model will be used, but you can follow the rest of the tutorial with either model.

.. tab-set::

    .. tab-item:: 2B GPT
        :sync: key1

        1. Get the 2B checkpoint at ``wget https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo``.
        2. Extract the NeMo file to a folder with ``mkdir model_checkpoint && tar -xvf GPT-2B-001_bf16_tp1.nemo -C model_checkpoint``.
        3. Run the script to convert from the old NeMo checkpoint to the Megatron Core checkpoint. The script is located `here <https://github.com/NVIDIA/NeMo/blob/0ec7e9090d3261b8ce81818b0555a204e50d814d/scripts/checkpoint_converters/convert_gpt_nemo_to_mcore.py>`__.
            .. code-block:: bash 

               python convert_gpt_nemo_to_mcore.py \
                  --input_name_or_path ./model_checkpoint \
                  --output_path ./mcore_gpt.nemo

    .. tab-item:: LLaMa2-7B
        :sync: key2

        1. Download the `Llama2-7B LLM model and tokenizer <https://huggingface.co/meta-llama/Llama-2-7b-hf>`__ into the model's folder.
        2. Convert the LLaMa2 LLM into ``.nemo`` format.
            .. code-block:: bash 

               python /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py \
                   --input_name_or_path /path/to/llama --output_path /output_path/mcore_gpt.nemo

    .. tab-item:: Nemotron-340B
        :sync: key3

        1. Download the model from `Hugging Face <https://huggingface.co/nvidia/Nemotron-4-340B-Base>`__.
        2. For all scripts, point ``*.restore_from_path`` to the directory where you have downloaded the files. 
           Note: Because of the 340B's size, it is recommended that you use TP8 PP24 which will be safe for algorithms in Aligner.

After these steps, you will have a file called ``mcore_gpt.nemo`` to use in NeMo-Aligner.

.. note::
   If you bring your own .nemo model, make sure to change the `model.encoder_seq_length` in the Aligner configs to match the sequence length of your own model.

.. note::
   When working with Megatron Core models, which utilize the Transformer engine as a backend, the system attempts to find efficient kernels. However, depending on your GPU, it may not always locate them. If you encounter errors related to kernel finding, consider setting these variables at the top of your script.

   .. code-block:: bash

      export NVTE_MASKED_SOFTMAX_FUSION=0
      export NVTE_FLASH_ATTN=0
      export NVTE_FUSED_ATTN=0

.. _model-aligner-sft:

Model Alignment by Supervised Fine-Tuning (SFT)
###############################################

**SFT** is the process of fine-tuning a model's parameters on supervised data of inputs and outputs. It teaches the model how to follow user-specified instructions. It is typically done after model pre-training. It is also an important prerequisite step in Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO). Nemo-Aligner supports two types of SFT formats:

1. **Prompt-Response**. In the *Prompt-Response* format, each example contains an input prompt and the annotated response. SFT fine-tunes the base model to follow the prompt instruction and answer in the style of the annotated responses. The prompt-response format can be used in various problems like Question Answering (Q&A) and Summarization.

2. **Chat**. In the *Chat* format, each example contains a multi-turn conversation between different roles (e.g., *User* and *Assistant*). Fine-tuning the base model on a chat format dataset is useful to align a chatbot.

Fine-Tune with a Prompt-Response Dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Step 1: Format the data
^^^^^^^^^^^^^^^^^^^^^^^

This example uses the `Dolly dataset <https://github.com/databrickslabs/dolly>`__ to demonstrate how to format your SFT data. This dataset consists of 15,000 instruction-context-response triples.

Download the data by entering the following command::

    wget https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl

The downloaded data, stored in the file ``databricks-dolly-15k.jsonl``, follows a JSONL format, with each line structured as shown below::

   {
       "instruction": "When did Virgin Australia start operating?",
       "context": "Virgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based airline. It is the largest airline by fleet size to use the Virgin brand. It commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.[3] It suddenly found itself as a major airline in Australia's domestic market after the collapse of Ansett Australia in September 2001. The airline has since grown to directly serve 32 cities in Australia, from hubs in Brisbane, Melbourne and Sydney.[4]",
       "response": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.",
       "category": "closed_qa"
   }

As this example shows, there are no clear "input" and "output" fields that SFT requires.

For an example of how to process this data format into a JSONL file with "input" and "output" fields, see `preprocess.py <https://github.com/NVIDIA/NeMo-Megatron-Launcher/blob/8779c6ef3d8f90356aefdab3ce69d5262d660d6f/launcher_scripts/nemo_launcher/collections/dataprep_scripts/dolly_dataprep/preprocess.py>`__::

   python preprocess.py --input databricks-dolly-15k.jsonl
   
This script converts the *Instruction*, *Context*, and *Response* fields into *Input* and *Output*. It also concatenates the *Instruction* and *Context* fields with a ``\n\n`` separator and randomizes the order in which they appear in the input to generate a new JSONL file. This generates an output file called `databricks-dolly-15k-output.jsonl`. An example looks like this::

   {
      "input": "When did Virgin Australia start operating?\n\nVirgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based airline. It is the largest airline by fleet size to use the Virgin brand. It commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route. It suddenly found itself as a major airline in Australia's domestic market after the collapse of Ansett Australia in September 2001. The airline has since grown to directly serve 32 cities in Australia, from hubs in Brisbane, Melbourne and Sydney.",
      "output": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.",
      "category": "closed_qa"
   }

Sequence packing is also supported with prompt-response datasets. Sequence packing is a training technique in which multiple training examples are concatenated to create one longer sequence.
This approach eliminates the need for padding and improves GPU utilization. Refer to the `sequence packing documentation <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/optimizations/sequence_packing.html?highlight=packing#>`_ for a detailed overview of sequence packing and its advantages.

NeMo provides a script to pack your SFT prompt-response dataset. Refer to the `prepare dataset <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/optimizations/sequence_packing.html?highlight=packing#prepare-dataset>`_ section of the documentation for details on how to use this script.

Step 2: Run SFT training
^^^^^^^^^^^^^^^^^^^^^^^^^

Now, you will use the data for supervised fine-tuning with NeMo-Aligner.

.. tab-set::

    .. tab-item:: Terminal
        :sync: key3

         To run SFT on the terminal directly, use the following command. For successful execution, ensure that the NeMo-Aligner repository is set as your current working directory.

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
               model.data.train_ds.file_path=/path/to/databricks-dolly-15k-output.jsonl \
               model.data.validation_ds.micro_batch_size=1 \
               model.data.validation_ds.global_batch_size=128 \
               model.data.validation_ds.file_path=/path/to/databricks-dolly-15k-output.jsonl \
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

         To run SFT via Slurm, run the following command:

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

            TRAIN_DATA_PATH="/path/to/databricks-dolly-15k-output.jsonl"
            VALID_DATA_PATH="/path/to/databricks-dolly-15k-output.jsonl"

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
               trainer.sft.val_check_interval=100 \
               trainer.sft.save_interval=100 \
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

If using sequence packing, replace the data paths with the paths to your packed datasets. For each packed dataset, you should also set ``packed_sequence=True`` in the config:

.. code-block:: python
   +model.data.train_ds.packed_sequence=True \
   +model.data.validation_ds.packed_sequence=True

It is not required to pack both the train and validation datasets. If packing only the train dataset, exclude ``+model.data.validation_ds.packed_sequence=True``.

To scale to thousands of GPUs, adjust the ``trainer.num_nodes`` and ``trainer.devices`` accordingly based on the size of your machine.

For this particular run on the 2B model, the final training loss is approximately 1.536. Once the training finishes, you’ll find a file called ``megatron_gpt_sft.nemo`` available for use.

.. note::
   NeMo Framework supports wandb logging. To get started with wandb, see the `Quick Start Guide <https://docs.wandb.ai/quickstart>`__. You can enable wandb logging with ``exp_manager.create_wandb_logger=True`` and it will log the job results to wandb.

   The provided Slurm scripts rely on the `pyxis <https://github.com/NVIDIA/pyxis>`__ Slurm extension, which requires specifying the ``--container-image=`` ``--container-mounts=``. However, it’s important to note that NeMo-Aligner can also function in regular Python environments without this extension.

Step 3: Run inference or further fine-tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given the trained SFT model, you can run inference on new examples or fine-tune the SFT model to boost the performance (e.g., RLHF or DPO). It is important to note that their inputs need to follow the **Prompt Template** used in this model. The template is set by ``data.train_ds.prompt_template``. The saved NeMo model, ``megatron_gpt_sft.nemo``, also stores the prompt format. You can ``tar -xvf megatron_gpt_sft.nemo`` and find it in `model_config.yaml`.

In this example, the template is ``"{input} {output}"``.

Fine-Tune with a Chat Dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Step 1: Format the data
^^^^^^^^^^^^^^^^^^^^^^^

In this example, you use the `OpenAssistant dataset <https://huggingface.co/datasets/OpenAssistant/oasst1>`__. Download and convert the dataset into the chat format by using the following script:

.. code-block:: bash

   python /opt/NeMo-Aligner/examples/nlp/data/steerlm/preprocess_openassistant_data.py --output_directory=data/oasst
      

Step 2: Run SFT training
^^^^^^^^^^^^^^^^^^^^^^^^^

Now, you will use the data for supervised fine-tuning with NeMo-Aligner. Compared to the SFT with a prompt-response dataset, you need to set ``model.data.chat=True``.

.. tab-set::

    .. tab-item:: Terminal
        :sync: key3

         To run SFT on the terminal directly, use the following command. For successful execution, ensure that the NeMo-Aligner repository is set as your current working directory.

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
               model.data.chat=True \
               model.data.num_workers=0 \
               model.data.train_ds.micro_batch_size=1 \
               model.data.train_ds.global_batch_size=128 \
               model.data.train_ds.max_seq_length=4096 \
               model.data.train_ds.file_path=data/oasst/train.jsonl \
               model.data.validation_ds.micro_batch_size=1 \
               model.data.validation_ds.global_batch_size=128 \
               model.data.validation_ds.file_path=data/oasst/val.jsonl \
               model.data.validation_ds.max_seq_length=4096 \
               exp_manager.create_wandb_logger=True \
               exp_manager.explicit_log_dir=/results \
               exp_manager.wandb_logger_kwargs.project=sft_run \
               exp_manager.wandb_logger_kwargs.name=chat_sft_run \
               exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
               exp_manager.resume_if_exists=True \
               exp_manager.resume_ignore_no_checkpoint=True \
               exp_manager.create_checkpoint_callback=True \
               exp_manager.checkpoint_callback_params.monitor=validation_loss

    .. tab-item:: Slurm
        :sync: key4

         To run SFT via Slurm, run the following command:

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

            TRAIN_DATA_PATH="data/oasst/train.jsonl"
            VALID_DATA_PATH="data/oasst/val.jsonl"

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
            && python -u ${GPFS}/examples/nlp/gpt/train_gpt_sft.py 
               trainer.precision=bf16 \
               trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
               trainer.devices=8 \
               trainer.sft.max_steps=-1 \
               trainer.sft.limit_val_batches=40 \
               trainer.sft.val_check_interval=1000 \
               model.megatron_amp_O2=True \
               model.restore_from_path=${PRETRAINED_ACTOR_NEMO_FILE} \
               model.optim.lr=5e-6 \
               model.data.chat=True \
               model.data.num_workers=0 \
               model.data.train_ds.micro_batch_size=1 \
               model.data.train_ds.global_batch_size=128 \
               model.data.train_ds.file_path=${TRAIN_DATA_PATH} \
               model.data.train_ds.max_seq_length=4096 \
               model.data.validation_ds.micro_batch_size=1 \
               model.data.validation_ds.global_batch_size=128 \
               model.data.validation_ds.file_path=${VALID_DATA_PATH} \
               model.data.validation_ds.max_seq_length=4096 \
               exp_manager.create_wandb_logger=True \
               exp_manager.explicit_log_dir=${RESULTS_DIR} \
               exp_manager.wandb_logger_kwargs.project=${PROJECT} \
               exp_manager.wandb_logger_kwargs.name=chat_sft_run \
               exp_manager.resume_if_exists=True \
               exp_manager.resume_ignore_no_checkpoint=True \
               exp_manager.create_checkpoint_callback=True \
               exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
               exp_manager.checkpoint_callback_params.monitor=validation_loss
            EOF

            srun -o $OUTFILE -e $ERRFILE --container-image=$CONTAINER $MOUNTS bash -c "${cmd}"
            set +x


To scale to thousands of GPUs, adjust the ``trainer.num_nodes`` and ``trainer.devices`` accordingly based on the size of your machine.

For this particular run on the Llama2-7b model, the final val loss is around 1.201. Once the training finishes, you'll find a file called ``megatron_gpt_sft.nemo`` available for use.


Step 3: Run inference or further fine-tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given the trained SFT model, you can run inference on new examples or further fine-tune the SFT model to boost the performance (e.g., RLHF or DPO). It is important to note that their inputs need to follow the **Prompt Template** used in this model. The template is set by ``data.chat_prompt_tokens``. The saved NeMo model, ``megatron_gpt_sft.nemo``, stores the prompt format. You can ``tar -xvf megatron_gpt_sft.nemo`` and find it in `model_config.yaml`. In this example, it is::

   prompt_template: "\0System\n{system message}\n\x11User\n{turn 1 user message}\n\x11Assistant\n\x12{turn 1 assistant label}\n{turn 1 assistant message}\n\x11User\n{turn 2 user message}\n\x11Assistant\n\x12{turn 2 assistant label}\n{turn 2 assistant message}\n\x11"


You can run inference using ``megatron_gpt_sft.nemo`` and the Prompt Template. When asking the model `What is machine learning?`, the answer will be as follows:

.. code-block:: python

   """
   Machine learning is a field of computer science that focuses on building algorithms that allow machines to improve their performance
   on a task without being explicitly programmed to do so. It involves using data to train a model to make predictions or perform
   tasks based on patterns in the data.\n\nExamples of machine learning include image recognition, natural language processing, and spam 
   filtering. Machine learning algorithms can be classified into supervised and unsupervised learning. In supervised learning, the
   algorithm is provided with labeled examples of a target variable (such as the number of stars in a picture) and is tasked with learning a 
   function that maps input features to the target variable.  Unsupervised learning, on the other hand, involves finding structures in unlabeled 
   data without knowing what those structures are.\n\nMachine learning algorithms can be trained using a variety of techniques, including 
   gradient descent, stochastic gradient descent, and reinforcement learning. Once trained, the algorithms can be used to make predictions
   or perform tasks on new data without explicit programming.\n\nMachine learning has been used in a wide range of fields, including healthcare,
   finance, retail, and robotics. It has the potential to transform industries by enabling machines to process and understand vast amounts 
   of data, make predictions, and take actions autonomously.\n\nIn summary, machine learning is a branch of computer science that focuses on building
   algorithms that allow machines to learn from data and improve their performance on a task without being explicitly programmed to do so.
   """
