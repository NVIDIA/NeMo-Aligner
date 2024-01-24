.. include:: /content/nemo.rsts

.. _prerequisite:

Prerequisite: Obtaining a pretrained model
##########################################

The NeMo framework supports efficient model alignment via the NeMo Aligner codebase.

All algorithms in NeMo Aligner will work with any GPT based model that is from mcore(i.e in the config it has ``mcore_gpt=True``). 

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

Model Alignment by Supervised Fine-Tuning (SFT)
############################################################

**Supervised Fine-Tuning** (SFT) is the process of fine-tuning a model's parameters on supervised data of inputs and outputs. It teaches the model how to follow user specified instructions. It is typically done after model pre-training. It is also an important prerequisite step in Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO). Nemo-Aligner supports two types of SFT formats:

1. **Prompt-Response**. In the *Prompt-Response* format, each example contains an input prompt and the annotated response. SFT fine-tunes the base model to follow the prompt instruction and answer in the style of the annotated responses. The prompt-response format can be used in various problems like Question Answering (Q&A) and Summarization.

2. **Chat**. In the *Chat* format, each example contains a multi-turn conversation between different roles (e.g., *User* and *Assistant*). Fine-tuning the base model on a chat format dataset is useful to align a chatbot.

Fine-Tune with a Prompt-Response Dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Step 1: Data Formatting
%%%%%%%%%%%%%%%%%%%%%%%

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
   
This script converts the *Instruction*, *Context*, and *Response* fields into *Input* and *Output*. It also concatenates the *Instruction* and *Context* fields with a ``\n\n`` separator and randomizes the order in which they appear in the input to generate a new JSONL file. This generates an output file called `databricks-dolly-15k-output.jsonl`. An example looks like this::

   {
      "input": "When did Virgin Australia start operating?\n\nVirgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based airline. It is the largest airline by fleet size to use the Virgin brand. It commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route. It suddenly found itself as a major airline in Australia's domestic market after the collapse of Ansett Australia in September 2001. The airline has since grown to directly serve 32 cities in Australia, from hubs in Brisbane, Melbourne and Sydney.",
      "output": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.",
      "category": "closed_qa"
   }

Step 2: SFT Training
%%%%%%%%%%%%%%%%%%%%

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

For this particular run on 2B model, the final train loss is around 1.536. When the training finishes there will be a file called ``megatron_gpt_sft.nemo`` for use.

.. note::
   NeMo FW has support for wandb logging. To get started with wandb please see their `quick start guide <https://docs.wandb.ai/quickstart>`__. You can turn on wandb logging with ``exp_manager.create_wandb_logger=True`` and it will log the job results to wandb.

   For the slurm scripts we provide, they all use the `pyxis <https://github.com/NVIDIA/pyxis>`__ slurm extension which require ``--container-image=`` ``--container-mounts=`` to be provided. It is not necessary to use this extension, NeMo Aligner will work on regular python environments as well.

Step 3: Inference or Further Fine-Tuning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Given the trained SFT model, we can run inference on new examples or further fine-tune the SFT model to boost the performance (e.g., RLHF or DPO). It is worthwhile to note that their inputs need to follow the **Prompt Response Template** used in this model. The template is set by ``data.train_ds.prompt_template``. The saved nemo model ``megatron_gpt_sft.nemo`` also stores the prompt format. We can ``tar -xvf megatron_gpt_sft.nemo`` and find it in `model_config.yaml`.

In this example, the template is ``"{input} {output}"``.

Fine-Tune with a Chat Dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Step 1: Data Formatting
%%%%%%%%%%%%%%%%%%%%%%%

We will use the `OpenAssistant dataset <https://huggingface.co/datasets/OpenAssistant/oasst1>`__ in this example. Download and convert the dataset into the chat format in the following script:

.. code-block:: bash

   python /opt/NeMo-Aligner/examples/nlp/data/steerlm/preprocess_openassistant_data.py --output_directory=data/oasst
      

Step 2: SFT Training
%%%%%%%%%%%%%%%%%%%%

Now that we have the data. We will use NeMo-Aligner to do the supervised fine tuning. Compared to the SFT with a prompt-response dataset, here it is important to set ``model.data.chat=True``.

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

         To run SFT via slurm do the following:

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

If you have a smaller or bigger machine make sure to change the ``trainer.num_nodes`` and ``trainer.devices`` correspondingly. This allows scaling to thousands of GPUs.

For this particular run on llama2-7b model, the final val loss is around 1.201. When the training finishes, there will be a file called ``megatron_gpt_sft.nemo`` for use.


Step 3: Inference or Further Fine-Tuning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Given the trained SFT model, we can run inference on new examples or further fine-tune the SFT model to boost the performance (e.g., RLHF or DPO). It is important to note that their inputs need to follow the **Prompt Template** used in this model. The template is set by ``data.chat_prompt_tokens``. The saved nemo model ``megatron_gpt_sft.nemo`` stores the prompt format. We can ``tar -xvf megatron_gpt_sft.nemo`` and find it in `model_config.yaml`. In this example, it is::

   prompt_template: "\0System\n{system message}\n\x11User\n{turn 1 user message}\n\x11Assistant\n\x12{turn 1 assistant label}\n{turn 1 assistant message}\n\x11User\n{turn 2 user message}\n\x11Assistant\n\x12{turn 2 assistant label}\n{turn 2 assistant message}\n\x11"


We can run inference using the ``megatron_gpt_sft.nemo`` and the prompte template. When asking the model `What is machine learning?`, its answer is

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
