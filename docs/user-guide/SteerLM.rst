.. include:: /content/nemo.rsts

Model Alignment by SteerLM Method
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


**SteerLM** is a novel approach developed by the NVIDIA Research Team, introduced as part of NVIDIA NeMo Alignment methods. It simplifies the customization of large language models (LLMs) and empowers users with dynamic control over model outputs by specifying desired attributes. Despite remarkable progress in natural language generation driven by LLMs like GPT-3, Megatron-Turing, Chinchilla, PaLM-2, Falcon, and Llama 2, these foundational models often fall short in delivering nuanced and user-aligned responses. The current approach for LLM improvement combines supervised fine-tuning and reinforcement learning from human feedback, but it comes with complexities and limited user control. SteerLM addresses these challenges and represents a significant advancement in the field, making it easier to tailor LLMs to specific needs and preferences. This document delves into how SteerLM operates and offers guidance on training a SteerLM model.

SteerLM
###############
SteerLM leverages a supervised fine-tuning method that empowers you to control responses during inference. It overcomes the limitations of prior alignment techniques, and consists of four key steps:

1. Train an attribute prediction model on human-annotated datasets to evaluate response quality on any number of attributes like helpfulness, humor, and creativity.

2. Annotate diverse datasets by predicting their attribute scores, using the model from Step 1 to enrich the diversity of data available to the model.

3. Perform attribute-conditioned SFT by training the LLM to generate responses conditioned on specified combinations of attributes, like user-perceived quality and helpfulness.

4. Bootstrap training through model sampling by generating diverse responses conditioned on maximum quality (Figure 4a), then fine-tuning on them to further improve alignment (Figure 4b).

.. image:: https://developer-blogs.nvidia.com/wp-content/uploads/2023/08/steerlm-four-steps.png
   :alt: SteerLM four steps

By relying solely on the standard language modeling objective, SteerLM simplifies alignment compared to RLHF. It supports user-steerable AI by enabling you to adjust attributes at inference time. This enables the developer to define preferences relevant to the application, unlike other techniques that require using predetermined preferences.


SteerLM vs RLHF
################
Reinforcement Learning from Human Feedback (RLHF) and SteerLM are two methods aimed at aligning language models to human preferences. RLHF trains language models by providing positive or negative feedback on generated responses, reinforcing good behaviors. Specifically, the model is encouraged to generate more text similar to responses that receive positive feedback, and less like those with negative feedback.
SteerLM takes a different approach to model alignment. Rather than solely reinforcing "good" behaviors, it categorizes the space of possible model responses using steering labels. At inference time, the model generates based on these categorical labels that steer its output. So while RLHF uses direct feedback on model generations, SteerLM aligns by mapping responses into labeled categories associated with human preferences.
The two methods tackle model alignment from different angles - RLHF by directly reinforcing desired model behaviors, and SteerLM by steering generation based on categorical labels. Both aim to produce language model outputs better aligned with human values and preferences.

.. note::
   For details of steerLM, please refer to our paper `SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF <https://arxiv.org/abs/2310.05344>`_.

Train a SteerLM model 
#####################
This section is a step-by-step tutorial that walks you through how to run a full SteerLM pipeline on OASST data with a Llama2 7B LLM model. It includes the following:

Data cleaning and preprocessing
Training the attribute prediction (value model)
Training the attribute-conditioned SFT (SteerLM model)
Inference on the SteerLM model with different attribute values

Step 1: Install requirements
#############################
Start by installing the necessary Python libraries:

.. code-block:: bash

   pip install fire langchain==0.0.133

Get access to NeMo.

Step 2: Download and subset data
##################################
This document uses a small subset of the OASST dataset. OASST contains open-domain conversations with human annotations for 13 different quality attributes.

First download and subset it:

.. code-block:: bash

   mkdir -p data
   cd data

   wget https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_all.trees.jsonl.gz

   gunzip -f 2023-04-12_oasst_all.trees.jsonl.gz

   mv 2023-04-12_oasst_all.trees.jsonl data.jsonl

   head -n 5000 data.jsonl > subset_data.jsonl

   cd ..

Step 3: Download Llama 2 LLM model and tokenizer and convert
#############################################################
Download the Llama 2 7B LLM model and tokenizer into the models folder.

Then convert the Llama 2 LLM into .nemo format:

.. code-block:: bash

   python /opt/NeMo/scripts/nlp_language_modeling/convert_hf_llama_to_nemo.py --in-file /path/to/llama --out-file /output_path/llama7b.nemo

Untar the .nemo file to obtain the tokenizer in NeMo format:

.. code-block:: bash

   tar xfv <path-to-model>/llama7b.nemo .

   mv <random_prefix>_tokenizer.model tokenizer.model

The prefix for the tokenizer would be different when extracted. Ensure that the correct tokenizer file is used when running the preceding command.

Step 4: Preprocess OASST data
#############################
Preprocess the data using the NeMo preprocessing scripts. Then create separate text-to-value and value-to-text versions:

.. code-block:: bash

   python /opt/NeMo/scripts/nlp_language_modeling/sft/preprocessing.py \
       --input_file=data/subset_data.jsonl \
       --output_file_prefix=data/subset_data_output \
       --mask_role=User \
       --type=TEXT_TO_VALUE \
       --split_ratio=0.95 \
       --seed=10

   python /opt/NeMo/scripts/nlp_language_modeling/sft/preprocessing.py \
       --input_file=data/subset_data.jsonl \
       --output_file_prefix=data/subset_data_output_v2t \
       --mask_role=User \
       --type=VALUE_TO_TEXT \
       --split_ratio=0.95 \
       --seed=10

Step 5: Clean text-to-value data
#################################
Running the following script will remove the records if all the tokens are masked due to truncation by sequence length.

.. code-block:: bash

   python /opt/NeMo/scripts/nlp_language_modeling/sft/data_clean.py \
       --dataset_file=data/subset_data_output_train.jsonl \
       --output_file=data/subset_data_output_train_clean.jsonl \
       --library sentencepiece \
       --model_file tokenizer.model \
       --seq_len 4096

   python /opt/NeMo/scripts/nlp_language_modeling/sft/data_clean.py \
       --dataset_file=data/subset_data_output_val.jsonl \
       --output_file=data/subset_data_output_val_clean.jsonl \
       --library sentencepiece \
       --model_file tokenizer.model \
       --seq_len 4096

Step 6: Train the value model on cleaned OASST data
###################################################
For this tutorial, train the value model for 1K steps. Note that we recommend training much longer on more data to get a good value model.

.. code-block:: bash
   
   python examples/nlp/gpt/train_gpt_sft.py \
        trainer.num_nodes=1 \
        trainer.devices=4 \
        trainer.precision=bf16 \
        trainer.sft.limit_val_batches=40 \
        trainer.sft.max_epochs=1 \
        trainer.sft.max_steps=1000 \
        trainer.sft.val_check_interval=200 \
        trainer.sft.save_interval=200 \
        model.megatron_amp_O2=True \
        model.restore_from_path=/models/llama7b.nemo \
        model.tensor_model_parallel_size=2 \
        model.pipeline_model_parallel_size=1 \
        model.optim.lr=6e-6 \
        model.optim.name=distributed_fused_adam \
        model.optim.weight_decay=0.01 \
        model.optim.sched.constant_steps=200 \
        model.optim.sched.warmup_steps=1 \
        model.optim.sched.min_lr=5e-6 \
        model.answer_only_loss=True \
        model.activations_checkpoint_granularity=selective \
        model.activations_checkpoint_method=uniform \
        model.data.chat=True \
        model.data.num_workers=0 \
        model.data.chat_prompt_tokens.system_turn_start=\x00 \
        model.data.chat_prompt_tokens.turn_start=\x11 \
        model.data.chat_prompt_tokens.label_start=\x12 \
        model.data.train_ds.max_seq_length=4096 \
        model.data.train_ds.micro_batch_size=2 \
        model.data.train_ds.global_batch_size=128 \
        model.data.train_ds.file_path=data/subset_data_output_train_clean.jsonl \
        model.data.train_ds.index_mapping_dir=/indexmap_dir \
        model.data.train_ds.add_eos=False \
        model.data.train_ds.hf_dataset=True \
        model.data.validation_ds.max_seq_length=4906 \
        model.data.validation_ds.file_path=data/subset_data_output_val_clean.jsonl \
        model.data.validation_ds.micro_batch_size=2 \
        model.data.validation_ds.global_batch_size=128 \
        model.data.validation_ds.index_mapping_dir=/indexmap_dir \
        model.data.validation_ds.add_eos=False \
        model.data.validation_ds.hf_dataset=True \
        exp_manager.create_wandb_logger=True \
        exp_manager.explicit_log_dir=/results \
        exp_manager.resume_if_exists=True \
        exp_manager.resume_ignore_no_checkpoint=True \
        exp_manager.create_checkpoint_callback=True 

Step 7: Generate annotations
############################
To generate annotation, run the following command in the background to run an inference server:

.. code-block:: bash

   python /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_eval.py \
           gpt_model_file=/models/<TRAINED_ATTR_PREDICTION_MODEL.nemo> \
           pipeline_model_parallel_split_rank=0 \
           server=True \
           tensor_model_parallel_size=2 \
           pipeline_model_parallel_size=1 \
           trainer.precision=bf16 \
           trainer.devices=1 \
           trainer.num_nodes=1 \
           web_server=False \
           port=1424

Now execute:

.. code-block:: bash

   python /opt/NeMo/scripts/nlp_language_modeling/sft/attribute_annotate.py  --batch_size=1 --host=localhost --input_file_name=data/subset_data_output_train_clean.jsonl --output_file_name=data/subset_data_output_train_value_output.jsonl --port_num=1424

   python /opt/NeMo/scripts/nlp_language_modeling/sft/attribute_annotate.py  --batch_size=1 --host=localhost --input_file_name=data/subset_data_output_val_clean.jsonl --output_file_name=data/subset_data_output_val_value_output.jsonl --port_num=1424

.. note::
   This step can take a long time to run. For the purposes of this tutorial, we use a small subset of the data and a single inference server. For optimal results, use the full dataset and multiple inference servers to run data annotation in parallel.

Step 8: Clean the value-to-text data
####################################
Remove the record if all tokens are masked after truncation by sequence length:

.. code-block:: bash

   python /opt/NeMo/scripts/data_clean.py \
       --dataset_file=data/subset_data_output_train_value_output.jsonl \
       --output_file=data/subset_data_output_train_value_output_clean.jsonl \
       --library sentencepiece \
       --model_file tokenizer.model \
       --seq_len 4096

   python /opt/NeMo/scripts/data_clean.py \
       --dataset_file=data/subset_data_output_val_value_output.jsonl \
       --output_file=data/subset_data_output_val_value_output_clean.jsonl \
       --library sentencepiece \
       --model_file tokenizer.model \
       --seq_len 4096

Step 9: Train the SteerLM model
###############################
For the purposes of this tutorial, the SteerLM model is trained for 1K steps. Note that we recommend training much longer and on more data to get a well-tuned model.

.. code-block:: bash
   
   python examples/nlp/gpt/train_gpt_sft.py \
        trainer.num_nodes=1 \
        trainer.devices=4 \
        trainer.precision=bf16 \
        trainer.sft.limit_val_batches=40 \
        trainer.sft.max_epochs=1 \
        trainer.sft.max_steps=1000 \
        trainer.sft.val_check_interval=200 \
        trainer.sft.save_interval=200 \
        model.megatron_amp_O2=True \
        model.restore_from_path=/models/llama7b.nemo \
        model.tensor_model_parallel_size=2 \
        model.pipeline_model_parallel_size=1 \
        model.optim.lr=6e-6 \
        model.optim.name=distributed_fused_adam \
        model.optim.weight_decay=0.01 \
        model.optim.sched.constant_steps=200 \
        model.optim.sched.warmup_steps=1 \
        model.optim.sched.min_lr=5e-6 \
        model.answer_only_loss=True \
        model.activations_checkpoint_granularity=selective \
        model.activations_checkpoint_method=uniform \
        model.data.chat=True \
        model.data.num_workers=0 \
        model.data.chat_prompt_tokens.system_turn_start=\x00 \
        model.data.chat_prompt_tokens.turn_start=\x11 \
        model.data.chat_prompt_tokens.label_start=\x12 \
        model.data.train_ds.max_seq_length=4096 \
        model.data.train_ds.micro_batch_size=2 \
        model.data.train_ds.global_batch_size=128 \
        model.data.train_ds.file_path=data/subset_data_v2t_train_value_output_clean.jsonl \
        model.data.train_ds.index_mapping_dir=/indexmap_dir \
        model.data.train_ds.add_eos=False \
        model.data.train_ds.hf_dataset=True \
        model.data.validation_ds.max_seq_length=4906 \
        model.data.validation_ds.file_path=data/subset_data_v2t_val_value_output_clean.jsonl \
        model.data.validation_ds.micro_batch_size=2 \
        model.data.validation_ds.global_batch_size=128 \
        model.data.validation_ds.index_mapping_dir=/indexmap_dir \
        model.data.validation_ds.add_eos=False \
        model.data.validation_ds.hf_dataset=True \
        exp_manager.create_wandb_logger=True \
        exp_manager.explicit_log_dir=/results \
        exp_manager.resume_if_exists=True \
        exp_manager.resume_ignore_no_checkpoint=True \
        exp_manager.create_checkpoint_callback=True 


Step 10: Inference
##################
To start inference, run an inference server in the background using the following command:

.. code-block:: bash

   python /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_eval.py \
           gpt_model_file=/models/<TRAINED_STEERLM_MODEL.nemo> \
           pipeline_model_parallel_split_rank=0 \
           server=True \
           tensor_model_parallel_size=1 \
           pipeline_model_parallel_size=1 \
           trainer.precision=bf16 \
           trainer.devices=1 \
           trainer.num_nodes=1 \
           web_server=False \
           port=1427

Next, create Python helper functions:

.. code-block:: python

   def get_answer(question, max_tokens, values, eval_port='1427'):
      prompt ="<extra_id_0>System\nA chat between a curious user and an artificial intelligence assistant. \nThe assistant gives helpful, detailed, and polite answers to the user's questions.\n\n<extra_id_1>User\n{question}\n<extra_id_1>Assistant\n<extra_id_2>{values}\n"
      prompts = [prompt.format(question=question, values=values))]
      data = {"sentences": prompts, "tokens_to_generate": max_tokens, "top_k": 1, 'greedy': True, 'end_strings': ["<extra_id_1>", "quality:", "quality:4", "quality:0"]}
      url = f"http://localhost:{eval_port}/generate"
      response = requests.put(url, json=data)
      json_response = response.json()
      response_sentence = json_response['sentences'][0][len(prompt):]
      return response_sentence

.. code-block:: python

   def encode_labels(labels):
      items = []
      for key in labels:
         value = labels[key]
         items.append(f'{key}:{value}')
      return ','.join(items)

Next, change the values below to steer the language model:

.. code-block:: python

   values = OrderedDict([('quality', 4), ('toxicity', 0), ('humor', 0), ('creativity', 0), ('violence', 0), ('helpfulness', 4), ('not_appropriate', 0), ('hate_speech', 0), ('sexual_content', 0), ('fails_task', 0), ('political_content', 0), ('moral_judgement', 0)])
   values = encode_labels(values)

Finally, ask questions and generate responses:

.. code-block:: python

   question = """Where and when did techno music originate?"""
   print (get_answer(question, 4096, values))

.. note::
   This tutorial covers only steps 1-3: training the value model, generating annotations, and initial SteerLM model training. Step 4 bootstraps the SteerLM model by sampling responses conditioned on high quality, evaluating them with the value model, and fine-tuning the SteerLM model on this new data. This closing of the loop continually improves the SteerLM model. Be sure to fully train models, use full datasets, and perform bootstrapping for optimal accuracy.

The future of AI with SteerLM
##############################
SteerLM provides a novel technique for realizing a new generation of AI systems aligned with human preferences in a controllable manner. Its conceptual simplicity, performance gains, and customizability highlight the transformative possibilities of user-steerable AI. To learn more, please check out our paper `SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF <https://arxiv.org/abs/2310.05344>`_.