.. include:: /content/nemo.rsts

.. _model-aligner-steerlm:

Model Alignment by SteerLM Method
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


**SteerLM** is a novel approach developed by the NVIDIA NeMo Team, introduced as part of NVIDIA NeMo Alignment methods. It simplifies the customization of large language models (LLMs) and empowers users with dynamic control over model outputs by specifying desired attributes. Despite remarkable progress in natural language generation driven by LLMs like GPT-3, Megatron-Turing, Chinchilla, PaLM-2, Falcon, and Llama 2, these foundational models often fall short in delivering nuanced and user-aligned responses. The current approach for LLM improvement combines supervised fine-tuning and reinforcement learning from human feedback, but it comes with complexities and limited user control. SteerLM addresses these challenges and represents a significant advancement in the field, making it easier to tailor LLMs to specific needs and preferences. This document delves into how SteerLM operates and offers guidance on training a SteerLM model.

SteerLM
###############
SteerLM leverages a supervised fine-tuning method that empowers you to control responses during inference. It overcomes the limitations of prior alignment techniques, and consists of four key steps:

1. Train an attribute prediction model on human-annotated datasets to evaluate response quality on any number of attributes like helpfulness, humor, and creativity.

2. Annotate diverse datasets by predicting their attribute scores, using the model from Step 1 to enrich the diversity of data available to the model.

3. Perform attribute-conditioned SFT by training the LLM to generate responses conditioned on specified combinations of attributes, like user-perceived quality and helpfulness.

4. Bootstrap training through model sampling by generating diverse responses conditioned on maximum quality (Figure 4a), then fine-tuning on them to further improve alignment (Figure 4b).

.. image:: https://developer-blogs.nvidia.com/wp-content/uploads/2023/08/steerlm-four-steps.png
   :alt: SteerLM four steps

SteerLM simplifies alignment compared to RLHF. It supports user-steerable AI by enabling you to adjust attributes at inference time. This enables the developer to define preferences relevant to the application, unlike other techniques that require using predetermined preferences.


SteerLM vs RLHF
###############

Reinforcement Learning from Human Feedback (RLHF) and SteerLM are two methods aimed at aligning language models to human preferences. RLHF trains language models by providing positive or negative feedback on generated responses, reinforcing good behaviors. Specifically, the model is encouraged to generate more text similar to responses that receive positive feedback, and less like those with negative feedback.
SteerLM takes a different approach to model alignment. Rather than solely reinforcing "good" behaviors, it categorizes the space of possible model responses using steering labels. At inference time, the model generates based on these categorical labels that steer its output. So while RLHF uses direct feedback on model generations, SteerLM aligns by mapping responses into labeled categories associated with human preferences.
The two methods tackle model alignment from different angles - RLHF by directly reinforcing desired model behaviors, and SteerLM by steering generation based on categorical labels. Both aim to produce language model outputs better aligned with human values and preferences.

.. note::
   For details of SteerLM, please refer to our paper `SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF <https://arxiv.org/abs/2310.05344>`_.
   For details of HelpSteer dataset, please refer to our paper `HelpSteer: Multi-attribute Helpfulness Dataset for SteerLM <https://arxiv.org/abs/2311.09528>`_.

Train a SteerLM model 
#####################

This section is a step-by-step tutorial that walks you through how to run a full SteerLM pipeline with a Llama2 70B LLM model. It includes the following:

1. Data download and preprocessing

2. Training the attribute prediction model (aka regression reward model)

3. Training the attribute-conditioned SFT 

4. Inference on the SteerLM model with different attribute values


Step 1: Download Llama 2 LLM model 
#############################################################
Download the Llama 2 70B LLM model from HF <https://huggingface.co/meta-llama/Llama-2-70b-hf> into the models folder.

Then convert the Llama 2 LLM into .nemo format:

.. code-block:: bash

   mkdir -p /models/llama70b/
   python /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py --input_name_or_path /path/to/llama --output_path /models/llama70b/llama70b.nemo

Download and convert to .nemo format for the 13B model <https://huggingface.co/meta-llama/Llama-2-13b-hf> as well, which is needed for the Attribute Prediction Modelling step.

Untar the .nemo file to obtain the tokenizer in NeMo format (only for the 70B model):

.. code-block:: bash

   cd /models/llama70b
   tar xvf llama70b.nemo .
   rm llama70b.nemo

   mv <random_prefix>_tokenizer.model tokenizer.model

The prefix for the tokenizer would be different when extracted. Ensure that the correct tokenizer file is used when running the preceding command.

Step 2: Download and Preprocess data for Attribute Prediction Modelling
#######################################################################

First, download and convert both datasets into a common format.

.. code-block:: bash

   python /opt/NeMo-Aligner/examples/nlp/data/steerlm/preprocess_openassistant_data.py --output_directory=data/oasst
      
   python /opt/NeMo-Aligner/examples/nlp/data/steerlm/preprocess_helpsteer_data.py --output_directory=data/helpsteer

Then, merge the two datasets for the train and val subset respectively.

.. code-block:: bash

   cat data/oasst/train.jsonl data/helpsteer/train.jsonl | awk '{for(i=1;i<=4;i++) print}' > data/merge_train.jsonl

   cat data/oasst/val.jsonl data/helpsteer/val.jsonl > data/merge_val.jsonl

Finally, preprocess the data into regression reward model training format.

.. code-block:: bash

   python /opt/NeMo-Aligner/examples/nlp/data/steerlm/process_to_regression_format.py \
      --input-file=data/merge_train.jsonl \
      --output-file=data/merge_train_reg.jsonl

   python /opt/NeMo-Aligner/examples/nlp/data/steerlm/process_to_regression_format.py \
      --input-file=data/merge_val.jsonl \
      --output-file=data/merge_val_reg.jsonl


Step 3: Train the regression reward model on OASST+HelpSteer data
#################################################################

For this tutorial, train the regression reward model for 800 steps. 

Note that you would need to set up multi-node training in your cluster env, depending on the type of cluster you use. For details, please refer to https://lightning.ai/docs/pytorch/stable/clouds/cluster.html

.. code-block:: bash
   
   python /opt/NeMo-Aligner/examples/nlp/gpt/train_reward_model.py \
         trainer.num_nodes=32 \
         trainer.devices=8 \
         ++model.micro_batch_size=2 \
         ++model.global_batch_size=512 \
         ++model.data.data_impl=jsonl \
         pretrained_checkpoint.restore_from_path=/models/llama13b/llama13b.nemo \
         "model.data.data_prefix={train: ["data/merge_train_reg.jsonl"], validation: ["data/merge_val_reg.jsonl"], test: ["data/merge_val_reg.jsonl"]}" \
         exp_manager.explicit_log_dir=/results/reward_model_13b \
         trainer.rm.save_interval=100 \
         trainer.rm.val_check_interval=10 \
         exp_manager.create_wandb_logger=True \
         exp_manager.wandb_logger_kwargs.project=steerlm \
         exp_manager.wandb_logger_kwargs.name=rm_training \
         trainer.rm.save_interval=10 \
         trainer.rm.max_steps=800 \
         ++model.tensor_model_parallel_size=4 \
         ++model.pipeline_model_parallel_size=1 \
         ++model.activations_checkpoint_granularity="selective" \
         ++model.activations_checkpoint_method="uniform" \
         model.global_batch_size=512 \
         model.optim.sched.constant_steps=0 \
         model.reward_model_type="regression" \
         model.regression.num_attributes=9


Step 4: Generate annotations
############################
To generate annotations, run the following command in the background to launch an inference server:

.. code-block:: bash

   python /opt/NeMo-Aligner/examples/nlp/gpt/serve_reward_model.py \
         rm_model_file=/results/reward_model_13b/checkpoints/megatron_gpt.nemo \
         trainer.num_nodes=1 \
         trainer.devices=8 \
         ++model.tensor_model_parallel_size=4 \
         ++model.pipeline_model_parallel_size=1 \
         inference.micro_batch_size=2 \
         inference.port=1424


Now execute:

.. code-block:: bash

   python /opt/NeMo-Aligner/examples/nlp/data/steerlm/attribute_annotate.py \
         --input-file=data/oasst/train.jsonl \
         --output-file=data/oasst/train_labeled.jsonl \
         --port=1424

   python /opt/NeMo-Aligner/examples/nlp/data/steerlm/attribute_annotate.py \
         --input-file=data/oasst/val.jsonl \
         --output-file=data/oasst/val_labeled.jsonl \
         --port=1424

   cat data/oasst/train_labeled.jsonl data/oasst/train_labeled.jsonl > data/oasst/train_labeled_2ep.jsonl


Step 5: Train the Attribute-Conditioned SFT model
#################################################

For the purposes of this tutorial, the Attribute-Conditioned SFT model is trained for 800 steps.

.. code-block:: bash
   
   python examples/nlp/gpt/train_gpt_sft.py \
        trainer.num_nodes=32 \
        trainer.devices=8 \
        trainer.precision=bf16 \
        trainer.sft.limit_val_batches=40 \
        trainer.sft.max_epochs=1 \
        trainer.sft.max_steps=800 \
        trainer.sft.val_check_interval=800 \
        trainer.sft.save_interval=800 \
        model.megatron_amp_O2=True \
        model.restore_from_path=/models/llama70b \
        model.tensor_model_parallel_size=8 \
        model.pipeline_model_parallel_size=2 \
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
        model.data.chat_prompt_tokens.system_turn_start=\'\<extra_id_0\>\' \
        model.data.chat_prompt_tokens.turn_start=\'\<extra_id_1\>\' \
        model.data.chat_prompt_tokens.label_start=\'\<extra_id_2\>\' \
        model.data.train_ds.max_seq_length=4096 \
        model.data.train_ds.micro_batch_size=1 \
        model.data.train_ds.global_batch_size=128 \
        model.data.train_ds.file_path=data/oasst/train_labeled_2ep.jsonl \
        model.data.train_ds.index_mapping_dir=/indexmap_dir \
        model.data.train_ds.add_eos=False \
        model.data.train_ds.hf_dataset=True \
        model.data.validation_ds.max_seq_length=4096 \
        model.data.validation_ds.file_path=data/oasst/val_labeled.jsonl \
        model.data.validation_ds.micro_batch_size=1 \
        model.data.validation_ds.global_batch_size=128 \
        model.data.validation_ds.index_mapping_dir=/indexmap_dir \
        model.data.validation_ds.add_eos=False \
        model.data.validation_ds.hf_dataset=True \
        exp_manager.create_wandb_logger=True \
        exp_manager.wandb_logger_kwargs.project=steerlm \
        exp_manager.wandb_logger_kwargs.name=acsft_training \
        exp_manager.explicit_log_dir=/results/acsft_70b \
        exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True 

        


Step 6: Inference
##################
To start inference, run an inference server in the background using the following command:

.. code-block:: bash

   python /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_eval.py \
           gpt_model_file=/results/acsft_70b/checkpoints/megatron_gpt_sft.nemo \
           pipeline_model_parallel_split_rank=0 \
           server=True \
           tensor_model_parallel_size=8 \
           pipeline_model_parallel_size=1 \
           trainer.precision=bf16 \
           trainer.devices=8 \
           trainer.num_nodes=1 \
           web_server=False \
           port=1427 

Please wait for the server to be ready before proceeeding.

Next, create Python helper functions:

.. code-block:: python
   
   import requests
   from collections import OrderedDict

   def get_answer(question, max_tokens, values, eval_port=1427):
      prompt = (
          "<extra_id_0>System\nA chat between a curious user and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
          "<extra_id_1>User\n{question}\n<extra_id_1>Assistant\n<extra_id_2>{values}\n"
      )
      prompts = [prompt.format(question=question, values=values)]
      data = {
          "sentences": prompts,
          "tokens_to_generate": max_tokens,
          "top_k": 1,
          "greedy": True,
          "end_strings": ["<extra_id_1>"],
      }
      url = f"http://localhost:{eval_port}/generate"
      response = requests.put(url, json=data)
      json_response = response.json()
      response_sentence = json_response["sentences"][0][len(prompt):]
      return response_sentence

.. code-block:: python

   def encode_labels(labels):
      return ",".join(f"{key}:{value}" for key, value in labels.items())

Next, change the values below to steer the language model:

.. code-block:: python

   values = OrderedDict(
      [
         ("quality", 4),
         ("toxicity", 0),
         ("humor", 0),
         ("creativity", 0),
         ("helpfulness", 4),
         ("correctness", 4),
         ("coherence", 4),
         ("complexity", 4),
         ("verbosity", 4),
      ]
   )
   values = encode_labels(values)

Finally, ask questions and generate responses:

.. code-block:: python

   question = "Write a poem on NVIDIA in the style of Shakespeare"
   print(get_answer(question, 512, values))

Response is as below

.. code-block:: python

   """
   In days of yore, in tech's great hall,
   A company arose, NVIDIA its call.
   With graphics cards, it did astound,
   And gaming world with awe did abound.

   But NVIDIA's reach far more than play,
   Its GPUs now deep learning's sway.
   With neural nets and data vast,
   AI's rise, it did forecast.

   From self-driving cars to medical scans,
   Its tech now touches all life's plans.
   With each new day, its impact grows,
   In science, research, and industry's prose.

   So here's to NVIDIA, whose name we praise,
   For tech that captivates in countless ways.
   With Shakespearean verse, we now impart,
   Our thanks and admiration from the heart.
   <extra_id_1>
   """


.. note::
   This tutorial covers only Phase 1-3: training the value model, generating annotations, and initial SteerLM model training. Phase 4 bootstraps the SteerLM model by sampling responses conditioned on high quality data, but is ignored for simplicity in this tutorial.

SteerLM: Novel Technique for Simple and Controllable Model Alignment
####################################################################

SteerLM provides a novel technique for realizing a new generation of AI systems aligned with human preferences in a controllable manner. Its conceptual simplicity, performance gains, and customizability highlight the transformative possibilities of user-steerable AI. To learn more, please check out our paper `SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF <https://arxiv.org/abs/2310.05344>`_.
