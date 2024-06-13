.. .. include:: /content/nemo.rsts

.. _model-aligner-steerlm2:


SteerLM 2.0: Iterative Training for Attribute-Conditioned Language Model Alignment
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

**SteerLM 2.0** is a novel approach for aligning large language models (LLMs) to generate responses with desired attribute values, building upon the original `SteerLM <model-aligner-steerlm>`_ method [1]_ . While SteerLM conducts attribute-conditioned supervised fine-tuning to steer LLM outputs, SteerLM 2.0 introduces an iterative training procedure to explicitly enforce the generated responses to follow the desired attribute distribution.


Overview
##########

The goal of SteerLM 2.0 is to train a model :math:`Q_\theta(y|a, x)` that can generate responses :math:`y` conditioned on a prompt :math:`x` and desired attributes :math:`a`, while approximating the optimal conditional distribution :math:`P(y|a, x)` derived from an attribute prediction model :math:`P(a|x, y)` and an unconditional response model :math:`P(y|x)`.
SteerLM 2.0 accomplishes this by minimizing the Kullback-Leibler (KL) divergence between :math:`P(y|a, x)` and :math:`Q_\theta(y|a, x)`:

.. math::

   \min_\theta \mathbb{E}_{a, x} D_{KL}(P(y|a, x) || Q_\theta(y|a, x))

This KL divergence loss can be optimized using samples from an initial SteerLM model :math:`Q'(y|a, x)`, leading to an efficient gradient estimation procedure (see [2]_ for derivations).

Method Details
###############

**Construct the optimal conditional distribution** :math:`P(y|a, x)`:
Using Bayes' rule and the attribute prediction model :math:`P(a|x, y)`, we can derive the optimal conditional distribution as:

.. math::

   P(y|a, x) \propto P(a|x, y) P(y|x)

**Train the SteerLM 2.0 model** :math:`Q_\theta(y|a, x)`:
The SteerLM 2.0 model :math:`Q_\theta(y|a, x)` is trained to approximate :math:`P(y|a, x)` by minimizing the KL divergence loss using samples from an initial SteerLM model :math:`Q'(y|a, x)`. The gradient is estimated as:

.. math::

   \nabla_\theta L \approx -\sum_{y_i \sim Q'(y|a, x)} (w'_i - b'_i) \nabla_{\theta} \log Q_{\theta}(y_i|a, x)

where :math:`w'_i` and :math:`b'_i` are normalized importance weights targeting :math:`P(y|a, x)` and a baseline for stable optimization, respectively.(see [2]_ for details).

**Iterative Training (optional)**: SteerLM 2.0 can be conducted in iterations (e.g., :math:`n=2`) using the optimized policy after each iteration to sample responses and train an improved policy. In each iteration, multiple diverse responses are sampled from the current model and used for the next round of training.

By iteratively training on this loss, SteerLM 2.0 can learn to generate responses :math:`y` that better conform to specified attribute values :math:`a` for a given prompt :math:`x`.

Train a SteerLM 2.0 Model
###########################

Preparing the Training Dataset
------------------------------

SteerLM 2.0 requires a specific data format to train the model effectively. According to the SteerLM 2.0 method, the following components are needed:

- A supervised fine-tuning (SFT) model :math:`P(y|x)` that generates responses :math:`y` given a prompt :math:`x`
- An original SteerLM model :math:`Q'(y|a, x)` that generates responses :math:`y` conditioned on attributes :math:`a` and prompt :math:`x`

The SteerLM 2.0 model :math:`Q_\theta(y|a, x)` is initialized with the weights from :math:`Q'(y|a, x)` and optimized to approximate the optimal conditional distribution :math:`P(y|a, x)` derived from the attribute prediction model :math:`P(a|x, y)` and the unconditional response model :math:`P(y|x)`.

To facilitate this training process, a specific data format is proposed:

.. code-block:: json

   {
   "system": "system prompt",
   "prompt_turns": [
      {"from": "User", "value": "x_user_turn_1"},
      {"from": "Assistant", "value": "x_assistant_turn_1"},
      {"from": "User", "value": "x_user_turn_2"}
   ],
   "label": "a",
   "responses": [
      {
         "from": "Assistant",
         "value": "y_1",
         "log(P(a|x,y))": "v1",
         "log(P(y|x))": "v2",
         "log(Q(y|a,x))": "v3"
      },
      {
         "from": "Assistant",
         "value": "y_2",
         "log(P(a|x,y))": "v1",
         "log(P(y|x))": "v2",
         "log(Q(y|a,x))": "v3"
      },
      ...
      {
         "from": "Assistant",
         "value": "y_n",
         "log(P(a|x,y))": "v1",
         "log(P(y|x))": "v2",
         "log(Q(y|a,x))": "v3"
      }
   ]
   }

For a given attribute string a and prompt x (constructed from prompt turns and the system turn), n responses :math:`y_i` are sampled. To compute the loss, the following values are required:

- :math:`\log P(a|y_i, x)`: The attribute prediction model's output log-probability for the attributes a given the prompt x and response :math:`y_i`
- :math:`\log P(y_i|x)`: The unconditional response model's output log-probability for the response :math:`y_i` given the prompt x
- :math:`\log Q'(y_i|a, x)`: The original SteerLM model's output log-probability for the response :math:`y_i` given the attributes a and prompt x

These values are provided as log(P(a|x,y)), log(P(y|x)), and log(Q(y|a,x)), respectively, for each sampled response :math:`y_i`.

Training Example
------------------

By organizing the data in this format, the SteerLM 2.0 model can be effectively trained to generate responses that conform to the desired attribute values while approximating the optimal conditional distribution :math:`P(y|a, x)`. Following is an example of launching the training of SteerLM 2.0:

.. code-block:: bash
   
   python examples/nlp/gpt/train_steerlm2.py \
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
        model.steerlm2.micro_batch_size=2 \
        model.steerlm2.forward_micro_batch_size=2 \
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

Inference
------------------

Since the SteerLM 2.0 Model is an extension of the original SteerLM model, the inference process is similar. Please refer to the `SteerLM <model-aligner-steerlm>`_ documentation for more details.

References
----------

.. [1] Dong, Y., Delalleau, O., Zeng, J., Shen, G., Zhang, J.J., Sreedhar, M.N., Kuchaiev, O. (2023). SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF.

.. [2] Wang, Z., Dong, Y., Delalleau, O., Zeng, J., Shen, G., Zhang, J.J., Sreedhar, M.N., Kuchaiev, O. (2024). HelpSteer2: Open-source dataset for training top-performing reward models.