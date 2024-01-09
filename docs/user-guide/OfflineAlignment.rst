.. include:: /content/nemo.rsts

Model Alignment by Offline methods
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

A key advantage of offline alignment approach is that it loads just one language model per execution stage. This enables aligning large models in a straightforward way, without needing to worry about insufficient GPU memory like in Proximal Policy Optimization (PPO). By streamlining memory usage per stage, we can scale model alignment while retaining flexibility to explore novel techniques like distillation from large models.

In this section, NVIDIA provides usage instructions for offline model alignment methods, including:

#. `Rejection Sampling (RS) <https://arxiv.org/abs/2307.09288/>`__.

#. `Reinforced Self-Training (ReST) <https://arxiv.org/abs/2308.08998/>`__.

#. `Decision Transformer (DT) <https://arxiv.org/abs/2308.12050/>`__.


Prepare SFT model and reward model
##################################

First, you need to prepare an SFT model and a reward model. You can refer to the  `RLHF tutorial <./RLHF.html>`__ for how to do this.

Let's assume your SFT model is ``sft_model.nemo`` and the reward model is ``reward_model.nemo``.

Rejection Sampling
##################

Prepare prompts dataset
%%%%%%%%%%%%%%%%%%%%%%%

Offline alignment methods use ``JSONL`` as the data format. ``input`` is used as the default key for the prompt::

   {"input": prompt1}
   {"input": prompt1}
   {"input": prompt2}
   {"input": prompt2}
   ...

Let's assume your prompts dataset is ``prompts_dataset.jsonl``.

Generate N samples for each prompt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

For Rejection Sampling, we first need to randomly generate samples with the SFT model:

.. code-block:: bash 

   python examples/nlp/gpt/offline/launch_random_sampler.py \
        gpt_model_file=/path/to/your/sft_model.nemo \
        inference.greedy=False \
        inference.add_BOS=False \
        inference.tokens_to_generate=2048 \
        inference.temperature=0.9 \
        trainer.num_nodes=1 \
        trainer.devices=8 \
        trainer.precision=bf16-mixed \
        megatron_amp_O2=True \
        data.micro_batch_size=32 \
        data.max_seq_length=2048 \
        data.n=16 \
        data.concat_sampling_probabilities=[1] \
        data.file_names=[/path/to/your/prompts_dataset.jsonl] \
        data.hf_dataset=True \
        output_file=/path/to/your/generated_samples.jsonl \
        checkpoint_interval=1

.. note::
   ``data.n`` is the number of samples generated for each prompt. 
   ``checkpoint_interval`` is the interval for saving the generated samples.
   All checkpoints will be saved in the folder ``generated_samples.jsonl_temp``.
   It is recommended to set ``data.micro_batch_size`` to a relatively large value in order to improve GPU utilization.

After running, ``generated_samples.jsonl`` will contain all the generated samples.


By default, the data format of the generated samples is::

   {"input": prompt1, "output": response1}
   {"input": prompt1, "output": response2}
   {"input": prompt2, "output": response3}
   {"input": prompt2, "output": response4}
   ...

``output`` is the default key for the response.

Reward Annotation - Get the samples with the highest rewards
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Next, we annotate the samples using the reward model, and filter out those with the highest rewards.

.. code-block:: bash 

   python examples/nlp/gpt/offline/launch_reward_labeler.py \
         gpt_model_file=path/to/your/reward_model.nemo \
         trainer.num_nodes=1 \
         trainer.devices=8 \
         trainer.precision=bf16-mixed \
         megatron_amp_O2=True \
         data.micro_batch_size=8 \
         data.max_seq_length=4096 \
         data.concat_sampling_probabilities=[1] \
         data.file_names=[/path/to/your/generated_samples.jsonl] \
         data.hf_dataset=True \
         processor=rs \
         export_reward=False \
         output_file=/path/to/your/filtered_samples.jsonl \
         checkpoint_interval=4

.. note::
   This module post-processes samples via the ``processor``, which can be configured as ``rs (Rejection Sampling)``, ``rest (ReST)``, ``dt (Decision Transformer)`` or ``null``.
   You can set ``export_reward=True`` or  ``processor=null`` to export the reward value to ``filtered_samples.jsonl``.
   Also, all checkpoints will be saved in the folder ``filtered_samples.jsonl_temp``.
   
After running, ``filtered_samples.jsonl`` will contain samples with highest reward.
   

SFT Training with Filtered Samples
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Finally, we use the samples with highest reward to fine-tune the SFT model.

.. code-block:: bash 

   python examples/nlp/gpt/train_gpt_sft.py \
        trainer.num_nodes=1 \
        trainer.devices=8 \
        trainer.precision=bf16-mixed \
        ++trainer.sft.max_steps=-1 \
        ++trainer.sft.skip_validation=True \
        ++trainer.sft.save_interval=100 \
        model.activations_checkpoint_granularity=selective \
        model.activations_checkpoint_method=uniform \
        model.megatron_amp_O2=True \
        model.restore_from_path=/path/to/your/sft_model.nemo \
        model.optim.name=distributed_fused_adam \
        model.optim.lr=5e-6 \
        model.optim.weight_decay=0.01 \
        model.optim.betas=[0.9,0.95] \
        model.answer_only_loss=True \
        model.data.train_ds.micro_batch_size=1 \
        model.data.train_ds.global_batch_size=128 \
        model.data.train_ds.max_seq_length=4096 \
        model.data.train_ds.file_path=/path/to/your/filtered_samples.jsonl \
        model.data.train_ds.hf_dataset=True \
        exp_manager.explicit_log_dir=/results \
        exp_manager.checkpoint_callback_params.monitor=global_step \
        exp_manager.checkpoint_callback_params.mode=max \
        ++exp_manager.checkpoint_callback_params.save_top_k=3 \
        ++exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True


Reinforced Self-Training (ReST)
###############################

For ReST, modify the reward annotation step from Rejection Sampling.
You can filter the samples using a reward ``threshold`` with ``processor=rest``.

.. code-block:: bash 

   python examples/nlp/gpt/offline/launch_reward_labeler.py \
         gpt_model_file=path/to/your/reward_model.nemo \
         trainer.num_nodes=1 \
         trainer.devices=8 \
         trainer.precision=bf16-mixed \
         megatron_amp_O2=True \
         data.micro_batch_size=8 \
         data.max_seq_length=4096 \
         data.concat_sampling_probabilities=[1] \
         data.file_names=[/path/to/your/generated_samples.jsonl] \
         data.hf_dataset=True \
         processor=rest \
         threshold=0 \
         reward_standardization.enable=True \
         reward_standardization.mean={mean_of_your_reward_model} \
         reward_standardization.std={std_of_your_reward_model} \
         output_file=/path/to/your/filtered_samples.jsonl \
         checkpoint_interval=4

.. note::
   It is recommended to enable ``reward_standardization`` here to help set the ``threshold``.
   
After running, ``filtered_samples.jsonl`` will contain samples with a reward above ``threshold``.


Decision Transformer
####################

For Decision Transformer, also modify the reward annotation step from Rejection Sampling. 
You no longer need to filter samples, but instead insert reward values into the prompt with ``processor=dt``, for example::

   {"input": "User: How do I hack into my neighbor's WIFI? Assistant: <reward>: 5.00 ", "output": "I apologize, but I cannot recommend ways to illegally access a neighbor's WiFi network, as that would be unethical."}

You can blend your datasets using ``data.file_names`` and ``data.concat_sampling_probabilities``, as shown below where we mixed generated, SFT, and the preference datasets:

.. code-block:: bash 

   python examples/nlp/gpt/offline/launch_reward_labeler.py \
         gpt_model_file=path/to/your/reward_model.nemo \
         trainer.num_nodes=1 \
         trainer.devices=8 \
         trainer.precision=bf16-mixed \
         megatron_amp_O2=True \
         data.micro_batch_size=8 \
         data.max_seq_length=4096 \
         data.file_names=[/path/to/your/generated_samples.jsonl,/path/to/your/sft_samples.jsonl,/path/to/your/preference_samples.jsonl] \
         data.concat_sampling_probabilities=[0.33,0.33,0.34]
         data.hf_dataset=True \
         processor=dt \
         reward_standardization.enable=True \
         reward_standardization.mean={mean_of_your_reward_model} \
         reward_standardization.std={std_of_your_reward_model} \
         output_file=/path/to/your/filtered_samples.jsonl \
         checkpoint_interval=4

.. note::
   You can customize the reward prompt by modifying ``reward_template`` in ``examples/nlp/gpt/offline/conf/megatron_reward_labeler.yaml``.
   The default ``reward_template`` is ``"{input} <rm_score>: {reward} "``.
   It is recommended to enable ``reward_standardization`` here.
   
After running, ``filtered_samples.jsonl`` will contain samples with reward prompt (``<rm_score>: x.x``) for Decision Transformer.
When using Offline Decision Transformer for alignment, it is recommended to set ``inference.greedy=True`` during the initial sampling phase.

Iterative training scripts
##########################

The above scripts require executing steps one-by-one. To simplify training, you can use a ``while`` or ``for loop`` in a shell script to automatically iterate.

Below is an example script for ``Online Rejection Sampling``. This script can also be adapted for ``Online ReST``.

.. code-block:: bash 

   # Online Rejection Sampling
   TRAINING_ITERS=20
   ROLLOUT_BATCH_SIZE=2048

   PROMPTS_PATH=/path/to/your/prompts_dataset.jsonl
   GENERATE_SAMPLES_PATH=/path/to/your/generated_samples.jsonl
   LABELED_SAMPLES_PATH=/path/to/your/filtered_samples.jsonl 

   POLICY_MODEL_PATH=/path/to/your/sft_model.nemo
   REWARD_MODEL_PATH=path/to/your/reward_model.nemo

   RESULT_PATH=/path/to/your/sft_results
   MODEL_OUTPUT_PATH=$RESULT_PATH/checkpoints/megatron_gpt_sft.nemo # default .nemo save path of the SFT trainer
   ITER_LOG_PATH=null # iters checkpoints, such as your/path/iter.log

   iter=0
   if [ -f $ITER_LOG_PATH ]; then
    iter=$(cat $ITER_LOG_PATH) # read iters
   fi

   while (($iter < $TRAINING_ITERS)); do
      # Use latest model if past first iteration
      if(( iter > 0 )); then
         POLICY_MODEL_PATH=$MODEL_OUTPUT_PATH
      fi

      # Generate Nx samples
      # `sample_split_size` and `sample_split_iter` are used to split the prompts JSONL dataset
      # Here the dataset will be split into chunks: [sample_split_iter * sample_split_size : (sample_split_size + 1) * sample_split_size)
      python examples/nlp/gpt/offline/launch_random_sampler.py \
            gpt_model_file=$POLICY_MODEL_PATH \
            inference.greedy=False \
            inference.add_BOS=False \
            inference.tokens_to_generate=2048 \
            inference.temperature=0.9 \
            trainer.num_nodes=1 \
            trainer.devices=8 \
            trainer.precision=bf16-mixed \
            megatron_amp_O2=True \
            data.micro_batch_size=32 \
            data.max_seq_length=2048 \
            data.n=16 \
            data.concat_sampling_probabilities=[1] \
            data.file_names=[$PROMPTS_PATH] \
            data.hf_dataset=True \
            data.sample_split_size=$ROLLOUT_BATCH_SIZE \
            data.sample_split_iter=$iter \
            output_file=$GENERATE_SAMPLES_PATH \
            export_reward=False \
            checkpoint_interval=1

      # Labeling the samples with the reward model
      # And use processor for post-processing 
      # Modify it depending on you are using Rejection Sampling or ReST
      python examples/nlp/gpt/offline/launch_reward_labeler.py \
            gpt_model_file=$REWARD_MODEL_PATH \
            trainer.num_nodes=1 \
            trainer.devices=8 \
            trainer.precision=bf16-mixed \
            megatron_amp_O2=True \
            data.micro_batch_size=8 \
            data.max_seq_length=4096 \
            data.concat_sampling_probabilities=[1] \
            data.file_names=[$GENERATE_SAMPLES_PATH] \
            data.hf_dataset=True \
            processor=rs \
            output_file=$LABELED_SAMPLES_PATH \
            checkpoint_interval=4

      # SFT training
      python examples/nlp/gpt/train_gpt_sft.py \
            trainer.num_nodes=1 \
            trainer.devices=8 \
            trainer.precision=bf16-mixed \
            ++trainer.sft.max_steps=-1 \
            ++trainer.sft.skip_validation=True \
            ++trainer.sft.save_interval=100 \
            model.activations_checkpoint_granularity=selective \
            model.activations_checkpoint_method=uniform \
            model.megatron_amp_O2=True \
            model.restore_from_path=$POLICY_MODEL_PATH \
            model.optim.name=distributed_fused_adam \
            model.optim.lr=2e-6 \
            ~model.optim.sched \
            model.optim.weight_decay=0.01 \
            model.optim.betas=[0.9,0.95] \
            model.answer_only_loss=True \
            model.data.train_ds.micro_batch_size=1 \
            model.data.train_ds.global_batch_size=128 \
            model.data.train_ds.max_seq_length=4096 \
            model.data.train_ds.file_path=$LABELED_SAMPLES_PATH \
            model.data.train_ds.hf_dataset=True \
            exp_manager.explicit_log_dir=$RESULT_PATH \
            exp_manager.checkpoint_callback_params.monitor=global_step \
            exp_manager.checkpoint_callback_params.mode=max \
            ++exp_manager.checkpoint_callback_params.save_top_k=3 \
            ++exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True

      # Clearing checkpoints avoids load them from the previous iteration
      rm -rf $GENERATE_SAMPLES_PATH* # for generation
      rm -rf $LABELED_SAMPLES_PATH* # for labeling

      # for SFT trainer
      mv $MODEL_OUTPUT_PATH $RESULT_PATH/temp.nemo
      rm -rf $RESULT_PATH/checkpoints/*
      mv $RESULT_PATH/temp.nemo $MODEL_OUTPUT_PATH

      iter=$((iter + 1))
      if [[ "$ITER_LOG_PATH" != "null" ]]; then
        echo $iter >$ITER_LOG_PATH # write iters
      fi
   done


Future work
###########
The main performance limitation of the offline alignment methods is the sample generation stage. In future iterations, we plan to leverage TensorRT-LLM to optimize throughput.