.. include:: /content/nemo.rsts

.. _known_errors_and_resolutions:

Known Errors and Resolutions
@@@@@@@@@@@@@@@@@@@@@@@@@@@@

This section details how to resolve common pitfalls that may arise during the alignment process.

Gated Huggingface Assets
########################

Some NeMo models will pull gated assets like tokenizers from Huggingface. Examples include Llama3 or Llama3.1 tokenizers.

Example error::

    ValueError: Unable to instantiate HuggingFace AUTOTOKENIZER for meta-llama/Meta-Llama-3.1-8B. Exception: You are trying to access a gated repo.
    Make sure to have access to it at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B.
    401 Client Error. (Request ID: Root=<redacted>)

    Cannot access gated repo for url https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/resolve/main/config.json.
    Access to model meta-llama/Llama-3.1-8B is restricted. You must have access to it and be authenticated to access it. Please log in.

Resolution:

1. Request access to the gated repo
2. Create `HF Personal Access Token <https://huggingface.co/settings/tokens>`__
3. Add token to environment:
   a. (opt 1): Store it in ``~/.cache/huggingface/token``
   a. (opt 2): In your script, set it in the environment ``export HF_TOKEN=<REDACTED_PAT>``


Checkpoints with Missing or Unexpected Weights
##############################################

Some NeMo model checkpoints will error when loaded if weights are missing or unexpected.

Example error::

    Traceback (most recent call last):
      File "/workspace/NeMo-Aligner/examples/nlp/gpt/serve_ppo_critic.py", line 119, in <module>
        main()
      ...
      <Traceback shorted for brevity>
      ...
      File "/opt/megatron-lm/megatron/core/dist_checkpointing/strategies/torch.py", line 528, in create_local_plan
        return super().create_local_plan()
      File "/usr/local/lib/python3.10/dist-packages/torch/distributed/checkpoint/default_planner.py", line 196, in create_local_plan
        return create_default_local_load_plan(
      File "/usr/local/lib/python3.10/dist-packages/torch/distributed/checkpoint/default_planner.py", line 315, in create_default_local_load_plan
        raise RuntimeError(f"Missing key in checkpoint state_dict: {fqn}.")
    RuntimeError: Missing key in checkpoint state_dict: model.decoder.layers.self_attention.core_attention._extra_state/shard_0_24.

Resolution:

Add the following to your script invocation:

.. code-block:: bash

   python .... \
   ++model.dist_ckpt_load_strictness=log_all

Visit `Megatron-LM's docs <https://github.com/NVIDIA/Megatron-LM/blob/85cd99bbf54acc1b188b28960155e5c6fcb06686/megatron/core/dist_checkpointing/validation.py#L44>`__ for more information on the options available.
