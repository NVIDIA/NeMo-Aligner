
.. _generic

Training with FP8
#################

NeMo-Aligner supports the FP8 datatype on H100 GPUs via `Transformer Engine <https://github.com/NVIDIA/TransformerEngine>`_ (TE). FP8 enables higher throughput of matrix multiplies and convolutions.
The following table summarizes the FP8-related arguments that can be configured in NeMo-Aligner (`example config setting <https://github.com/NVIDIA/NeMo/blob/2e1814c9f031ad2aeeebad44597365e97253d2c4/examples/nlp/language_modeling/conf/megatron_gpt_config.yaml/#L192-L200>`_). For a more detailed overview, refer to the TE `documentation <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html>`_, specifically the FP8 `format <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html#transformer_engine.common.recipe.Format>`_ and `recipe <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html#transformer_engine.common.recipe.DelayedScaling>`_.

.. list-table:: FP8 arguments
   :widths: 10 20
   :header-rows: 1

   * - Argument
     - Description
   * - fp8
     - The training recipe format for FP8 can be set to either 'hybrid' or 'e4m3', with 'hybrid' being the default. In the 'hybrid' format, activations and weight tensors use the E4M3 format, while gradients use the E5M2 format to meet the additional dynamic range requirements for backward tensors.
   * - fp8_margin
     - The scaling factor for FP8 tensors can be shifted by a factor of $2 ^ {margin}$ using this argument.
   * - fp8_amax_history_len
     - The window size for amax history. The window size determines how many instances of the most recent absolute max values (amaxes) are stored per tensor.
   * - fp8_amax_compute_algo
     - The choice between “max” and “most_recent” specifies how to select an amax value from the given history.
   * - fp8_params
     - Indicates whether to store module-level parameters in FP8. Enabling this option can reduce memory consumption by eliminating the need to store a copy of weights in higher precision for cases where these weights are externally maintained, such as master parameters in the optimizer. For more information, refer to the `fp8_model_init <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.fp8_model_init>`_ API in TE.

Importantly, you must enable Transformer Engine in order to leverage FP8. Make sure ``model.transformer_engine`` is set to ``True`` in your config.


The following code can be appended to your train script to enable FP8 training. Note that the following configuration may need to modified for optimal performance
depending on your model configuration.

.. code-block:: bash

    ++model.fp8=True \
    ++model.fp8_hybrid=True \
    ++model.fp8_e4m3=False \
    ++model.fp8_margin=0 \
    ++model.fp8_amax_history_len=1024 \
    ++model.fp8_amax_compute_algo="max" \
    ++model.fp8_params=True
