.. include:: /content/nemo.rsts

.. _nemo-aligner-eval:

Evaluating a Trained Model
@@@@@@@@@@@@@@@@@@@@@@@@@@

After training a model, you may want to run evaluation to understand how the model performs on unseen tasks. You can use Eleuther AI's `Language Model Evaluation Harness <https://github.com/EleutherAI/lm-evaluation-harness>`_
to quickly run a variety of popular benchmarks, including MMLU, SuperGLUE, HellaSwag, and WinoGrande.
A full list of supported tasks can be found `here <https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md>`_.

Install LM Evaluation Harness
#############################

Run the following commands inside of a NeMo container to install the LM Evaluation Harness:

.. code-block:: bash

   git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
   cd lm-evaluation-harness
   pip install -e .


Run Evaluations
###############

A detailed description of running evaluation with ``.nemo`` models can be found in Eleuther AI's `documentation <https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#nvidia-nemo-models>`_.
Single- and multi-GPU evaluation is supported. The following is an example of running evaluation on the ``hellaswag`` task using a ``.nemo`` file from NeMo-Aligner.
Note that it is recommended, but not required, to unzip your ``.nemo`` file prior to running evaluations.

.. code-block:: bash

   mkdir unzipped_checkpoint
   tar -xvf /path/to/model.nemo -c unzipped_checkpoint

   PYTHONPATH=/path/to/lm-evaluation-harness:${PYTHONPATH} torchrun --nproc-per-node=8 --no-python  lm_eval --model nemo_lm \
     --model_args path='unzipped_checkpoint',devices=8,tensor_model_parallel_size=8 \
     --tasks hellaswag \
     --batch_size 8
