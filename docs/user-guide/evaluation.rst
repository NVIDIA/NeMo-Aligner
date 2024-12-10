.. include:: /content/nemo.rsts

.. _nemo-aligner-eval:

Evaluate a Trained Model
@@@@@@@@@@@@@@@@@@@@@@@@

After training a model, you may want to run evaluation to understand how the model performs on unseen tasks. You can use Eleuther AI's `Language Model Evaluation Harness <https://github.com/EleutherAI/lm-evaluation-harness>`_
to quickly run a variety of popular benchmarks, including MMLU, SuperGLUE, HellaSwag, and WinoGrande.
A full list of supported tasks can be found `here <https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md>`_.

Install the LM Evaluation Harness
#################################

Run the following commands inside of a NeMo container to install the LM Evaluation Harness:

.. code-block:: bash

   git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
   cd lm-evaluation-harness
   pip install -e .


Run Evaluations
###############

A detailed description of running evaluation with ``.nemo`` models can be found in Eleuther AI's `documentation <https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#nvidia-nemo-models>`_.
Single- and multi-GPU evaluation is supported. The following is an example of running evaluation using 8 GPUs on the ``hellaswag``, ``super_glue``, and ``winogrande`` tasks using a ``.nemo`` file from NeMo-Aligner.
Please note that while it is recommended, you are not required to unzip your .nemo file before running evaluations.

.. code-block:: bash

   mkdir unzipped_checkpoint
   tar -xvf /path/to/model.nemo -c unzipped_checkpoint

   torchrun --nproc-per-node=8 --no-python lm_eval --model nemo_lm \
     --model_args path='unzipped_checkpoint',devices=8,tensor_model_parallel_size=8 \
     --tasks lambada_openai,super-glue-lm-eval-v1,winogrande \
     --batch_size 8
