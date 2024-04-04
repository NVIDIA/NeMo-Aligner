.. include:: /content/nemo.rsts

.. include:: ModelAlignment.rsts

:ref:`Prerequisite Obtaining a Pre-Trained Model <prerequisite>`
   This section provides instructions on how to download pre-trained LLMs in .nemo format. The following section will use These base LLMs for further fine-tuning and alignment. 

:ref:`Model Alignment by Supervised Fine-Tuning (SFT) <model-aligner-sft>`
   In this section, we walk you through the most straightforward alignment method, using a supervised dataset in the prompt-response pairs format, to fine-tune the base model to the desired behavior.

:ref:`Model Alignment by RLHF <model-aligner-rlhf>`
   RLHF is the next step up in alignment and is still responsible for most state-of-the-art chat models. In this section, we walk you through the process of RLHF alignment, including training a reward model and the RLHF training with the  PPO algorithm.
 
:ref:`Model Alignment by SteerLM Method <model-aligner-steerlm>`
   SteerLM is a novel approach developed by the NVIDIA. SteerLM simplifies alignment compared to RLHF. It is based on SFT but allows user-steerable AI by enabling you to adjust attributes at inference time.

:ref:`Model Alignment by Direct Preference Optimisation (DPO) <model-aligner-dpo>`
   DPO is a simpler alignment method compared to RLHF. DPO introduces a novel parameterization of the reward model in RLHF. This parameterization allows us to extract the corresponding optimal 

:ref:`Fine-tuning Stable Diffusion with DRaFT+ <model-aligner-draftp>`
   DRaFT+ is an algorithm for fine-tuning text-to-image generative diffusion models by directly backpropagating through a reward model which alleviates the mode collapse issues from DRaFT algorithm and improves diversity through regularization. 

.. toctree::
   :maxdepth: 4
   :titlesonly:

   SFT.rst
   RLHF.rst
   SteerLM.rst
   DPO.rst
   SPIN.rst
   DRaFTP.rst
