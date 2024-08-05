.. include:: /content/nemo.rsts

.. include:: modelalignment.rsts

.. toctree::
   :maxdepth: 4
   :titlesonly:

   sft.rst
   rlhf.rst
   steerlm.rst
   steerlm2.rst
   dpo.rst
   spin.rst
   draftp.rst
   cai.rst

:ref:`Prerequisite Obtaining a Pre-Trained Model <prerequisite>`
   This section provides instructions on how to download pre-trained LLMs in .nemo format. The following section will use these base LLMs for further fine-tuning and alignment. 

:ref:`Model Alignment by Supervised Fine-Tuning (SFT) <model-aligner-sft>`
   In this section, we walk you through the most straightforward alignment method. We use a supervised dataset in the prompt-response pairs format to fine-tune the base model according to the desired behavior.

:ref:`Model Alignment by RLHF <model-aligner-rlhf>`
   RLHF is the next step up in alignment and is still responsible for most state-of-the-art chat models. In this section, we walk you through the process of RLHF alignment, including training a reward model and RLHF training with the  PPO algorithm.
 
:ref:`Model Alignment by SteerLM Method <model-aligner-steerlm>`
   SteerLM is a novel approach developed by NVIDIA. SteerLM simplifies alignment compared to RLHF. It is based on SFT, but allows user-steerable AI by enabling you to adjust attributes at inference time.

:ref:`Model Alignment by SteerLM 2.0 Method <model-aligner-steerlm2>`
   SteerLM 2.0 is an extenstion to SteerLM method that introduces an iterative training procedure to explicitly enforce the generated responses to follow the desired attribute distribution.

:ref:`Model Alignment by Direct Preference Optimization (DPO) <model-aligner-dpo>`
   DPO is a simpler alignment method compared to RLHF. DPO introduces a novel parameterization of the reward model in RLHF. This parameterization allows us to extract the corresponding optimal policy.

:ref:`Fine-tuning Stable Diffusion with DRaFT+ <model-aligner-draftp>`
   DRaFT+ is an algorithm for fine-tuning text-to-image generative diffusion models. It achieves this by directly backpropagating through a reward model. This approach addresses the mode collapse issues from the original DRaFT algorithm and improves diversity through regularization. 

:ref:`Constitutional AI: Harmlessness from AI Feedback <model-aligner-cai>`
   CAI, an alignment method developed by Anthropic, enables the incorporation of AI feedback for aligning LLMs. This feedback is grounded in a small set of principles (referred to as the ‘Constitution’) that guide the model toward desired behaviors, emphasizing helpfulness, honesty, and harmlessness.
