.. include:: /content/nemo.rsts

.. include:: nemoaligner.rsts

.. toctree::
   :maxdepth: 2

   reinforce.rst
   sft.rst
   knowledge-distillation.rst
   dpo.rst
   rlhf.rst
   steerlm.rst
   steerlm2.rst
   rs.rst
   spin.rst
   draftp.rst
   cai.rst

:ref:`Prerequisite Obtaining a Pre-Trained Model <prerequisite>`
   This section provides instructions on how to download pre-trained LLMs in .nemo format. The following section will use these base LLMs for further fine-tuning and alignment. 

:ref:`Model Alignment by REINFORCE <nemo-aligner-reinforce>`
   In this tutorial, we will guide you through the process of aligning a NeMo Framework model using REINFORCE. This method can be applied to various models, including LLaMa2 and Mistral, with our scripts functioning consistently across different models.

:ref:`Model Alignment by Supervised Fine-Tuning (SFT) <nemo-aligner-sft>`
   In this section, we walk you through the most straightforward alignment method. We use a supervised dataset in the prompt-response pairs format to fine-tune the base model according to the desired behavior.

:ref:`Supervised Fine-Tuning (SFT) with Knowledge Distillation <nemo-aligner-knowledge-distillation>`
   In this section, we walk through a variation of SFT using Knowledge Distillation where we train a smaller "student" model using a larger "teacher" model.

:ref:`Model Alignment by DPO, RPO and IPO <nemo-aligner-dpo>`
   DPO, RPO, and IPO are simpler alignment methods compared to RLHF. DPO introduces a novel parameterization of the reward model in RLHF, which allows us to extract the corresponding optimal policy. Similarly, RPO and IPO provide alternative parameterizations or optimization strategies, each contributing unique approaches to refining model alignment.

:ref:`Model Alignment by RLHF <nemo-aligner-rlhf>`
   RLHF is the next step up in alignment and is still responsible for most state-of-the-art chat models. In this section, we walk you through the process of RLHF alignment, including training a reward model and RLHF training with the  PPO algorithm.
 
:ref:`Model Alignment by SteerLM Method <nemo-aligner-steerlm>`
   SteerLM is a novel approach developed by NVIDIA. SteerLM simplifies alignment compared to RLHF. It is based on SFT, but allows user-steerable AI by enabling you to adjust attributes at inference time.

:ref:`Model Alignment by SteerLM 2.0 Method <nemo-aligner-steerlm2>`
   SteerLM 2.0 is an extension to SteerLM method that introduces an iterative training procedure to explicitly enforce the generated responses to follow the desired attribute distribution.

:ref:`Model Alignment by Rejection Sampling (RS) <nemo-aligner-rs>`
   RS is a simple online alignment algorithm. In RS, the policy model generates several responses. These responses are assigned a score by the reward model, and the highest scoring responses are used for SFT. 

:ref:`Fine-tuning Stable Diffusion with DRaFT+ <nemo-aligner-draftp>`
   DRaFT+ is an algorithm for fine-tuning text-to-image generative diffusion models. It achieves this by directly backpropagating through a reward model. This approach addresses the mode collapse issues from the original DRaFT algorithm and improves diversity through regularization. 

:ref:`Constitutional AI: Harmlessness from AI Feedback <nemo-aligner-cai>`
   CAI, an alignment method developed by Anthropic, enables the incorporation of AI feedback for aligning LLMs. This feedback is grounded in a small set of principles (referred to as the ‘Constitution’) that guide the model toward desired behaviors, emphasizing helpfulness, honesty, and harmlessness.

.. list-table:: Algorithm vs. (NLP) Models
   :widths: auto
   :header-rows: 1
   :stub-columns: 1

   * - Algorithm
     - TRTLLM Accelerated
     - GPT 2B
     - LLaMA2
     - LLaMA3
     - Mistral
     - Nemotron-4
     - Mixtral
   * - :ref:`REINFORCE <nemo-aligner-reinforce>`
     - Yes
     - Yes
     - Yes
     - Yes (✓)
     - Yes
     - Yes
     - 
   * - :ref:`SFT <nemo-aligner-sft>`
     - 
     - Yes (✓)
     - Yes
     - Yes
     - Yes
     - Yes (✓)
     - 
   * - :ref:`SFT with Knowledge Distillation <nemo-aligner-knowledge-distillation>`
     - 
     - Yes (✓)
     - Yes
     - Yes
     - Yes
     - Yes
     - 
   * - :ref:`DPO <nemo-aligner-dpo>`
     - 
     - Yes (✓)
     - Yes
     - Yes
     - Yes
     - Yes (✓)
     - In active development
   * - :ref:`RLHF <nemo-aligner-rlhf>`
     - Yes
     - Yes
     - Yes
     - Yes (✓)
     - Yes
     - Yes (✓)
     - 
   * - :ref:`SteerLM <nemo-aligner-steerlm>`
     - 
     - Yes
     - Yes (✓)
     - Yes
     - Yes
     - Yes
     - 
   * - :ref:`SteerLM 2.0 <nemo-aligner-steerlm2>`
     - 
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - 
   * - :ref:`Rejection Sampling <nemo-aligner-rs>`
     - 
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - 
   * - :ref:`CAI <nemo-aligner-cai>`
     - 
     - Yes
     - Yes
     - Yes
     - Yes (✓)
     - Yes
     - 

.. list-table:: Algorithm vs. (Multimodal) Models
   :widths: auto
   :header-rows: 1
   :stub-columns: 1

   * - Algorithm
     - Stable Diffusion
   * - :ref:`Draft+ <nemo-aligner-draftp:>`
     - Yes (✓)

.. note::

   * (✓): Indicates the model is verified to work with the algorithm. Models without this demarcation are expected to work but have not been formally verified yet.

Hardware Requirements
#####################

NeMo-Aligner is powered by other NVIDIA libraries that support several NVIDIA GPUs.
NeMo-Aligner is tested on H100 but also works on A100. Several tutorials assume 80GB VRAM,
so if you are following along with GPUs with 40GB, adjust your config accordingly.

Examples of config adjustments are increasing node count, introducing more tensor/pipeline
parallelism, lowering batch size, and increasing gradient accumulation.
