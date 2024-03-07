.. include:: /content/nemo.rsts

.. include:: ModelAlignment.rsts

Prerequisite
   Obtaining a pre-trained model. This section provides instructions on how to download pre-trained LLMs in .nemo format. The following section will use These base LLMs for further fine-tuning and alignment. 

Model Alignment by Supervised Fine-Tuning (SFT)
   In this section, we walk you through the most straightforward alignment method, using a supervised dataset in the prompt-response pairs format, to fine-tune the base model to the desired behavior.

Model Alignment by RLHF
   RLHF is the next step up in alignment and is still responsible for most state-of-the-art chat models. In this section, we walk you through the process of RLHF alignment, including training a reward model and the RLHF training with the  PPO algorithm.
 
Model Alignment by SteerLM Method
   SteerLM is a novel approach developed by the NVIDIA. SteerLM simplifies alignment compared to RLHF. It is based on SFT but allows user-steerable AI by enabling you to adjust attributes at inference time.

Model Alignment by Direct Preference Optimisation (DPO)
   DPO is a simpler alignment method compared to RLHF. DPO introduces a novel parameterization of the reward model in RLHF. This parameterization allows us to extract the corresponding optimal 

.. toctree::
   :maxdepth: 4
   :titlesonly:

   SFT.rst
   RLHF.rst
   SteerLM.rst
   DPO.rst
