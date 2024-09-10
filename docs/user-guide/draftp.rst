.. include:: /content/nemo.rsts

.. _model-aligner-draftp:

Fine-tuning Stable Diffusion with DRaFT+
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

In this tutorial, we will go through the step-by-step guide for fine-tuning Stable Diffusion model using DRaFT+ algorithm by NVIDIA. 
DRaFT+ is an improvement over the `DRaFT <https://arxiv.org/pdf/2309.17400.pdf>`__ algorithm by alleviating the mode collapse and improving diversity through regularization. 
For more technical details on the DRaFT+ algorithm, check out our technical blog.


Data Input for running DRaFT+
#############################

The data for running DRaFT+ should be a ``.tar`` file consisting of a plain prompt. You can generate a tarfile from a ``.txt``
file containing the prompts separated by new lines, such as following format::

    prompt1
    prompt2
    prompt3
    prompt4
    ...

Use the following script to download and save the prompts from the `Pick a pic <https://huggingface.co/datasets/yuvalkirstain/pickapic_v1_no_images>`__ dataset:

    .. code-block:: bash 

        from datasets import load_dataset

        dataset = load_dataset("yuvalkirstain/pickapic_v1_no_images")
        captions = dataset['train']['caption']  
        file_path = # path to save as a .txt file
        with open(file_path, 'w') as file:
            for caption in captions:
                file.write(caption + '\n')

You can then run the following snipet to convert it to a ``.tar`` file:

   .. code-block:: bash 

        import webdataset as wds

        txt_file_path = # Path for the input txt file
        tar_file_name = # Path for the output tar file

        with open(txt_file_path, 'r') as f:
            prompts = f.readlines()
        prompts = [item.strip() for item in prompts]
        sink = wds.TarWriter(tar_file_name)
        for index, line in enumerate(prompts):
            sink.write({
                "__key__": "sample%06d" % index,
                "txt": line.strip(),
            })
        sink.close()

Reward Model
############

Currently, we only have support for `Pickscore-style <https://arxiv.org/pdf/2305.01569.pdf>`__ reward models (PickScore/HPSv2). Since Pickscore is a CLIP-based model, 
you can use the `conversion script <https://github.com/NVIDIA/NeMo/blob/main/examples/multimodal/vision_language_foundation/clip/convert_external_clip_to_nemo.py>`__ from NeMo to convert it from huggingface to NeMo.

DRaFT+ Training
###############

To launch reward model training, you must have checkpoints for `UNet <https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/unet>`__ and 
`VAE <https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/vae>`__ of a trained Stable Diffusion model and a checkpoint for the Reward Model. 

.. tab-set::

    .. tab-item:: Terminal
        :sync: key1

         To run DRaFT+ on the terminal directly:

         .. code-block:: bash 

            GPFS="/path/to/nemo-aligner-repo"
            TRAIN_DATA_PATH="/path/to/train_dataset.tar"
            UNET_CKPT="/path/to/unet_weights.ckpt"
            VAE_CKPT="/path/to/vae_weights.bin"
            RM_CKPT="/path/to/reward_model.nemo"
            DRAFTP_SCRIPT="train_sd_draftp.py"       # or train_sdxl_draftp.py 

            torchrun --nproc_per_node=2 ${GPFS}/examples/mm/stable_diffusion/${DRAFTP_SCRIPT} \
               trainer.num_nodes=1 \
               trainer.devices=2 \
               model.micro_batch_size=1 \
               model.global_batch_size=8 \
               model.kl_coeff=0.2 \
               model.optim.lr=0.0001 \
               model.unet_config.from_pretrained=${UNET_CKPT} \
               model.first_stage_config.from_pretrained=${VAE_CKPT} \
               rm.model.restore_from_path=${RM_CKPT} \
               model.data.train.webdataset.local_root_path=${TRAIN_DATA_PATH} \
               exp_manager.create_wandb_logger=False \
               exp_manager.explicit_log_dir=/results

    .. tab-item:: Slurm
        :sync: key4

         To run DRaFT+ using Slurm. The script below uses 1 node:

         .. code-block:: bash 

            #!/bin/bash
            #SBATCH -A <<ACCOUNT NAME>>
            #SBATCH -p <<PARTITION NAME>>
            #SBATCH -N 4
            #SBATCH -t 4:00:00
            #SBATCH -J <<JOB NAME>>
            #SBATCH --ntasks-per-node=8
            #SBATCH --exclusive
            #SBATCH --overcommit

            GPFS="/path/to/nemo-aligner-repo"

            GPFS="/path/to/nemo-aligner-repo"
            TRAIN_DATA_PATH="/path/to/train_dataset.tar"
            UNET_CKPT="/path/to/unet_weights.ckpt"
            VAE_CKPT="/path/to/vae_weights.bin"
            RM_CKPT="/path/to/reward_model.nemo"

            PROJECT="<<WANDB PROJECT>>"

            CONTAINER=<<<CONTAINER>>> # use the latest NeMo Training container, Aligner will work there

            MOUNTS="--container-mounts=MOUNTS" # mounts

            RESULTS_DIR="/path/to/result_dir"

            OUTFILE="${RESULTS_DIR}/rm-%j_%t.out"
            ERRFILE="${RESULTS_DIR}/rm-%j_%t.err"
            mkdir -p ${RESULTS_DIR}

            MOUNTS="--container-mounts=MOUNTS" # mounts

            DRAFTP_SCRIPT="train_sd_draftp.py"       # or train_sdxl_draftp.py 

            read -r -d '' cmd <<EOF
            echo "*******STARTING********" \
            && echo "---------------" \
            && echo "Starting training" \
            && cd ${GPFS} \
            && export PYTHONPATH="${GPFS}:${PYTHONPATH}" \
            && export HYDRA_FULL_ERROR=1 \
            && python -u ${GPFS}/examples/mm/stable_diffusion/${DRAFTP_SCRIPT} \
               trainer.num_nodes=1 \
               trainer.devices=8 \
               model.micro_batch_size=2 \
               model.global_batch_size=16 \
               model.kl_coeff=0.2 \
               model.optim.lr=0.0001 \
               model.unet_config.from_pretrained=${UNET_CKPT} \
               model.first_stage_config.from_pretrained=${VAE_CKPT} \
               rm.model.restore_from_path=${RM_CKPT} \
               model.data.webdataset.local_root_path=${TRAIN_DATA_PATH} \
               exp_manager.explicit_log_dir=${RESULTS_DIR} \
               exp_manager.create_wandb_logger=True \
               exp_manager.wandb_logger_kwargs.name=${NAME} \
               exp_manager.wandb_logger_kwargs.project=${PROJECT}
            EOF

            srun -o $OUTFILE -e $ERRFILE --container-image=$CONTAINER $MOUNTS bash -c "${cmd}"
            set +x


.. note::
   For more info on DRaFT+ hyperparameters please see the model config files (for SD and SDXL respectively):
   
    ``NeMo-Aligner/examples/mm/stable_diffusion/conf/draftp_sd.yaml``
    ``NeMo-Aligner/examples/mm/stable_diffusion/conf/draftp_sdxl.yaml``

DRaFT+ Results
%%%%%%%%%%%%%%

Once you have completed fine-tuning Stable Diffusion with DRaFT+, you can run inference on your saved model using the `sd_infer.py <https://github.com/NVIDIA/NeMo/blob/main/examples/multimodal/text_to_image/stable_diffusion/sd_infer.py>`__ 
and `sd_lora_infer.py <https://github.com/NVIDIA/NeMo/blob/main/examples/multimodal/text_to_image/stable_diffusion/sd_lora_infer.py>`__  scripts from the NeMo codebase. The generated images with the fine-tuned model should have 
better prompt alignment and aesthetic quality.

User controllable finetuning with Annealed Importance Guidance (AIG)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

AIG provides the inference-time flexibility to interpolate between the base Stable Diffusion model (with low rewards and high diversity) and DRaFT-finetuned model (with high rewards and low diversity) to obtain images with high rewards and high diversity. AIG inference is easily done by specifying comma-separated `weight_type` strategies to interpolate between the base and finetuned model.

.. tab-set::
    .. tab-item:: AIG on Stable Diffusion XL 
        :sync: key2

        Weight type of `base` uses the base model for AIG, `draft` uses the finetuned model (no interpolation is done in either case).
        Weight type of the form `power_<float>` interpolates using an exponential decay specified in the AIG paper.

        To run AIG inference on the terminal directly:

         .. code-block:: bash 

            NUMNODES=1
            LR=${LR:=0.00025}
            INF_STEPS=${INF_STEPS:=25}
            KL_COEF=${KL_COEF:=0.1}
            ETA=${ETA:=0.0}
            DATASET=${DATASET:="pickapic50k.tar"}
            MICRO_BS=${MICRO_BS:=1}
            GRAD_ACCUMULATION=${GRAD_ACCUMULATION:=4}
            PEFT=${PEFT:="sdlora"}
            NUM_DEVICES=${NUM_DEVICES:=8}
            GLOBAL_BATCH_SIZE=$((MICRO_BS*NUM_DEVICES*GRAD_ACCUMULATION*NUMNODES))
            LOG_WANDB=${LOG_WANDB:="False"}

            echo "additional kwargs: ${ADDITIONAL_KWARGS}"

            WANDB_NAME=SDXL_Draft_annealing
            WEBDATASET_PATH=/path/to/${DATASET}

            CONFIG_PATH="/opt/nemo-aligner/examples/mm/stable_diffusion/conf"
            CONFIG_NAME=${CONFIG_NAME:="draftp_sdxl"}
            UNET_CKPT="/path/to/unet.ckpt"
            VAE_CKPT="/path/to/vae.ckpt"
            RM_CKPT="/path/to/reward_model.nemo"
            PROMPT=${PROMPT:="Bananas growing on an apple tree"}
            DIR_SAVE_CKPT_PATH=/path/to/explicit_log_dir

            if [ ! -z "${ACT_CKPT}" ]; then
                ACT_CKPT="model.activation_checkpointing=$ACT_CKPT "
                echo $ACT_CKPT
            fi

            EVAL_SCRIPT=${EVAL_SCRIPT:-"anneal_sdxl.py"}
            export DEVICE="0,1,2,3,4,5,6,7" && echo "Running DRaFT+ on ${DEVICE}" && export HYDRA_FULL_ERROR=1 
            set -x
            CUDA_VISIBLE_DEVICES="${DEVICE}" torchrun --nproc_per_node=$NUM_DEVICES /opt/nemo-aligner/examples/mm/stable_diffusion/${EVAL_SCRIPT} \
                --config-path=${CONFIG_PATH} \
                --config-name=${CONFIG_NAME} \
                model.optim.lr=${LR} \
                model.optim.weight_decay=0.0005 \
                model.optim.sched.warmup_steps=0 \
                model.sampling.base.steps=${INF_STEPS} \
                model.kl_coeff=${KL_COEF} \
                model.truncation_steps=1 \
                trainer.draftp_sd.max_epochs=5 \
                trainer.draftp_sd.max_steps=10000 \
                trainer.draftp_sd.save_interval=200 \
                trainer.draftp_sd.val_check_interval=20 \
                trainer.draftp_sd.gradient_clip_val=10.0 \
                model.micro_batch_size=${MICRO_BS} \
                model.global_batch_size=${GLOBAL_BATCH_SIZE} \
                model.peft.peft_scheme=${PEFT} \
                model.data.webdataset.local_root_path=$WEBDATASET_PATH \
                rm.model.restore_from_path=${RM_CKPT} \
                trainer.devices=${NUM_DEVICES} \
                trainer.num_nodes=${NUMNODES} \
                rm.trainer.devices=${NUM_DEVICES} \
                rm.trainer.num_nodes=${NUMNODES} \
                +prompt="${PROMPT}" \
                exp_manager.create_wandb_logger=${LOG_WANDB} \
                model.first_stage_config.from_pretrained=${VAE_CKPT} \
                model.first_stage_config.from_NeMo=True \
                model.unet_config.from_pretrained=${UNET_CKPT} \
                model.unet_config.from_NeMo=True \
                $ACT_CKPT \
                exp_manager.wandb_logger_kwargs.name=${WANDB_NAME} \
                exp_manager.resume_if_exists=True \
                exp_manager.explicit_log_dir=${DIR_SAVE_CKPT_PATH} \
                exp_manager.wandb_logger_kwargs.project=${PROJECT} +weight_type='draft,base,power_2.0'



