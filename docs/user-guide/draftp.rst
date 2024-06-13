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

Currently, we only have support for `Pickscore <https://arxiv.org/pdf/2305.01569.pdf>`__ reward model. Since Pickscore is a CLIP-based model, 
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

            torchrun --nproc_per_node=2 ${GPFS}/examples/mm/stable_diffusion/train_sd_draftp.py \
               trainer.num_nodes=1 \
               trainer.devices=2 \
               model.micro_batch_size=1 \
               model.global_batch_size=8 \
               model.kl_coeff=0.2 \
               model.optim.lr=0.0001 \
               model.unet_config.from_pretrained=${UNET_CKPT} \
               model.first_stage_config.from_pretrained=${VAE_CKPT} \
               rm.model.restore_from_path=${RM_CKPT} \
               model.data.trian.webdataset.local_root_path=${TRAIN_DATA_PATH} \
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

            read -r -d '' cmd <<EOF
            echo "*******STARTING********" \
            && echo "---------------" \
            && echo "Starting training" \
            && cd ${GPFS} \
            && export PYTHONPATH="${GPFS}:${PYTHONPATH}" \
            && export HYDRA_FULL_ERROR=1 \
            && python -u ${GPFS}/examples/nlp/gpt/train_reward_model.py \
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
   For more info on DRaFT+ hyperparameters please see the model config file:
   
    ``NeMo-Aligner/examples/mm/stable_diffusion/conf/draftp_sd.yaml``

DRaFT+ Results
%%%%%%%%%%%%%%

Once you have completed fine-tuning Stable Diffusion with DRaFT+, you can run inference on your saved model using the `sd_infer.py <https://github.com/NVIDIA/NeMo/blob/main/examples/multimodal/text_to_image/stable_diffusion/sd_infer.py>`__ 
and `sd_lora_infer.py <https://github.com/NVIDIA/NeMo/blob/main/examples/multimodal/text_to_image/stable_diffusion/sd_lora_infer.py>`__  scripts from the NeMo codebase. The generated images with the fine-tuned model should have 
better prompt alignment and aesthetic quality.