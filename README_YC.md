
## Step 1: Format the data
python /opt/NeMo-Aligner/examples/nlp/data/steerlm/preprocess_openassistant_data.py --output_directory=data/oasst

## Step 2: Run SFT training

export WANDB_DISABLED=true
export NCCL_IB_DISABLE=1  # 禁用 InfiniBand，如果通信出错时可尝试
export NCCL_P2P_DISABLE=1  # 禁用 P2P 传输，排查问题时有用

export NCCL_DEBUG=INFO
export TMPDIR=/mnt/workspace/yangchao.zhou/opt/models/tmp
MODEL="/mnt/workspace/yangchao.zhou/opt/models/Mistral-NeMo-12B-Instruct/Mistral-NeMo-12B-Instruct.nemo"
TRAIN_DS="/mnt/workspace/yangchao.zhou/opt/data/oasst/train.jsonl"
VALID_DS="/mnt/workspace/yangchao.zhou/opt/data/oasst/val.jsonl"
RESULTS="/mnt/workspace/yangchao.zhou/opt/RESULTS/7B"


python examples/nlp/gpt/train_gpt_sft4linky.py \
   trainer.precision=bf16 \
   trainer.num_nodes=1 \
   trainer.devices=8 \
   trainer.sft.max_steps=-1 \
   trainer.sft.limit_val_batches=40 \
   trainer.sft.val_check_interval=1000 \
   model.tensor_model_parallel_size=1 \
   model.pipeline_model_parallel_size=8 \
   model.megatron_amp_O2=True \
   model.activations_checkpoint_granularity=selective\
   model.restore_from_path=${MODEL} \
   model.optim.lr=5e-6 \
   model.data.chat=True \
   model.data.num_workers=0 \
   model.data.train_ds.micro_batch_size=1 \
   model.data.train_ds.global_batch_size=8 \
   model.data.train_ds.max_seq_length=1024 \
   model.data.train_ds.file_path=${TRAIN_DS} \
   model.data.validation_ds.micro_batch_size=1 \
   model.data.validation_ds.global_batch_size=8 \
   model.data.validation_ds.file_path=${VALID_DS} \
   model.data.validation_ds.max_seq_length=1024 \
   exp_manager.create_wandb_logger=False \
   exp_manager.explicit_log_dir=${RESULTS} \
   exp_manager.wandb_logger_kwargs.project=sft_run \
   exp_manager.wandb_logger_kwargs.name=chat_sft_run \
   exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
   exp_manager.resume_if_exists=True \
   exp_manager.resume_ignore_no_checkpoint=True \
   exp_manager.create_checkpoint_callback=True \
   exp_manager.checkpoint_callback_params.monitor=validation_loss

### 杀掉进程
ps -ef | grep train_gpt_sft4linky
pkill -f train_gpt_sft4linky.py