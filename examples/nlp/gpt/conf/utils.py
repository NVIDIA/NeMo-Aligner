from megatron.core.optimizer import OptimizerConfig

from nemo_aligner.algorithms.dpo import DPOTrainer
from nemo_aligner.models.nlp.train_gpt_dpo import DPOConfig

def default_dpo_config():
    return DPOConfig(
        ref_policy_kl_penalty=0.2,
        preference_average_log_probs=False,
        gt_reward_scale=1.,
        preference_loss='dpo',
        preference_loss_weight=1,
        sft_loss_weight=0,
    )

## hparams not mapped
## bucket_cap_mb -- passed to NLPDDPStrategy
## overlap_grad_sync
## contiguous_grad_buffer
## ** scheduler **
def default_dpo_optimizer():
    return OptimizerConfig(
        optimizer="adam",
        lr=9e-6,
        weight_decay=0.1,
        bf16=True,
        adam_beta1=0.9,
        adam_beta2=0.98,
        use_distributed_optimizer=True,
        #clip_grad=clip_grad,
    )

## config for NeMo-Aligner trainer
def default_dpo_trainer():
    return functools.partial(
        DPOTrainer,
        limit_val_batches=,
        val_check_interval=0.1,
        gradient_clip_val=1.0,
        max_epochs=1,
        save_interval=100,
        limit_train_batches=1.0,
        max_steps=-1,
    )