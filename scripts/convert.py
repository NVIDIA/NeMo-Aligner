

def submit_fake_nemo_job():
    new_ckpt_name="minitron-helpsteer-gsm8k-kl0.05-lr1e-6-llama3.1RM-step60.nemo"
    model_path="/lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/minitron-8b-helpsteer-gsm8k-llama3.1-70BRM-lr1e-6-kl0.05/actor_results"
    # base_model_path="/lustre/fs3/portfolios/llmservice/projects/llmservice_modelalignment_sft/rlhf/checkpoints/community/llama3/8b_instruct_no_target"
    base_model_path="/lustre/fsw/portfolios/llmservice/users/geshen/share/8b_dpo-urban_3.002e-7-kl-1e-3-dpo-loss-rpo_fwd_kl-sft-weight-1e-5_megatron_gpt--val_loss=0.061-step=150-consumed_samples=38400-epoch=0/megatron_gpt--val_loss=0.061-step=150-consumed_samples=38400-epoch=0"
    ckpt="megatron_gpt-step=60-consumed_samples=30720-reinforce_optimization_step=0-epoch=0-val_global_rewards=1.823"
    cmd = f"""
    mkdir -p "{model_path}/checkpoints/{ckpt}/model_weights/";
    rm -r {model_path}/checkpoints/{ckpt}/optimizer*;
    mv {model_path}/checkpoints/{ckpt}/model.* {model_path}/checkpoints/{ckpt}/model_weights/;
    mv {model_path}/checkpoints/{ckpt}/common.pt {model_path}/checkpoints/{ckpt}/model_weights/;
    mv {model_path}/checkpoints/{ckpt}/metadata.json {model_path}/checkpoints/{ckpt}/model_weights/;
    cp {base_model_path}/*.yaml "{model_path}/checkpoints/{ckpt}/";
    cp {base_model_path}/1* "{model_path}/checkpoints/{ckpt}/";
    cp {base_model_path}/0* "{model_path}/checkpoints/{ckpt}/";
    cp {base_model_path}/3* "{model_path}/checkpoints/{ckpt}/";
    cp {base_model_path}/*.model "{model_path}/checkpoints/{ckpt}/";
    mv {model_path}/checkpoints/{ckpt} {model_path}/checkpoints/{new_ckpt_name};
    """
    print(cmd)

submit_fake_nemo_job()