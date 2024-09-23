
model_path=

def submit_fake_nemo_job(model_path, ckpt, new_ckpt_name):
    cmd = f"""
    mkdir -p "{model_path}/checkpoints/{ckpt}/model_weights/";
    rm -r {model_path}/checkpoints/{ckpt}/optimizer*;
    mv {model_path}/checkpoints/{ckpt}/model.* {model_path}/checkpoints/{ckpt}/model_weights/;
    mv {model_path}/checkpoints/{ckpt}/common.pt {model_path}/checkpoints/{ckpt}/model_weights/;
    mv {model_path}/checkpoints/{ckpt}/metadata.json {model_path}/checkpoints/{ckpt}/model_weights/;
    cp {args.base_model_path}/*.yaml "{model_path}/checkpoints/{ckpt}/";
    cp {args.base_model_path}/*.model "{model_path}/checkpoints/{ckpt}/";
    mv {model_path}/checkpoints/{ckpt} {model_path}/checkpoints/{new_ckpt_name};
    """

/lustre/fsw/portfolios/llmservice/users/zhilinw/models/llama31_70b_instruct_regression_helpsteer_v11_0_to_4_helpfulness_only_to_bt_weighted_shuffled_all_weights_1_epochs_constant_lr_1e-6_step_80