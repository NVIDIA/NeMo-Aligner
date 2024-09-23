bash ./eval_coding_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama8b-ifeval-gsm8k-lr1e-6-bsz64-kl0.05-save_interval10-rmmult0.1-shuffled/actor_results/checkpoints/llama3-8b-ifeval-gsm8k-lr1e-6-bsz512-kl0.05-rmmult0.1-ifeval1-step100.nemo -o mbpp -p llama3_instruct_empty_sys -t 8 -i 1 -d mbpp



bash ./eval_coding_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr1e-6-bsz512-kl0.1-4node-joint-rmmult0.5-imult1-roseleech1k-roserm/actor_results/checkpoints/llama3-8b-ifeval-roseleech1k-lr1e-6-bsz512-kl0.05-rmmult0.5-ifeval1-step75.nemo -o mbpp -p llama3_instruct_empty_sys -t 8 -i 1 -d mbpp


bash ./eval_coding_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr1e-6-bsz512-kl0.1-4node-joint-rmmult0.1-imult1-roseleech10k-roserm/actor_results/checkpoints/llama3-8b-ifeval-roseleech10k-lr1e-6-bsz512-kl0.1-rmmult0.1-ifeval1-step50.nemo -o mbpp -p llama3_instruct_empty_sys -t 8 -i 1 -d mbpp


bash ./eval_coding_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr1e-6-bsz512-kl0.1-4node-joint-rmmult0.1-imult1-roseleech10k-roserm/actor_results/checkpoints/llama3-8b-ifeval-roseleech10k-lr1e-6-bsz512-kl0.1-rmmult0.1-ifeval1-step50.nemo -o humaneval -p llama3_instruct_empty_sys -t 8 -i 1 -d humaneval


bash ./eval_coding_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/checkpoints/llama3.1-8b-instruct.nemo -o mbpp -p llama3_instruct_empty_sys -t 8 -i 1 -d mbpp

bash ./eval_coding_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/checkpoints/llama3.1-8b-instruct.nemo -o humaneval -p llama3_instruct_empty_sys -t 8 -i 1 -d humaneval

/lustre/fsw/portfolios/llmservice/users/zhilinw/models/llama31_70b_instruct_regression_helpsteer_v11_0_to_4_helpfulness_only_to_bt_weighted_shuffled_all_weights_1_epochs_constant_lr_1e-6_step_80



bash ./eval_coding_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/minitron-8b-test4/actor_results/checkpoints/minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step70.nemo -o mbpp -p extra_sft_empty_sys -t 8 -i 1 -b 20 -d mbpp

bash ./eval_coding_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/minitron-8b-test4/actor_results/checkpoints/minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step70.nemo -o humaneval -p extra_sft_empty_sys -t 8 -i 1 -b 20 -d humaneval


bash ./eval_coding_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/minitron-8b-gsm8k-llama3.1-70BRM-lr1e-6-kl0.05/actor_results/checkpoints/minitron-gsm8k-kl0.05-lr1e-6-llama3.1RM-step40.nemo -o humaneval -p extra_sft_empty_sys -t 8 -i 1 -b 20 -d humaneval

bash ./eval_coding_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/minitron-8b-gsm8k-llama3.1-70BRM-lr1e-6-kl0.05/actor_results/checkpoints/minitron-gsm8k-kl0.05-lr1e-6-llama3.1RM-step40.nemo -o mbpp -p extra_sft_empty_sys -t 8 -i 1 -b 20 -d mbpp