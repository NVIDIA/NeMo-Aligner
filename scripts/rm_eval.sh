bash ./generate_mt-judgement_for_rm.sh -c /lustre/fsw/portfolios/llmservice/users/shengyangs/results/rm_llama3_8b_competent-agama_lr_rand_3e-6/checkpoints/megatron_gpt.nemo -o rm_eval -p llama3_instruct_binary -t 8 -i 1 -d evals/rm_evals/data/rloo_mt_bench_step25_r2.658.jsonl

bash ./generate_mt-judgement_for_rm.sh -c /lustre/fsw/portfolios/llmservice/users/shengyangs/results/rm_llama3_8b_competent-agama_lr_rand_3e-6/checkpoints/megatron_gpt.nemo -o rm_eval -p llama3_instruct_binary -t 8 -i 1 -d rewardbench


bash ./generate_mt-judgement_for_rm.sh -c /lustre/fsw/portfolios/llmservice/users/zhilinw/models/Nemotron-4-340B-RM-v1-editted.nemo -o rm_eval -p steerlm_value_reg -t 8 -i 4 -d evals/rm_evals/data/rloo_mt_bench_step125_r3.361.jsonl

bash ./generate_mt-judgement_for_rm.sh -c /lustre/fsw/portfolios/llmservice/users/zhilinw/models/Nemotron-4-340B-RM-v1-editted.nemo -o rm_eval -p steerlm_value_reg -t 8 -i 2 -d evals/rm_evals/data/rloo_ifeval_lmsys_step50.jsonl

bash ./generate_mt-judgement_for_rm.sh -c /lustre/fsw/portfolios/llmservice/users/zhilinw/models/Nemotron-4-340B-RM-v1-editted.nemo -o rm_eval -p steerlm_value_reg -t 8 -i 2 -d evals/rm_evals/data/llama3-8b-rloo-lr3e-7-bsz512-kl0.05-step25.lmsys500.jsonl

bash ./generate_mt-judgement_for_rm.sh -c /lustre/fsw/portfolios/llmservice/users/zhilinw/models/Nemotron-4-340B-RM-v1-editted.nemo -o rm_eval -p steerlm_value_reg -t 8 -i 2 -d evals/rm_evals/data/sft_mt_bench.jsonl



# ALIGN_CONTAINER_RM=/lustre/fsw/portfolios/llmservice/users/yidong/data/models/images/nvidian+nemo+aligner.sqsh

python calculate_rewardbench.py  ../data/reward_bench.jsonl accuracy [1] -100


bash ./generate_mt-judgement_for_rm.sh -c /lustre/fsw/portfolios/llmservice/users/shengyangs/results/rm_llama3_8b_competent-agama_lr_rand_3e-6/checkpoints/megatron_gpt.nemo -o rm_eval -p llama3_instruct_binary -t 8 -i 1 -d evals/rm_evals/data/rloo_mt_bench_step125_r3.361.jsonl

bash ./generate_mt-judgement_for_rm.sh -c /lustre/fsw/portfolios/llmservice/users/shengyangs/results/rm_llama3_8b_competent-agama_lr_rand_3e-6/checkpoints/megatron_gpt.nemo -o rm_eval -p llama3_instruct_binary -t 8 -i 1 -d evals/rm_evals/data/rloo_mt_bench_step125_r3.361.jsonl



bash ./generate_mt-judgement_for_rm.sh -c /lustre/fsw/portfolios/llmservice/users/zhilinw/models/Nemotron-4-340B-RM-v1-editted.nemo -o rm_eval -p steerlm_value_reg -t 8 -i 2 -d evals/rm_evals/data/rloo_ifeval_roseleech1k_lr1e-6_kl0.05_rmmult0.5_step75.jsonl



# scp draco-oci-login-01.draco-oci-iad.nvidia.com:/lustre/fsw/portfolios/llmservice/users/abukharin/results/lmsys/evals/llama3-8b-ifeval-only-lr1e-6-bsz512-kl0.05-rmmult0.1-ifeval1-step* ../data
# scp ../data/rloo_ifeval_prompts_* draco-oci-login-01.draco-oci-iad.nvidia.com:/lustre/fsw/portfolios/llmservice/users/abukharin/steerlm_launcher/evals/rm_evals/data