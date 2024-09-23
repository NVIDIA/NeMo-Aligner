bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/ablation-eloirm-rloo-hh-lr3e-7-bsz64-kl0.01-test-8node-clear-memory-torch/actor_results/checkpoints/megatron_gpt-step=313-consumed_samples=80128-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.396-last -o mt_bench_results -p scale -n tokenizer.model -t 8 -i 1



bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/jiaqiz/results/15b_8T_ct3_vegan-skunk_lr3e-6/checkpoints/megatron_gpt_sft--val_loss=0.469-step=2400-consumed_samples=307200 -o mt_bench_results -p scale -n megatron_2.model -t 8 -i 1


bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/ablation-eloirm-rloo-hh-lr3e-7-bsz64-kl0.01-test-8node-clear-memory-torch/actor_results/checkpoints/megatron_gpt-step=304-consumed_samples=77824-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.444 -o mt_bench_results -p scale -n megatron_2.model -t 8 -i 1


bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr1e-6-bsz512-kl0.01-test-4node-tokenized-v4/actor_results/checkpoints/megatron_gpt-step=107-consumed_samples=54784-reinforce_optimization_step=0-epoch=0-val_global_rewards=6.944 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1


bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.1-test-4node/actor_results/checkpoints/megatron_gpt-step=44-consumed_samples=22528-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.856 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1


bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.1-test-4node-save_freq2/actor_results/checkpoints/megatron_gpt-step=25-consumed_samples=12800-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.692 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1

bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.1-test-4node-save_freq2/actor_results/checkpoints/megatron_gpt-step=25-consumed_samples=12800-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.692 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1

bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.1-test-4node-save_freq2/actor_results/checkpoints/megatron_gpt-step=25-consumed_samples=12800-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.692 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1

bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.1-test-4node-save_freq2/actor_results/checkpoints/megatron_gpt-step=50-consumed_samples=25600-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.728 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1

bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.1-test-4node-save_freq2/actor_results/checkpoints/megatron_gpt-step=75-consumed_samples=38400-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.951 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1



bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.1-test-4node-save_freq2/actor_results/checkpoints/megatron_gpt-step=25-consumed_samples=12800-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.692 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1

bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.1-test-4node-save_freq2/actor_results/checkpoints/megatron_gpt-step=50-consumed_samples=25600-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.728 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1

bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.05-test-4node-save_freq5/actor_results/checkpoints/megatron_gpt-step=25-consumed_samples=12800-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.658 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1

bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.05-test-4node-save_freq5/actor_results/checkpoints/megatron_gpt-step=25-consumed_samples=12800-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.658 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1

bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.05-test-4node-save_freq5/actor_results/checkpoints/megatron_gpt-step=25-consumed_samples=12800-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.658 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1




bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.05-test-4node-save_freq5/actor_results/checkpoints/megatron_gpt-step=25-consumed_samples=12800-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.658 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1

bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.05-test-4node-save_freq5/actor_results/checkpoints/megatron_gpt-step=50-consumed_samples=25600-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.925 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1

bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.05-test-4node-save_freq5/actor_results/checkpoints/megatron_gpt-step=75-consumed_samples=38400-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.976 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1

bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.05-test-4node-save_freq5/actor_results/checkpoints/megatron_gpt-step=100-consumed_samples=51200-reinforce_optimization_step=0-epoch=0-val_global_rewards=3.223 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1

bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.05-test-4node-save_freq5/actor_results/checkpoints/megatron_gpt-step=125-consumed_samples=64000-reinforce_optimization_step=0-epoch=0-val_global_rewards=3.361 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1




bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.01-test-4node-save_freq5/actor_results/checkpoints/megatron_gpt-step=30-consumed_samples=15360-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.820 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1

bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.01-test-4node-save_freq5/actor_results/checkpoints/megatron_gpt-step=55-consumed_samples=28160-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.985 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1

bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.01-test-4node-save_freq5/actor_results/checkpoints/megatron_gpt-step=75-consumed_samples=38400-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.948 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1

bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.01-test-4node-save_freq5/actor_results/checkpoints/megatron_gpt-step=100-consumed_samples=51200-reinforce_optimization_step=0-epoch=0-val_global_rewards=3.178 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1

bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.01-test-4node-save_freq5/actor_results/checkpoints/megatron_gpt-step=124-consumed_samples=63488-reinforce_optimization_step=0-epoch=0-val_global_rewards=3.454 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1


bash generate_mt_responses_for_ckpt.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.1-test-4node-save_freq2/actor_results/checkpoints/megatron_gpt-step=25-consumed_samples=12800-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.692 -o mt_bench_results -p llama3_instruct_empty_sys -n llama3 -t 8 -i 1

bash generate_mt_responses_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/yianz/shared/rpo_analysis/llama3_sft_models/llama3_8b_sft_alpha_nodes8_tp4_3e-6_bs384_rerun_1200.nemo -o mt_bench_results -p llama3_instruct_empty_sys -t 8 -i 1


bash generate_mt_responses_for_nemo2.sh -c /lustre/fsw/portfolios/llmservice/users/yianz/shared/rpo_analysis/llama3_sft_models/llama3_8b_sft_alpha_nodes8_tp4_3e-6_bs384_rerun_1200.nemo -o mt_bench_results -p llama3_instruct_empty_sys -t 8 -i 1


bash generate_mt_responses_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/yianz/shared/rpo_analysis/llama3_sft_models/llama3_8b_sft_alpha_nodes8_tp4_3e-6_bs384_rerun_1200.nemo -o mt_bench_results -p llama3_instruct_empty_sys -t 8 -i 1


bash generate_mt_responses_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.05-test-4node-save_freq5/actor_results/checkpoints/llama3-8b-rloo-lr3e-7-bsz512-kl0.05-step50.nemo -o mt_bench_results -p llama3_instruct_empty_sys -t 8 -i 1

bash generate_df_responses_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.05-test-4node-save_freq5/actor_results/checkpoints/llama3-8b-rloo-lr3e-7-bsz512-kl0.05-step75.nemo -o lmsys -p llama3_instruct_empty_sys -t 8 -i 1 -v lmsys500

bash generate_df_responses_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.05-test-4node-save_freq5/actor_results/checkpoints/llama3-8b-rloo-lr3e-7-bsz512-kl0.05-step125.nemo -o lmsys -p llama3_instruct_empty_sys -t 8 -i 1 -v lmsys500



bash generate_df_responses_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr1e-6-bsz512-kl0.05-4node-joint-training-equalsplit_saveint5-mult1/actor_results/checkpoints/llama3-8b-rloo-joint-equal-lr3e-7-bsz512-kl0.05-step25.nemo -o lmsys -p llama3_instruct_empty_sys -t 8 -i 1 -v lmsys500

bash generate_df_responses_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama8b-ifeval-only-lr1e-6-bsz64-kl0.05-eos-removed-save_interval5-mult1/actor_results/checkpoints/llama3-8b-rloo-ifeval-only-lr3e-7-bsz512-kl0.05-step50.nemo -o lmsys -p llama3_instruct_empty_sys -t 8 -i 1 -v lmsys500

bash generate_df_responses_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama8b-ifeval-only-lr1e-6-bsz64-kl0.05-eos-removed-save_interval5-mult1/actor_results/checkpoints/llama3-8b-rloo-ifeval-only-lr3e-7-bsz512-kl0.05-step100.nemo -o lmsys -p llama3_instruct_empty_sys -t 8 -i 1 -v lmsys500

bash generate_df_responses_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr1e-6-bsz512-kl0.05-4node-joint-training-equalsplit_saveint5-mult1/actor_results/checkpoints/llama3-8b-rloo-joint-equal-lr3e-7-bsz512-kl0.05-step75.nemo -o lmsys -p llama3_instruct_empty_sys -t 8 -i 1 -v lmsys500

bash generate_mt_responses_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama8b-ifeval-only-lr1e-6-bsz64-kl0.05-eos-removed-save_interval5-mult1/actor_results/checkpoints/llama3-8b-rloo-ifeval-only-lr3e-7-bsz512-kl0.05-step100.nemo -o mt_bench_results -p llama3_instruct_empty_sys -t 8 -i 1

bash generate_mt_responses_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr1e-6-bsz512-kl0.05-4node-joint-training-equalsplit_saveint5-mult1/actor_results/checkpoints/llama3-8b-rloo-joint-equal-lr3e-7-bsz512-kl0.05-step50.nemo -o mt_bench_results -p llama3_instruct_empty_sys -t 8 -i 1







bash generate_mt_responses_for_nemo.sh -c /lustre/fs12/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama8b-ifeval-only-lr1e-6-bsz64-kl0.05-eos-removed-save_interval10-rmmult0.1/actor_results/checkpoints/llama3-8b-ifeval-only-lr1e-6-bsz512-kl0.05-rmmult0.1-ifeval1-step100.nemo -o mt_bench_results -p llama3_instruct_empty_sys -t 8 -i 1


bash generate_df_responses_for_nemo.sh -c /lustre/fs12/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama8b-ifeval-only-lr1e-6-bsz64-kl0.05-eos-removed-save_interval10-rmmult0.1/actor_results/checkpoints/llama3-8b-ifeval-only-lr1e-6-bsz512-kl0.05-rmmult0.1-ifeval1-step100.nemo -o lmsys -p llama3_instruct_empty_sys -t 8 -i 1 -v lmsys500


bash generate_df_responses_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr1e-6-bsz512-kl0.1-4node-joint-rmmult0.5-imult1-roseleech1k-roserm/actor_results/checkpoints/llama3-8b-ifeval-roseleech1k-lr1e-6-bsz512-kl0.05-rmmult0.5-ifeval1-step75.nemo -o lmsys -p llama3_instruct_empty_sys -t 8 -i 1 -v lmsys500


bash generate_mt_responses_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr1e-6-bsz512-kl0.1-4node-joint-rmmult0.1-imult1-roseleech10k-roserm/actor_results/checkpoints/llama3-8b-ifeval-roseleech10k-lr1e-6-bsz512-kl0.1-rmmult0.1-ifeval1-step70.nemo -o mt_bench_results -p llama3_instruct_empty_sys -t 8 -i 1 -b 20


bash generate_mt_responses_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr1e-6-bsz512-kl0.1-4node-joint-rmmult0.5-imult1-roseleech10K-roserm/actor_results/checkpoints/llama3-8b-ifeval-roseleech10k-lr1e-6-bsz512-kl0.1-rmmult0.5-ifeval1-step50.nemo -o mt_bench_results -p llama3_instruct_empty_sys -t 8 -i 1 -b 20


bash generate_mt_responses_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/minitron-8b-test4/actor_results/checkpoints/minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step70.nemo -o mt_bench_results -p extra_sft_empty_sys -t 8 -i 1 -b 20

ckpt_name = 
bash generate_mt_responses_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/checkpoints/minitron -o mt_bench_results -p extra_sft_empty_sys -t 8 -i 1 -b 20
