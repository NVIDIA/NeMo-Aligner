bash eval_all.sh vegan_skunk megatron_gpt_sft extra_sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/gen/megatron_gpt_sft--val_loss=0.469-step=2400-consumed_samples=307200.jsonl 10

bash eval_mixtral_mt_bench.sh vegan_skunk_sft2 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/gen/megatron_gpt_sft--val_loss=0.469-step=2400-consumed_samples=307200.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/

bash eval_mixtral_mt_bench.sh llama38b-lr1e-5-bsz512 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/gen/megatron_gpt-step=107-consumed_samples=54784-reinforce_optimization_step=0-epoch=0-val_global_rewards=6.944.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/

bash eval_mixtral_mt_bench.sh llama38b-lr3e-7-bsz512-kl0.1 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/gen/megatron_gpt-step=44-consumed_samples=22528-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.856.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/

bash eval_mt_bench.sh llama38b-lr3e-7-bsz512-kl0.1 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/gen/megatron_gpt-step=44-consumed_samples=22528-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.856.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/

bash eval_mixtral_mt_bench.sh llama38b-lr3e-7-bsz512-kl0.1-step25 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/gen/megatron_gpt-step=25-consumed_samples=12800-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.692.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/

bash eval_mt_bench.sh llama38b-lr3e-7-bsz512-kl0.1-step50 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/gen/megatron_gpt-step=50-consumed_samples=25600-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.728.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/


bash eval_mt_bench.sh llama38b-lr3e-7-bsz512-kl0.05-step75 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/gen/megatron_gpt-step=25-consumed_samples=12800-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.658.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/


bash eval_mt_bench.sh llama38b-lr3e-7-bsz512-kl0.05-step25 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/gen/megatron_gpt-step=25-consumed_samples=12800-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.658.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/




bash eval_mixtral_mt_bench.sh llama38b-lr3e-7-bsz512-kl0.05-step25 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/gen/megatron_gpt-step=25-consumed_samples=12800-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.658.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/

bash eval_mixtral_mt_bench.sh llama38b-lr3e-7-bsz512-kl0.05-step50 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/gen/megatron_gpt-step=50-consumed_samples=25600-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.925.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/

bash eval_mixtral_mt_bench.sh llama38b-lr3e-7-bsz512-kl0.05-step75 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/gen/megatron_gpt-step=75-consumed_samples=38400-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.976.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/

bash eval_mixtral_mt_bench.sh llama38b-lr3e-7-bsz512-kl0.05-step100 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/gen/megatron_gpt-step=100-consumed_samples=51200-reinforce_optimization_step=0-epoch=0-val_global_rewards=3.223.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/

bash eval_mixtral_mt_bench.sh llama38b-lr3e-7-bsz512-kl0.05-step125 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/gen/megatron_gpt-step=125-consumed_samples=64000-reinforce_optimization_step=0-epoch=0-val_global_rewards=3.361.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/



bash eval_mixtral_mt_bench.sh llama38b-sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/llama3_8b_sft_alpha_nodes8_tp4_3e-6_bs384_rerun_1200.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/

bash eval_mixtral_mt_bench.sh llama38b-sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/llama3_8b_sft_alpha_nodes8_tp4_3e-6_bs384_rerun_1200.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/

bash eval_mixtral_mt_bench.sh llama38b-lr3e-7-bsz512-kl0.1-step25 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/gen/megatron_gpt-step=25-consumed_samples=12800-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.692.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/


bash eval_gpt-4-0125_mt_bench.sh llama38b-lr3e-7-bsz512-kl0.05-step50 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/gen/megatron_gpt-step=50-consumed_samples=25600-reinforce_optimization_step=0-epoch=0-val_global_rewards=2.925.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/

bash eval_gpt-4-0125_mt_bench.sh llama38b-sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/llama3_8b_sft_alpha_nodes8_tp4_3e-6_bs384_rerun_1200.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/

bash eval_gpt-4-0125_mt_bench.sh llama38b-lr3e-7-bsz512-kl0.05-step125 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/gen/megatron_gpt-step=125-consumed_samples=64000-reinforce_optimization_step=0-epoch=0-val_global_rewards=3.361.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/



bash eval_mixtral_mt_bench.sh llama38b-joint-lr3e-7-bsz512-kl0.05-step50-2 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/llama3-8b-rloo-joint-equal-lr3e-7-bsz512-kl0.05-step50.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/

bash eval_gpt-4-0125_mt_bench.sh llama38b-joint-lr3e-7-bsz512-kl0.05-step50-gpt4 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/llama3-8b-rloo-joint-equal-lr3e-7-bsz512-kl0.05-step50.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/

bash eval_mixtral_mt_bench.sh llama38b-ifeval-lr3e-7-bsz512-kl0.05-step100 /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/llama3-8b-rloo-ifeval-only-lr3e-7-bsz512-kl0.05-step100.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/


bash eval_mixtral_mt_bench.sh llama38b-ifeval-lr1e-6-bsz512-kl0.05-step100 /lustre/fs12/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/llama3-8b-ifeval-only-lr1e-6-bsz512-kl0.05-rmmult0.1-ifeval1-step100.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/


bash eval_gpt-4-0125_mt_bench.sh llama38b-ifeval-lr1e-6-bsz512-kl0.05-step100-gpt4 /lustre/fs12/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/llama3-8b-ifeval-only-lr1e-6-bsz512-kl0.05-rmmult0.1-ifeval1-step100.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/


bash eval_mixtral_mt_bench.sh llama38b-roseleech1k-lr1e-6-bsz512-kl0.05-rmmult0.1-ifeval1-step50 /lustre/fs12/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/llama3-8b-ifeval-roseleech1k-lr1e-6-bsz512-kl0.05-rmmult0.1-ifeval1-step50.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/


bash eval_gpt-4-0125_mt_bench.sh llama38b-roseleech10k-lr1e-6-bsz512-kl0.05-rmmult0.1-ifeval1-step50 /lustre/fs12/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/llama3-8b-ifeval-roseleech10k-lr1e-6-bsz512-kl0.1-rmmult0.1-ifeval1-step50.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/


bash eval_mixtral_mt_bench.sh llama38b-roseleech10k-lr1e-6-bsz512-kl0.05-rmmult0.1-ifeval1-step50 /lustre/fs12/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/llama3-8b-ifeval-roseleech10k-lr1e-6-bsz512-kl0.1-rmmult0.1-ifeval1-step100.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/


bash eval_gpt-4-0125_mt_bench.sh minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step40 /lustre/fs12/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step40.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/


bash eval_gpt-4-0125_mt_bench.sh minitron /lustre/fs12/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/minitron.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/

bash eval_gpt-4-0125_mt_bench.sh minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step60 /lustre/fs12/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step60.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/



bash eval_gpt-4-0125_mt_bench.sh llama3.1-70b-instruct /lustre/fs12/portfolios/llmservice/users/abukharin/results/mt_bench_results/evals/70b_instruct.jsonl /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/sft /lustre/fsw/portfolios/llmservice/users/abukharin/results/mt_bench_results/grades/
