bash ./eval_instruction_following_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr3e-7-bsz512-kl0.05-test-4node-save_freq5/actor_results/checkpoints/llama3-8b-rloo-lr3e-7-bsz512-kl0.05-step50.nemo -o ifeval_bench_results -p llama3_instruct_empty_sys -t 8 -i 1 -n llama3


bash ./eval_instruction_following_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr1e-6-bsz512-kl0.05-4node-joint-training-equalsplit_saveint5-mult1/actor_results/checkpoints/llama3-8b-rloo-joint-equal-lr3e-7-bsz512-kl0.05-step75.nemo -o ifeval_bench_results -p llama3_instruct_empty_sys -t 8 -i 1 -n llama3

bash ./eval_instruction_following_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr1e-6-bsz512-kl0.1-4node-joint-rmmult0-imult1-roseleech1k-roserm/actor_results/checkpoints/llama3-8b-ifeval-roseleech1k-lr1e-6-bsz512-kl0.05-rmmult0.0-ifeval1-step100.nemo -o ifeval_bench_results -p llama3_instruct_empty_sys -t 8 -i 1 -n llama3



bash ./eval_instruction_following_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr1e-6-bsz512-kl0.1-4node-joint-rmmult0.5-imult1-roseleech1k-roserm/actor_results/checkpoints/llama3-8b-ifeval-roseleech1k-lr1e-6-bsz512-kl0.05-rmmult0.5-ifeval1-step130.nemo -o ifeval_bench_results -p llama3_instruct_empty_sys -t 8 -i 1 -n llama3


bash ./eval_instruction_following_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr1e-6-bsz512-kl0.1-4node-joint-rmmult0.1-imult1-roseleech10k-roserm/actor_results/checkpoints/llama3-8b-ifeval-roseleech10k-lr1e-6-bsz512-kl0.1-rmmult0.1-ifeval1-step100.nemo -o ifeval_bench_results -p llama3_instruct_empty_sys -t 8 -i 1 -n llama3


bash ./eval_instruction_following_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/minitron-8b-test4/actor_results/checkpoints/minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step40.nemo -o ifeval_bench_results -p llama3_instruct_empty_sys -t 8 -i 1 -n llama3


bash ./eval_instruction_following_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/minitron-8b-test4/actor_results/checkpoints/minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step70.nemo -o ifeval_bench_results -p extra_sft_empty_sys -t 8 -i 1 -b 20


bash ./eval_instruction_following_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/minitron-8b-gsm8k-llama3.1-70BRM-lr1e-6-kl0.05/actor_results/checkpoints/minitron-helpsteer-gsm8k-kl0.05-lr1e-6-llama3.1RM-step60.nemo -o ifeval_bench_results -p extra_sft_empty_sys -t 8 -i 1 -b 20

bash ./eval_instruction_following_for_nemo.sh -c /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/minitron-8b-test4/actor_results/checkpoints/minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step110.nemo -o ifeval_bench_results -p extra_sft_empty_sys -t 8 -i 1 -b 20
