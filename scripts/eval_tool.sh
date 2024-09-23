bash evals/eval_eval_tool.sh --nemo_file /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama8b-ifeval-gsm8k-lr1e-6-bsz64-kl0.05-save_interval10-rmmult0.1-shuffled/actor_results/checkpoints/llama3-8b-ifeval-gsm8k-lr1e-6-bsz512-kl0.05-rmmult0.1-ifeval1-step120.nemo --output_dir results/eval_tool --prompt_template llama3_instruct_empty_sys model.tensor_model_parallel_size=8 model.pipeline_model_parallel_size=1 model.batch_size.generation=16 model.batch_size.log_probs=4 num_nodes=1


bash evals/eval_eval_tool.sh --nemo_file /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/llama3-8b-lr1e-6-bsz512-kl0.1-4node-joint-rmmult0.1-imult1-roseleech10k-roserm/actor_results/checkpoints/llama3-8b-ifeval-roseleech10k-lr1e-6-bsz512-kl0.1-rmmult0.1-ifeval1-step70.nemo --output_dir results/eval_tool/llama3-8b-roseleech10K-lr1e-6-bsz512-kl0.1-rmmult0.1-ifeval1-step70 --prompt_template llama3_instruct_empty_sys model.tensor_model_parallel_size=8 model.pipeline_model_parallel_size=1 model.batch_size.generation=16 model.batch_size.log_probs=4 num_nodes=1


bash evals/eval_eval_tool.sh --nemo_file /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/minitron-8b-test4/actor_results/checkpoints/minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step40.nemo --output_dir results/eval_tool//minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step40 --prompt_template llama3_instruct_empty_sys model.tensor_model_parallel_size=8 model.pipeline_model_parallel_size=1 model.batch_size.generation=16 model.batch_size.log_probs=4 num_nodes=1


bash evals/eval_eval_tool.sh --nemo_file /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/minitron-8b-test4/actor_results/checkpoints/minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step40.nemo --output_dir results/eval_tool/minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step40 --prompt_template extra_sft_empty_sys model.tensor_model_parallel_size=8 model.pipeline_model_parallel_size=1 model.batch_size.generation=16 model.batch_size.log_probs=4 num_nodes=1


bash evals/eval_eval_tool.sh --nemo_file /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/minitron-8b-test4/actor_results/checkpoints/minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step35.nemo --output_dir results/eval_tool/minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step35 --prompt_template extra_sft_empty_sys model.tensor_model_parallel_size=8 model.pipeline_model_parallel_size=1 model.batch_size.generation=16 model.batch_size.log_probs=4 num_nodes=1



bash evals/eval_eval_tool.sh --nemo_file /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/minitron-8b-test4/actor_results/checkpoints/minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step70.nemo --output_dir results/eval_tool/minitron-helpsteer-kl0.05-lr1e-6-llama3.1RM-step70 --prompt_template extra_sft_empty_sys model.tensor_model_parallel_size=8 model.pipeline_model_parallel_size=1 model.batch_size.generation=16 model.batch_size.log_probs=4 num_nodes=1



bash evals/eval_eval_tool.sh --nemo_file /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/minitron-8b-gsm8k-llama3.1-70BRM-lr1e-6-kl0.05/actor_results/checkpoints/minitron-gsm8k-kl0.05-lr1e-6-llama3.1RM-step40.nemo --output_dir results/eval_tool/minitron-gsm8k-kl0.05-lr1e-6-llama3.1RM-step40 --prompt_template extra_sft_empty_sys model.tensor_model_parallel_size=8 model.pipeline_model_parallel_size=1 model.batch_size.generation=16 model.batch_size.log_probs=4 num_nodes=1


bash evals/eval_eval_tool.sh --nemo_file /lustre/fsw/portfolios/llmservice/users/abukharin/test/exp/rlhf/minitron-8b-helpsteer-gsm8k-llama3.1-70BRM-lr1e-6-kl0.05/actor_results/checkpoints/minitron-helpsteer-gsm8k-kl0.05-lr1e-6-llama3.1RM-step60.nemo --output_dir results/eval_tool/minitron-helpsteer-gsm8k-kl0.05-lr1e-6-llama3.1RM-step60 --prompt_template extra_sft_empty_sys model.tensor_model_parallel_size=8 model.pipeline_model_parallel_size=1 model.batch_size.generation=16 model.batch_size.log_probs=4 num_nodes=1
