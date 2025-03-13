### Context Parallelism
Context Parallelism is supported for the training stage. Response tokens are sharded along the sequence dimension across context parallel GPUs, which correspondingly shards activations reducing memory usage. In the generation stage, context parallel GPUs are resharded and combined into data parallel groups. 

Context parallelism can be enabled with:  
`trainer.grpo.inference_backend.reshard=True`  
`model.context_parallel_size={CP_SIZE}`

