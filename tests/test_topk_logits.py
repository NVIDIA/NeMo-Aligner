## TODO: integrate pytest, add to test_distributed.py
## also maybe refactor so we don't have to duplicate code here?

## TODO: test on multiple GPUs
import os
import torch
from megatron.core import tensor_parallel
from megatron.core.parallel_state import get_tensor_model_parallel_group
#from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from megatron.core.parallel_state import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, initialize_model_parallel
from megatron.core.tensor_parallel.utils import VocabUtility

def naive_topk(output_tensor, target_topk_token_ids):
    output_tensor_max = torch.max(output_tensor, dim=-1)[0]
    torch.distributed.all_reduce(output_tensor_max,
                                op=torch.distributed.ReduceOp.MAX,
                                group=get_tensor_model_parallel_group())
    output_tensor = output_tensor - output_tensor_max.unsqueeze(dim=-1).detach()
    output_tensor = tensor_parallel.gather_from_tensor_model_parallel_region(output_tensor)

    # compute the knowlodge distillation loss against the ground-truth logits
    topk_logits = torch.gather(output_tensor, dim=-1, index=target_topk_token_ids)

    return topk_logits

def efficient_topk(vocab_parallel_logits, target_token_ids, target_logits):
    # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
        )
        # Subtract the maximum value.
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)
        
        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target_token_ids < vocab_start_index) | (target_token_ids >= vocab_end_index)
        masked_target_token_ids = target_token_ids.clone() - vocab_start_index
        masked_target_token_ids[target_mask] = 0

        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target_token_ids.view(-1)
        K = target_logits.size()[-1]
        ## we want to select K tokens per example
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device).repeat_interleave(K)
        target_token_ids_1d = masked_target_token_ids.view(-1) ## B * seq_length * K
        predicted_logits_1d = logits_2d[arange_1d, target_token_ids_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target_logits)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(
            predicted_logits, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
        )

        return predicted_logits

def test_topk_logits(K = 3, batch_size = 4, seq_len = 8, partition_vocab_size = 16):
    
    rank = int(os.environ['LOCAL_RANK'])
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)
    initialize_model_parallel(
        tensor_model_parallel_size = world_size,
    )


    torch.manual_seed(0)

    true_logits = (torch.randint(low=0, high=100, size=(batch_size, seq_len, partition_vocab_size * world_size)) / 5).to(torch.cuda.current_device())
    target_logits, target_token_ids = torch.topk(true_logits, 3)

    torch.manual_seed(torch.cuda.current_device())
    vocab_parallel_logits = (torch.randint(low=0, high=100, size=(batch_size, seq_len, partition_vocab_size)) / 5).to(torch.cuda.current_device())


    naive_topk_result = naive_topk(vocab_parallel_logits, target_token_ids)
    efficient_topk_result = efficient_topk(vocab_parallel_logits, target_token_ids, target_logits)

    assert torch.all(naive_topk_result == efficient_topk_result)


if __name__ == '__main__':
    test_topk_logits()
