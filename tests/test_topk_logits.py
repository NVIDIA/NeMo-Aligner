## TODO: integrate pytest, add to test_distributed.py
## also maybe refactor so we don't have to duplicate code here?

## TODO: test on multiple GPUs
import os
import torch
from nemo_aligner.utils.distributed import _TopKLogitsCrossEntropy
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

### the two functions below are copied & slightly modified (removing self.)
### from  (https://github.com/NVIDIA/NeMo-Aligner/blob/8927528c20f0a16254e72fceeea0adf842e34c94/nemo_aligner/models/nlp/gpt/megatron_gpt_knowledge_distillation.py#L182)
def loss_func(logits, target_logits, loss_mask, kd_loss="bwd_kl", logits_scale=1., target_logits_scale=1.,):
    """The cross entropy function between two categorical distributions. 
    logits: Tensor of [B, seq_len, K].
    target_logits: Tensor of [B, seq_len, K].
    loss_mask: Tensor of [B, seq_len].
    """

    logprobs = torch.nn.functional.log_softmax(logits_scale * logits, dim=-1)
    target_logprobs = torch.nn.functional.log_softmax(target_logits_scale * target_logits, dim=-1)

    if kd_loss == "bwd_kl":
        loss = torch.sum(target_logprobs.exp() * (target_logprobs - logprobs), dim=-1)
    elif kd_loss == "fwd_kl":
        loss = torch.sum(logprobs.exp() * (logprobs - target_logprobs), dim=-1)
    else:
        raise ValueError(f"kd_loss {kd_loss} is not supported.")
    return torch.sum(loss * loss_mask) / torch.sum(loss_mask).clamp(min=1.)
    

def naive_topk_loss_function(output_tensor, target_topk_logits, target_topk_token_ids, target_log_sum_exp_logits, loss_mask):

    output_tensor_max = torch.max(output_tensor, dim=-1)[0]
    torch.distributed.all_reduce(output_tensor_max,
                                op=torch.distributed.ReduceOp.MAX,
                                group=get_tensor_model_parallel_group())
    output_tensor = output_tensor - output_tensor_max.unsqueeze(dim=-1).detach()
    output_tensor = tensor_parallel.gather_from_tensor_model_parallel_region(output_tensor)
    
    # compute the knowlodge distillation loss against the ground-truth logits
    topk_logits = torch.gather(output_tensor, dim=-1, index=target_topk_token_ids)


    if False: #use_k_add_1_logits: ## TODO: add support
        # When target_log_sum_exp_logits is not None. The objective is
        # target_prob_k = exp(target_logits_k) / exp(target_log_sum_exp_logits), k=1,..., K
        # target_prob_{K+1} = 1 - sum_{k=1}^K target_prob_k
        # prob_k = exp(logits_k) / sum_{v=1}^V exp(logits_v), k=1,..., K
        # prob_{K+1} = 1 - sum_{k=1}^K prob_k
        # neg_loss = sum_{k=1}^{K+1} target_prob_k * log prob_{k}
        
        log_sum_exp_logits = torch.logsumexp(output_tensor, dim=-1)
        # We can't use `gather_from_tensor_model_parallel_region` here since it discards
        # gradients from other ranks - we need to all_reduce the gradients as well.
        sum_exp_logits_subtract_topk_exp_logits = (log_sum_exp_logits.exp() - topk_logits.exp().sum(-1)).clamp(min=1e-10)
        topk_logits = torch.cat([topk_logits, sum_exp_logits_subtract_topk_exp_logits.log().unsqueeze(-1)], -1)
        
        target_sum_exp_logits_subtract_topk_exp_logits = (target_log_sum_exp_logits.exp() - target_topk_logits.exp().sum(-1)).clamp(min=1e-10)
        target_topk_logits_in_loss = torch.cat([target_topk_logits, target_sum_exp_logits_subtract_topk_exp_logits.log().unsqueeze(-1)], -1)
    else:
        # When not use_k_add_1_logits. The objective is 
        # target_prob_k = exp(target_logits_k) / sum_{k=1}^K exp(target_logits_k)
        # prob_k = exp(logits_k) / sum_{k=1}^K exp(logits_k)
        # neg_loss = sum_{k=1}^{K} target_prob_k * log prob_{k} 
        
        log_sum_exp_logits = None
        target_topk_logits_in_loss = target_topk_logits
        
    kd_loss = loss_func(topk_logits, target_topk_logits_in_loss, loss_mask=loss_mask)
    
    # compute the sft loss against the ground-truth labels
    sft_loss = torch.zeros_like(kd_loss)
    if False: #sft_loss_weight != 0: ## TODO: add support
        target_label_logits = torch.gather(output_tensor, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        if log_sum_exp_logits is None:
            log_sum_exp_logits = torch.logsumexp(output_tensor, dim=-1)
        target_label_logprobs = target_label_logits - log_sum_exp_logits
        sft_loss = - torch.sum(target_label_logprobs * loss_mask) / torch.sum(loss_mask).clamp(min=1.)
    
    # compute the aggregated loss ## TODO: support
    #loss = self.kd_loss_weight * kd_loss + self.sft_loss_weight * sft_loss
    return kd_loss

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
    target_log_sum_exp_logits = true_logits.exp().sum(-1)
    loss_mask = torch.ones(target_logits.size()[:-1]).to(torch.cuda.current_device())

    torch.manual_seed(torch.cuda.current_device() + 10)
    vocab_parallel_logits = torch.autograd.Variable((torch.randint(low=0, high=100, size=(batch_size, seq_len, partition_vocab_size)) / 5).to(torch.cuda.current_device()),  requires_grad=True)

    naive_topk_result = naive_topk(vocab_parallel_logits, target_token_ids)
    efficient_topk_result = efficient_topk(vocab_parallel_logits, target_token_ids, target_logits)

    assert torch.all(naive_topk_result == efficient_topk_result)

    ## test loss function
    # test forward
    ctx = torch.autograd.function.FunctionCtx()

    naive_loss = naive_topk_loss_function(vocab_parallel_logits, target_logits, target_token_ids, target_log_sum_exp_logits, loss_mask)
    
    ## TODO: loss mask?
    new_loss = _TopKLogitsCrossEntropy.forward(ctx, vocab_parallel_logits, target_logits, target_token_ids, None) #, target_log_sum_exp_logits)
    ## sum p(x)logp(x) - p(x) logq(x)
    target_probs = target_logits.exp()
    target_probs = target_probs / target_probs.sum(-1, keepdims=True)
    new_loss = torch.mean(new_loss) ## TODO -- at what point do we reduce?

    torch.testing.assert_close(naive_loss, new_loss)

    ctx.saved_tensors = ctx.to_save ## WAR for "AttributeError: 'FunctionCtx' object has no attribute 'saved_tensors'"

    # test backward
    naive_loss.backward()
    naive_grad = vocab_parallel_logits.grad
    new_grad = _TopKLogitsCrossEntropy.backward(ctx, 1. / (batch_size * seq_len) * torch.ones(batch_size, seq_len).to(torch.cuda.current_device()))[0]

    torch.testing.assert_close(naive_grad, new_grad)

if __name__ == '__main__':
    test_topk_logits()
