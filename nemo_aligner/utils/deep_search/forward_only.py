from megatron.core import InferenceParams, parallel_state


def get_forward_output_only_func(obj):

    def fwd_output_only_func(dataloader_iter, model):
        batch = next(dataloader_iter)
        extra_arg = {}
        (
            tokens,
            attention_mask,
            position_ids,
        ) = batch
        tokens = tokens.cuda()
        position_ids = position_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
            # attention_mask = attention_mask[0:1]
        extra_arg['inference_params'] = obj.inference_params
        output_tensor = model(tokens, position_ids, attention_mask, **extra_arg)

        def id_func(output_tensor):
            return output_tensor, {'logits': output_tensor}

        return output_tensor, id_func

    return fwd_output_only_func
