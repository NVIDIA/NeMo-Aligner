from megatron.core import InferenceParams, parallel_state


def get_forward_output_only_func(obj):

    def fwd_output_only_func(dataloader_iter, model):
        batch = next(dataloader_iter)
        extra_arg = {}
        if len(batch) == 3:
            batch = [x.cuda() for x in batch]
            tokens, attention_mask, position_ids = batch
            attention_mask = attention_mask[0:1]
        else:
            (
                tokens,
                attention_mask,
                position_ids,
                set_inference_key_value_memory,
                inference_max_sequence_len,
            ) = batch
            tokens = tokens.cuda()
            position_ids = position_ids.cuda()
            if attention_mask is not None:
                attention_mask = attention_mask.cuda()
                attention_mask = attention_mask[0:1]
            # if first step, then clear KV cache, otherwise reuse inference_paarms
            if obj.mcore_gpt:
                # if first step, then clear KV cache, otherwise reuse inference_paarms
                if set_inference_key_value_memory[0].item():
                    obj.inference_params = InferenceParams(
                        max_batch_size=tokens.size(0), max_sequence_length=inference_max_sequence_len[0].item()
                    )
                extra_arg['inference_params'] = obj.inference_params
            else:
                raise ValueError("mcore_gpt must be True")
        output_tensor = model(tokens, position_ids, attention_mask, **extra_arg)

        # Advance inference sequence offset.
        if obj.inference_params:
            # if last stage, then (final) output is [b, s, h], otherwise it's [s, b, h]
            if parallel_state.is_pipeline_last_stage():
                obj.inference_params.sequence_len_offset += output_tensor.size(1)
            else:
                obj.inference_params.sequence_len_offset += output_tensor.size(0)

        def id_func(output_tensor):
            return output_tensor, {'logits': output_tensor}

        return output_tensor, id_func

    return fwd_output_only_func
