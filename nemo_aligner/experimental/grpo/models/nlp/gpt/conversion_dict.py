import torch 
import einops

def get_layer_num(s):
    """Assumes layer number is preceeded by 'layers.'"""
    segments = s.split('.')
    number = None
    for i, segment in enumerate(segments):
        if segment == 'layers':
            if segments[i+1].isdigit():
                number = int(segments[i+1])
                break
    return number

def split_qkv_llama(gathered_mcore_qkv_layer, cfg):
    hidden_size = cfg.hidden_size
    head_num = cfg.num_attention_heads
    num_query_groups = cfg.get("num_query_groups", head_num)  # different num_query_groups for 70B

    head_size = cfg.get("kv_channels") or (hidden_size // head_num)  # equivalent to hf's head_dim
    heads_per_group = head_num // num_query_groups
    qkv_total_dim = head_num + 2 * num_query_groups

    qkv_weights = gathered_mcore_qkv_layer
    qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))
    ## Example of slices
    ## 7b: num_query_groups = head_num = 32,
    ## q_slice = [0, 3, 6, 9 , ... 90, 93]
    ## k_slice = [1, 4, 7, 10, ... 91, 94]
    ## v_slice = [2, 5, 8, 11, ... 92, 95]
    ## 70b (with GQA): num_query_groups = 8, head_num = 64
    ## q_slice = [0, 1, .. 6, 7, 10, 11, .. 16, 17, 20, 21, .. 67, 70, ... 76, 77]
    ## k_slice = [8, 18, 28, ... 68, 78]
    ## v_slice = [9, 19, 29, ... 69, 79]

    q_name = 'model.layers.{l}.self_attn.q_proj.weight'
    k_name = 'model.layers.{l}.self_attn.k_proj.weight'
    v_name = 'model.layers.{l}.self_attn.v_proj.weight'
    q = qkv_weights[q_slice].reshape(-1, hidden_size)
    k = qkv_weights[k_slice].reshape(-1, hidden_size)
    v = qkv_weights[v_slice].reshape(-1, hidden_size)

    return {q_name: q, k_name: k, v_name: v}

def split_fc1_gate_down_llama(gathered_mcore_fc1, cfg):
    # gate proj and up proj are mixed right now, and we need to reshape them
    # [ gate_tp0 ]     [ gate_tp0 ] 
    # [  up_tp0  ] --\ [ gate_tp1 ] --\ (split gate)
    # [ gate_tp1 ] --/ [  up_tp0  ] --/ (split  up)
    # [  up_tp1  ]     [  up_tp1  ]
    tp = cfg.tensor_model_parallel_size
    gathered_mcore_fc1 = einops.rearrange(gathered_mcore_fc1, '(t c d) a1 ->  c (t d) a1', c=2, t=tp)
    mlp_gate_proj_weight = gathered_mcore_fc1[0]
    mlp_up_proj_weight = gathered_mcore_fc1[1]
    mlp_gate_proj_base_name = 'model.layers.{l}.mlp.gate_proj.weight'
    mlp_up_proj_base_name = 'model.layers.{l}.mlp.up_proj.weight'
    return {mlp_up_proj_base_name: mlp_up_proj_weight, mlp_gate_proj_base_name: mlp_gate_proj_weight}


mcore_te_to_hf_llama = {
    'model.embedding.word_embeddings.weight': {"tp": 0, "hf":"model.embed_tokens.weight"},
    'model.decoder.final_layernorm.weight': {"hf": "model.norm.weight"},
    'model.output_layer.weight': {"tp": 0, "hf":"lm_head.weight"},
    'model.decoder.layers.{l}.self_attention.linear_proj.weight': {"tp":1, "hf":"model.layers.{l}.self_attn.o_proj.weight"},
    'model.decoder.layers.{l}.self_attention.linear_qkv.weight': {"tp":0, "hf_func": split_qkv_llama},
    'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight': {"hf": "model.layers.{l}.input_layernorm.weight"},
    'model.decoder.layers.{l}.mlp.linear_fc1.weight': {"tp": 0, "hf_func": split_fc1_gate_down_llama},
    'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight': {"hf": "model.layers.{l}.post_attention_layernorm.weight"},
    'model.decoder.layers.{l}.mlp.linear_fc2.weight': {"tp": 1, "hf": "model.layers.{l}.mlp.down_proj.weight"},
}

