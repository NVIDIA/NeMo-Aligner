import torch

from nemo_aligner.utils.text_generation_utils import (
    TrackLengthGPTModelTextGenerationStrategy,
    verify_is_valid_and_clamp_range_,
)


def test_verify_is_valid_and_clamp_range(dummy_gpt_model):
    max_gen_length = 8

    random_gen = [9, 8]  # chosen arbitrarily
    extra_id_1_ids = dummy_gpt_model.tokenizer.text_to_ids("<extra_id_1>")
    extra_id_2_ids = dummy_gpt_model.tokenizer.text_to_ids("<extra_id_2>")
    eos_id = dummy_gpt_model.tokenizer.eos_id

    # response contains prompt + generation
    response_tokens = [
        [1] + random_gen,  # doesn't end with an eos
        [1, 1] + random_gen + [eos_id],
        [1] + random_gen + extra_id_1_ids,
        [1, 1] + random_gen + extra_id_1_ids,
        [1] + random_gen + extra_id_2_ids,
    ]

    # The padding has to be eos_id
    response_tokens = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x) for x in response_tokens], batch_first=True, padding_value=eos_id
    )

    context_lengths = torch.tensor([1, 2, 1, 2, 1])
    generation_lengths = torch.tensor([0, 1, len(extra_id_1_ids), len(extra_id_2_ids), len(extra_id_2_ids)]) + len(
        random_gen
    )
    response_lengths = context_lengths + generation_lengths

    strategy = TrackLengthGPTModelTextGenerationStrategy(dummy_gpt_model, context_lengths, max_gen_length)
    is_end = verify_is_valid_and_clamp_range_(
        response_tokens=response_tokens,
        response_lengths=response_lengths,
        strategy=strategy,
        tokenizer=dummy_gpt_model.tokenizer,
        end_strings=["<extra_id_1>"],
    )
    assert is_end.tolist() == [False, True, True, True, False]
