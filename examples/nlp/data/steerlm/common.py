SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

SYSTEM_PROMPT_TEMPLATE = "<extra_id_0>System\n{value}\n"

USER_TURN_TEMPLATE = "<extra_id_1>User\n{value}\n"

ASSISTANT_TURN_TEMPLATE = "<extra_id_1>Assistant\n{value}\n"

LABEL_PREFIX = "<extra_id_2>"

OPEN_ASSISTANT_ATTRIBUTES = ["quality", "toxicity", "humor", "creativity"]

HELPSTEER_ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

ALL_STEERLM_ATTRIBUTES = OPEN_ASSISTANT_ATTRIBUTES + HELPSTEER_ATTRIBUTES
