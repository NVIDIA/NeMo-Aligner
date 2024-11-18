import argparse
import omegaconf
import json
import os
from tqdm import tqdm

from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo_aligner.utils.utils import load_checkpoint_model_config
from nemo_aligner.data.nlp.builders import build_dataset_generic
from nemo_aligner.data.nlp.datasets import DPOModelDataset, DataItemInvalidError
from tempfile import TemporaryDirectory
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from omegaconf import DictConfig

# TODO(terryk): replace once NeMo has a tokenizer building API
# Copied from nemo.collections.nlp.models.language_modeling.megatron_base_model.MegatronBaseModel._build_tokenizer
#  to avoid loading full model
def build_tokenizer(model_cfg: DictConfig, restore_path: str):
    def maybe_unpack_nemo_artifact(restore_path: str, artifact_name: str | None, tmpdir: str):
        if artifact_name is None:
            return None
        
        rel_art_path = artifact_name[len('nemo:'):] if artifact_name.startswith('nemo:') else artifact_name
        if os.path.isdir(restore_path):
            return os.path.join(restore_path, rel_art_path)
        else:
            NLPSaveRestoreConnector._unpack_nemo_file(restore_path, tmpdir, members=[rel_art_path])
            return os.path.join(tmpdir, rel_art_path)
            
    with TemporaryDirectory() as tmpdir:
        #artifacts = NLPSaveRestoreConnector._filtered_tar_info(restore_path, filter_fn=lambda name: name.startswith('nemo:'))

        if hasattr(model_cfg.tokenizer, "sentencepiece_legacy"):
            legacy = model_cfg.tokenizer.sentencepiece_legacy
        else:
            legacy = True if model_cfg.tokenizer.library == 'sentencepiece' else False
        tokenizer = get_nmt_tokenizer(
            library=model_cfg.tokenizer.library,
            model_name=model_cfg.tokenizer.get("type", None),
            tokenizer_model=maybe_unpack_nemo_artifact(
                restore_path=restore_path,
                artifact_name=model_cfg.tokenizer.get('model', None),
                tmpdir=tmpdir,
            ),
            vocab_file=maybe_unpack_nemo_artifact(
                restore_path=restore_path,
                artifact_name=model_cfg.tokenizer.get('vocab_file', None),
                tmpdir=tmpdir,
            ),
            merges_file=maybe_unpack_nemo_artifact(
                restore_path=restore_path,
                artifact_name=model_cfg.tokenizer.get('merge_file', None),
                tmpdir=tmpdir,
            ),
            use_fast=model_cfg.tokenizer.get('use_fast', False),
            delimiter=model_cfg.tokenizer.get('delimiter', None),
            special_tokens=model_cfg.tokenizer.get('special_tokens', None),
            trust_remote_code=model_cfg.tokenizer.get('trust_remote_code', False),
            legacy=legacy,
            chat_template=getattr(model_cfg.tokenizer, "chat_template", None),
        )

    if model_cfg.tokenizer.get('additional_special_tokens', None) is not None:
        tokens_list = omegaconf.OmegaConf.to_object(model_cfg.tokenizer.additional_special_tokens)
        tokenizer.add_special_tokens(tokens_list)
    return tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('restore_from_path')
    parser.add_argument('input_jsonl')
    parser.add_argument('output_jsonl')

    args = parser.parse_args()

    if not os.path.exists(args.input_jsonl):
        raise FileNotFoundError(f"{args.input_jsonl} does not exist")
    elif os.path.exists(args.output_jsonl):
        raise FileExistsError(f"About to override {args.output_jsonl}. Please delete before continuing")
    
    model_cfg = load_checkpoint_model_config(args.restore_from_path)

    tokenizer = build_tokenizer(model_cfg, args.restore_from_path)

    dataset: DPOModelDataset = build_dataset_generic(
        cls=DPOModelDataset,
        cfg=model_cfg,
        data_prefix=[args.input_jsonl],
        data_impl='jsonl',
        num_samples=float('inf'), # Doesn't matter for jsonl
        seq_length=float('inf'),  # Doesn't matter since we are checking tokenization, not the length
        seed=42,  # Doesn't matter
        tokenizer=tokenizer,
        name='does not matter',
    )

    num_skipped = 0
    orig_len = len(dataset)

    with open(args.output_jsonl, 'w') as f:
        for i in tqdm(range(orig_len), desc='Filtering DPO dataset', total=orig_len):
            try:
                # If this throws an error, then the datum was invalid
                _ = dataset[i]
                f.write(json.dumps(dataset.data[i]) + '\n')
            except DataItemInvalidError:
                num_skipped += 1
    print(f"{orig_len=}")
    print(f"{num_skipped=}")
    print(f"Output available at: {args.output_jsonl}")
