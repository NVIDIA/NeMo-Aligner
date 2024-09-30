# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union, Dict

import hydra
import torch
from megatron.core import parallel_state
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MegatronNevaModel, MCoreNevaModel
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_default_length_params,
    get_default_sampling_params,
)
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import get_specs
from nemo.utils import logging
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, OutputType, SamplingParam
from nemo.collections.nlp.modules.common.text_generation_utils import generate
from megatron.core import InferenceParams
from nemo_aligner.utils.text_generation_utils import MGPTModelTextGenerationStrategy

class MultimodalGPTModel(MegatronNevaModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        inference_params = dict(cfg.get("inference", {}))
        # note that this will fail if import path is not available when the model is restored
        # this is by design as it might not be possible to use model correctly without a matching
        # inference strategy
        if "strategy" in inference_params:
            if inference_params["strategy"] is not None:
                inference_params["strategy"] = hydra.utils.instantiate(inference_params["strategy"], model=self)
        self.set_inference_params(**inference_params)

    def set_inference_params(self, length_params=None, sampling_params=None, strategy=None):
        # TODO (igitman): the name self._inference_params is very similar to self.inference_params
        #    that's used by the base model for another purpose. There is also self._inference_config
        #    that has a similar role to the parameters below but is less convenient.
        #    While there is a danger for accidental name collision and this adds confusion, it's ok for now
        #    as we are planning to remove dependence on the MegatronGPTModel after which we can remove this note

        # registering inference parameters or default values
        self._inference_params = {
            "length_params": length_params or get_default_length_params(),
            "sampling_params": sampling_params or get_default_sampling_params(),
            "strategy": strategy,
        }

    def get_inference_params(self):
        return self._inference_params

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        media_start_id = self.tokenizer.token_to_id(self.cfg.mm_cfg.get("im_start_token", "[IMG_BREAK]"))
        media_end_id = self.tokenizer.token_to_id(self.cfg.mm_cfg.get("im_end_token", "[IMG_END]"))
        media_token_id = self.tokenizer.token_to_id(self.cfg.mm_cfg.get("image_patch_token", "[IMG]"))

        if self.mcore_gpt:
            if not parallel_state.is_initialized():

                def dummy():
                    return

                if self.trainer.strategy.launcher is not None:
                    self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
                self.trainer.strategy.setup_environment()

            model = MCoreNevaModel(
                mm_cfg=self.cfg.mm_cfg,
                media_start_id=media_start_id,
                media_token_id=media_token_id,
                media_end_id=media_end_id,
                mcore_gpt=self.mcore_gpt,
                config=self.transformer_config,
                transformer_layer_spec=get_specs(self.spec_name),
                vocab_size=self.cfg.get('override_vocab_size', self.padded_vocab_size),
                max_sequence_length=self.cfg.get('encoder_seq_length', 512),
                pre_process=pre_process,
                post_process=post_process,
                parallel_output=True,
                share_embeddings_and_output_weights=self.cfg.get('share_embeddings_and_output_weights', True),
                position_embedding_type=self.cfg.get('position_embedding_type', 'learned_absolute'),
                rotary_percent=self.cfg.get('rotary_percentage', 1.0),
                seq_len_interpolation_factor=self.cfg.get('seq_len_interpolation_factor', None),
                rotary_base=self.cfg.get('rotary_base', 10000),
            )
        else:
            raise NotImplementedError("Only MCoreGPT models are supported! Please set mcore_gpt=True.")
            

        logging.info(
            f"Neva model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
        )

        return model
    
    def generate(
        self,
        inputs: Union[Dict, List[Dict]],
        length_params: LengthParam,
        sampling_params: SamplingParam = None,
        *,
        strategy: Optional[MGPTModelTextGenerationStrategy] = None,
    ) -> OutputType:
        
        # set the default sampling params if it is None.
        # default do greedy sampling
        if sampling_params is None:
            sampling_params = get_default_sampling_params()

        # set the default length params if it is None.
        # default do greedy sampling
        if length_params is None:
            length_params = get_default_length_params()

        extra = {}
        if strategy is None:
            strategy = MGPTModelTextGenerationStrategy(self.cuda())
            extra['strategy'] = strategy
        else:
            extra['strategy'] = strategy
        
        if isinstance(inputs, tuple):
            context_tokens_tensor, context_length_tensor = inputs
        else:
            context_tokens_tensor, context_length_tensor = strategy.tokenize_batch(
                inputs.get('prompt'), length_params['max_length'], sampling_params['add_BOS']
            )
        
        output = generate(
            self.cuda(),
            inputs=(context_tokens_tensor, context_length_tensor),
            tokens_to_generate=length_params['max_length'],
            all_probs=sampling_params['all_probs'],
            compute_logprob=sampling_params['compute_logprob'],
            temperature=sampling_params['temperature'],
            add_BOS=sampling_params['add_BOS'],
            top_k=sampling_params['top_k'],
            top_p=sampling_params['top_p'],
            greedy=sampling_params['use_greedy'],
            repetition_penalty=sampling_params['repetition_penalty'],
            end_strings=sampling_params['end_strings'],
            min_tokens_to_generate=length_params['min_length'],
            compute_attention_mask=sampling_params.get("compute_attention_mask", True),
            image_list=inputs.get("image"),
            **extra,
        )
       
        if output is not None:  # may be `None` for intermediate PP ranks when PP>2
            for k in output:
                if isinstance(output[k], torch.Tensor):
                    output[k] = output[k].tolist()

        max_response_length = length_params['max_length']
        if output is not None:  # may be `None` for intermediate PP ranks when PP>2
            # adding predictions key which contains only model predictions without the prompt
            output["predictions"] = [
                self.tokenizer.ids_to_text(tokens[length.item() :][:max_response_length])
                for tokens, length in zip(output["token_ids"], context_length_tensor)
            ]
            
            if not sampling_params['all_probs']:
                del output['full_logprob']             

        return output

    def load_state_dict(self, state_dict, strict=False):
        logging.warning('Loading state dict for MegatronNevaModel...')
        results = NLPModel.load_state_dict(self, state_dict, strict=False)
        missing_keys, unexpected_keys = results

        if len(unexpected_keys) > 0:
            logging.critical('Unexpected keys were detected during the load. Please double check.')
            logging.critical(f'Unexpected keys: \n{unexpected_keys}')
        return results

    def get_forward_output_only_func(self):
        def fwd_output_only_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            if isinstance(batch, tuple):
                batch = batch[0]
            extra_arg = {}
            (
                tokens,
                attention_mask,
                position_ids,
                media,
                set_inference_key_value_memory,
                inference_max_sequence_len,
            ) = batch
            tokens = tokens.cuda()
            position_ids = position_ids.cuda()
            if attention_mask != None:
                attention_mask = attention_mask.cuda()
                attention_mask = attention_mask[0:1]
            if media is not None:
                media = [element.cuda() for sample in media for element in sample] # we have list of list of images
                #media = media.cuda()
            labels = None
            if self.mcore_gpt:
                # if first step, then clear KV cache, otherwise reuse inference_paarms
                if set_inference_key_value_memory[0].item():
                    self.inference_params = InferenceParams(
                        max_batch_size=tokens.size(0), max_sequence_length=inference_max_sequence_len[0].item()
                    )
                extra_arg['inference_params'] = self.inference_params
            else:
                extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
                extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()

            forward_args = {
                'input_ids': tokens,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'media': media,
            }
            if not self.mcore_gpt:
                forward_args['checkpoint_activations_all_layers'] = None
            output_tensor = model(**forward_args, **extra_arg)

            # Advance inference sequence offset.
            if self.inference_params:
                # if last stage, then (final) output is [b, s, h], otherwise it's [s, b, h]
                if parallel_state.is_pipeline_last_stage():
                    self.inference_params.sequence_len_offset += output_tensor.size(1)
                else:
                    self.inference_params.sequence_len_offset += output_tensor.size(0)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func