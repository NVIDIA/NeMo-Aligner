# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize

from nemo.collections.multimodal.data.clip.clip_dataset import get_preprocess_fns
from nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_models import (
    CLIPTextTransformer,
    CLIPVisionTransformer,
    MegatronCLIPModel,
)
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.utils import logging
from nemo_aligner.data.mm.pickscore_dataset import build_train_valid_datasets

BICUBIC = InterpolationMode.BICUBIC


class PickscoreRewardModel(MegatronModule):
    """CLIP-Based Model"""

    def __init__(self, model_cfg, model_parallel_config, padded_vocab_size, pre_process=True, post_process=True):
        super(PickscoreRewardModel, self).__init__()
        self.config = model_parallel_config
        self.pre_process = pre_process
        self.post_process = post_process
        self.vision_encoder = CLIPVisionTransformer(
            model_cfg.vision, model_parallel_config, pre_process=self.pre_process, post_process=self.post_process,
        )
        self.text_encoder = CLIPTextTransformer(
            model_cfg.text,
            model_parallel_config,
            padded_vocab_size,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        # TODO (yuya): fix this
        pass

    def get_reward(self, images, captions):
        text_features = self.text_encoder(captions)
        image_features = self.vision_encoder(images)

        rewards = (
            self.logit_scale.exp()
            * torch.matmul(F.normalize(image_features, dim=-1), F.normalize(text_features, dim=-1).t()).diag()
        )

        return rewards

    def forward(self, images, captions):
        return self.get_reward(images, captions)


class MegatronCLIPRewardModel(MegatronCLIPModel):
    def __init__(self, cfg, trainer):
        super().__init__(cfg, trainer)
        self.openai_dataset_mean = (0.48145466, 0.4578275, 0.40821073)
        self.openai_dataset_std = (0.26862954, 0.26130258, 0.27577711)
        self.transform_size = 224
        self.rescale_param = 0.00392156862745098
        self.differentiable_preprocess = self.diff_preprocess()

    def diff_preprocess(self):
        return Compose(
            [
                Resize(self.transform_size, interpolation=BICUBIC, antialias=True),
                CenterCrop(self.transform_size),
                self.rescale,
                Normalize(self.openai_dataset_mean, self.openai_dataset_std),
            ]
        )

    def rescale(self, image):
        return image * self.rescale_param

    def preprocess(self, images, captions):
        _, text_transform = get_preprocess_fns(self.cfg, tokenizer=self.tokenizer, is_train=False)
        images = (
            torch.stack([self.differentiable_preprocess(img.permute(2, 0, 1)) for img in images])
            .to(torch.cuda.current_device())
            .float()
        )
        captions_list = [text_transform(captions[i]) for i in range(images.shape[0])]
        captions = torch.stack(captions_list).to(torch.cuda.current_device())

        return images, captions

    def get_reward(self, images, captions):
        images, captions = self.preprocess(images, captions)
        return self.model.get_reward(images, captions)

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        model = PickscoreRewardModel(
            model_cfg=self.cfg,
            model_parallel_config=self.model_parallel_config,
            padded_vocab_size=self.padded_vocab_size,
            pre_process=pre_process,
            post_process=post_process,
        )
        return model

    def forward(self, images, captions):
        rewards = self.get_reward(images, captions)
        return rewards

    def loss_func(self, output_tensor):
        """ 
        rewards_0, rewards_1 - tensor of size [B,] each that computes rewards 
        labels - Bx2 tensor containing labels
        """
        rewards_0, rewards_1, labels = output_tensor
        logits = torch.stack([rewards_0, rewards_1], 1)  # [B, 2]
        logsoftmax = F.log_softmax(logits, dim=1)
        labels_s = labels + 1e-4
        labels_s = labels_s / labels_s.sum(1, keepdim=True)
        # to balance out the non-zero term for 0.5 case
        entropy = (labels_s * torch.log(labels_s)).sum(1).mean().detach()
        # this is the KL div loss for categorical
        local_loss = -(labels * logsoftmax).sum(1).mean() + entropy
        # compute accuracy
        gt_exists = torch.where(labels.max(1).values > 0.5)[0]
        logits_masked = torch.argmax(logits, 1)[gt_exists]
        labels_masked = torch.argmax(labels, 1)[gt_exists]
        accuracy = torch.mean(1.0 * (logits_masked == labels_masked))
        # compute reduced metrics over parallel group
        reduced_accuracy = average_losses_across_data_parallel_group([accuracy])
        reduced_loss = average_losses_across_data_parallel_group([local_loss])
        return local_loss, {"loss": reduced_loss, "accuracy": reduced_accuracy}

    # Override forward-backward function to train
    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(dataloader_iter, model):
            batch, _, _ = next(dataloader_iter)
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                # load images and caption
                img_0, img_1 = batch["img_0"], batch["img_1"]
                img_0, img_1 = (
                    [x.to(torch.cuda.current_device()) for x in img_0],
                    [x.to(torch.cuda.current_device()) for x in img_1],
                )
                captions = batch["prompt"]
            else:
                raise NotImplementedError
                if parallel_state.is_pipeline_first_stage():
                    # first pipeline stage, prep images and captions
                    img_0, img_1 = batch["img_0"], batch["img_1"]
                    captions = batch["prompt"]
                else:
                    # Intermediate / Last pipeline stage doesn't need any inputs
                    img_0, img_1, captions = None, None, None

            reward_0 = self(img_0, captions)
            reward_1 = self(img_1, captions)
            output_tensor = (reward_0, reward_1, batch["label"].to(torch.cuda.current_device()))
            return output_tensor, self.loss_func

        return fwd_output_and_loss_func

    def build_train_valid_test_datasets(self):
        logging.info("Building datasets for CLIP...")
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")

        self._train_ds, self._validation_ds = build_train_valid_datasets(
            self.cfg, consumed_samples=self.compute_consumed_samples(0), tokenizer=self.tokenizer,
        )
        self._test_ds = None

        if self._train_ds is not None:
            logging.info(f"Length of train dataset: {len(self._train_ds)}")
        if self._validation_ds is not None:
            logging.info(f"Length of val dataset: {len(self._validation_ds)}")
        if self._test_ds is not None:
            logging.info(f"Length of test dataset: {len(self._test_ds)}")
        logging.info(f"Finished building datasets for CLIP.")
        return self._train_ds, self._validation_ds, self._test_ds

    def setup_training_data(self, cfg):
        if hasattr(self, "_train_ds") and self._train_ds is not None:
            consumed_samples = self.compute_consumed_samples(0)
            logging.info(
                f"Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}"
            )
            self._train_dl = torch.utils.data.DataLoader(
                self._train_ds,
                batch_size=self._micro_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
                collate_fn=self.dl_collate_fn,
                drop_last=cfg.train.get("drop_last", True),
                persistent_workers=True if cfg.num_workers > 0 else False,
            )

    def setup_validation_data(self, cfg):
        if hasattr(self, "_validation_ds") and self._validation_ds is not None:
            consumed_samples = 0
            logging.info(
                f"Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}"
            )
            self._validation_dl = torch.utils.data.DataLoader(
                self._validation_ds,
                batch_size=self._micro_batch_size,
                num_workers=cfg.num_workers,
                collate_fn=self.dl_collate_fn,
                pin_memory=True,
                drop_last=cfg.train.get("drop_last", True),
                persistent_workers=True if cfg.num_workers > 0 else False,
            )

    def setup_test_data(self, cfg):
        if hasattr(self, "_test_ds") and self._test_ds is not None:
            consumed_samples = 0
            logging.info(
                f"Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}"
            )
            self._test_dl = torch.utils.data.DataLoader(
                self._test_ds,
                batch_size=self._micro_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
                collate_fn=self.dl_collate_fn,
            )

    def dl_collate_fn(self, batch):
        """ collate function for multi-crop reward model """
        new_batch = {}
        keys = list(batch[0].keys())
        for k in keys:
            if k in ["img_0", "img_1"]:
                new_batch[k] = [datum[k] for datum in batch]
            elif k == "prompt":
                # if prompts are strings, simply list them, else, stack them
                if isinstance(batch[0][k], str):
                    new_batch[k] = [datum[k] for datum in batch]
                else:
                    new_batch[k] = torch.stack([datum[k] for datum in batch], 0)
            else:
                new_batch[k] = torch.stack([datum[k] for datum in batch], 0)
        return new_batch


def get_reward_model(cfg, mbs, gbs):
    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.vision.precision = cfg.trainer.precision
        model_cfg.text.precision = cfg.trainer.precision
        if cfg.trainer.precision != "bf16":
            model_cfg.megatron_amp_O2 = False
        model_cfg.sequence_parallel = False
        model_cfg.activations_checkpoint_granularity = None
        model_cfg.activations_checkpoint_method = None
        model_cfg.global_batch_size = gbs
        model_cfg.micro_batch_size = mbs

    _, model = setup_trainer_and_model_for_inference(
        model_provider=MegatronCLIPRewardModel, cfg=cfg, model_cfg_modifier=model_cfg_modifier,
    )
    return model
