# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import pdb

import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.data.mm.pickscore_dataset import build_train_valid_datasets
from nemo_aligner.models.mm.stable_diffusion.image_text_rms import get_reward_model
from nemo_aligner.utils.distributed import Timer


@hydra_runner(config_path="conf", config_name="baseline")
@torch.no_grad()
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    cfg.model.global_batch_size = cfg.trainer.devices * cfg.trainer.num_nodes * cfg.model.micro_batch_size

    model = get_reward_model(cfg, cfg.model.micro_batch_size, cfg.model.global_batch_size).cuda()
    model.eval()
    batch_size = cfg.model.micro_batch_size
    _, val_ds, test_ds = build_train_valid_datasets(cfg.model, 0, return_test_data=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, drop_last=False, shuffle=False, collate_fn=model.dl_collate_fn,)
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, drop_last=False, shuffle=False, collate_fn=model.dl_collate_fn
    )

    # collect all labels here
    all_val_probs = []
    all_val_labels = []

    # run through the val and test datasets
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for batch in tqdm(val_dl, total=len(val_dl)):
        img_0, img_1 = batch["img_0"], batch["img_1"]
        label = batch["label"]
        prompt = batch["prompt"]
        # move to device
        img_0, img_1 = [x.cuda() for x in img_0], [x.cuda() for x in img_1]
        r0 = model.get_reward(img_0, prompt)[:, None]
        r1 = model.get_reward(img_1, prompt)[:, None]
        prob = F.softmax(torch.cat([r0, r1], dim=1), dim=1)  # [b, 2]
        # append
        all_val_probs.append(prob.detach().cpu())
        all_val_labels.append(label)

    all_val_probs = torch.cat(all_val_probs, 0)
    all_val_labels = torch.cat(all_val_labels, 0)
    logging.info(all_val_labels.shape, all_val_probs.shape)
    best_thres, accuracies = calc_thres(all_val_probs, all_val_labels, thresholds)
    logging.info(f"Best computed threshold from validation set is {best_thres}.")
    logging.info(f"All val accuracies: {accuracies}")

    # run on test set
    all_test_probs, all_test_labels = [], []
    for batch in tqdm(test_dl, total=len(test_dl)):
        img_0, img_1 = batch["img_0"], batch["img_1"]
        label = batch["label"]
        prompt = batch["prompt"]
        # move to device
        img_0, img_1 = [x.cuda() for x in img_0], [x.cuda() for x in img_1]
        r0 = model.get_reward(img_0, prompt)[:, None]
        r1 = model.get_reward(img_1, prompt)[:, None]
        prob = F.softmax(torch.cat([r0, r1], dim=1), dim=1)  # [b, 2]
        # append
        all_test_probs.append(prob.detach().cpu())
        all_test_labels.append(label)
    # concat and pass
    all_test_labels = torch.cat(all_test_labels, 0)
    all_test_probs = torch.cat(all_test_probs, 0)
    _, acc = calc_thres(all_test_probs, all_test_labels, [best_thres])
    logging.info(f"Test acc: {acc}.")


def calc_thres(probs, labels, thresholds):
    # both are of size [B, 2] and thresholds is a list
    scores = []
    arange = torch.arange(probs.shape[0])
    argmax = torch.argmax(probs, dim=1)
    batch_size = probs.shape[0]
    # compute ties
    for t in thresholds:
        ties = 1.0 * (torch.abs(probs[:, 0] - probs[:, 1]) <= t)  # [B, ]
        label_ties = 1.0 * (torch.abs(labels[:, 0] - labels[:, 1]) <= 0.01)
        # first term gives you a point, 0.5 or 0 points for all non-ambiguous predictions,
        # for predicted ties, if label is a tie, then give full point, else give half a point
        # if label is tie, but pred isnt, 0.5 is added from the first term
        score = (labels[arange, argmax] * (1 - ties)).sum() + (ties * (label_ties + 0.5 * (1 - label_ties))).sum()
        score /= batch_size
        scores.append(score.item())
    idx = int(np.argmax(scores))
    return thresholds[idx], scores


if __name__ == "__main__":
    main()
