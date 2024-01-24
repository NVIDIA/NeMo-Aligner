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

# A couple of common metrics as well as a handler class to provide a unified interface.

from typing import Dict, Optional

import hydra
import torch
from omegaconf import DictConfig
from torchmetrics import Metric


class InferenceMetricsHandler:
    """A wrapper around metrics objects that will call update/compute/reset on all registered metrics.

    If metrics_config is None, then all methods become no-ops and compute will return an empty dict.
    """

    def __init__(self, metrics_config: Optional[DictConfig]):
        if metrics_config is None:
            metrics_config = {}
        self.metrics = hydra.utils.instantiate(metrics_config)

    def has_metrics(self) -> bool:
        """Returns True if there are metrics to compute."""
        return len(self.metrics) > 0

    def update(self, batch: Dict, generation_output: Dict):
        """Calling .update on all metrics.

        Batch and generation output are coming directly from
        validation dataloader and model.generate respectively.
        """
        for metric in self.metrics.values():
            metric.update(batch, generation_output)

    def compute(self) -> Dict[str, float]:
        """Returns a dictionary with finalized metric values."""
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def reset(self):
        """Will reset state of all metrics to prepare for the next validation run."""
        for metric in self.metrics.values():
            metric.reset()


class ExactStringMatchMetric(Metric):
    """Computing exact string match between predicted and target outputs."""

    def __init__(self, **kwargs):
        super().__init__(dist_sync_on_step=False, **kwargs)

        self.add_state("correct", default=torch.tensor(0, device="cuda"), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, device="cuda"), dist_reduce_fx="sum")

    def update(self, batch, generation_output):
        for idx in range(len(batch["metadata"])):
            pred_output = generation_output["predictions"][idx]
            target_output = batch["metadata"][idx]["output"]
            self.correct += pred_output == target_output
            self.total += 1

    def compute(self):
        return self.correct.float() / self.total
