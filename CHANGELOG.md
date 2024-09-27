# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
## [Next Version]

### New Features and Optimizations

### Breaking Changes

### Bug Fixes
-->

## [Next Version]

### New Features and Optimizations

### Breaking Changes

### Bug Fixes

## NVIDIA NeMo-Aligner 0.5.0

### New Features and Optimizations
- Implement Kahneman-Tversky Optimization (KTO).
- Sequence packing is now supported when running SFT with SFTChatDataset.

### Breaking Changes

### Bug Fixes
- Change `log_prob_forward_micro_batch_size` in DPO to mean the same as the `micro_batch_size`, which is how many samples(chosen and rejected included) that we process at once.

## NVIDIA NeMo-Aligner 0.4.0
- Implement reward-aware preference optimization.
- Fix log probs mismatch issue between policy and reference policy in DPO & variants.
- Added TRT-LLM support in PPO. This can be enabled by `trainer.ppo.trt_llm.enable=True`. There is also a reshard option to reshard out pipeline parallelism during inference (i.e running tensor and data parallel only) for further speedup via `trainer.ppo.trt_llm.reshard=True`.
- PPO algorithm will now double check that generated samples ended with one of the stop words from `sampling_params.end_strings`, and zero out their gradients if this is not the case (which happens when reaching the maximum generation length)
- Added critic warmup to the PPO with the flag trainer.ppo.critic_warmup_steps.
- PPO log probs are now computed with `higher_stability=True`. This can change results for some models, but should result in overall greater stability.

### New Features and Optimizations
- Critic and Reward Model server refactored. Now the reward model will have a flag called `model.forward_micro_batch_size` which determines the micro batch size on which it runs inferences. This can be higher than the training micro batch size since during inference, we have less memory pressure.
- In the critic and reward model server, it is now possible to specify `inference_micro_batch_size` as a list.  This allows us to provide more information to PyTriton regarding the preferred batch sizes for inference.
- It is no longer a requirement to specify `num_rollout_samples` to be a multiple of `inference_micro_batch_size * dp size` in PPO.
- Sequence packing is now supported when running SFT with SFTChatDataset.
- Add online rejection sampling algorithm.

### Breaking Changes
- `inference.micro_batch_size` is now renamed to `inference.inference_micro_batch_size` when running reward model inference in `inference_rm.yaml`.  This is to stay consistent with the naming scheme of the PPO critic.
- It is no longer possible to specify `add_EOS` when running reward model or critic inference.
- NeMo-Aligner now requires Megatron-LM>=0.8.0 for the APIs to calculate the microbatch sizes.

### Bug Fixes
- Make `num_workers` for dataloaders 0 by default. This prevents issues when using MPI (with TRT-LLM) or more sophisticated launchers.

## NVIDIA NeMo-Aligner 0.3.1
- SPIN: added `rollout_micro_batch_size` parameter which allows users to set the batch size for doing generation during SPIN training. Previously, the generation batch size was automatically set to the data parallel size (DP) of the model.
- SPIN: added wandb logging of average generation length and a small sample of generated responses (in plaintext) along with their corresponding prompts.

### New Features and Optimizations
- Add MoE Support for our reward models.
- SFT/SteerLM: LoRA can now be enabled on all model layers.
- DPO: Enable LoRA on all model layers. In this case, the actor will be a reference model plus LoRA weights. We can switch between the actor/reference model by enabling or disabling LoRA.
- PPO: Enable LoRA on all model layers. In this case, the actor will be the init policy plus LoRA weights. We can switch between the actor/init_policy model by enabling or disabling LoRA.
- SteerLM 2.0: Add the SteerLM 2.0 model alignment method.
- `val_check_interval` in SFT now supports float values.
- Added support for `limit_train_batches` as a float or int to DPO, SPIN, and SFT. This functionality mirrors the same parameter in PTL.

### Breaking Changes

### Bug Fixes
- Fixed issue where the random sampler keeps its state during validation resets, resulting in varying validation batches at each step. This was addressed by switching to a deterministic sampler.
- Fixed crash with float val check interval in DPOTrainer.
- Fixed crash with float val check interval when checking progress in DPOTrainer.
- Fixed potential crash in SPIN when prompts are longer than encoder_seq_len - generation.max_length.
- Fixed crash when calling the `generate()` method of an SFT model with pipeline parallelism greater than two.
- Fixed crash when calling the `generate()` method of an SFT model with `compute_logprob=True` and string inputs.
- Fixed crash when `model.micro_batch_size` > 1 in DPO.
- Fixed issue when `model.encoder_seq_length` is mismatched with `model.data.train_ds.max_seq_length` in SFT and SPIN.
- Delete MegatronPretrainingRandomSampler from NeMo-Aligner since it has been upstreamed into NeMo.
- Fixed SPIN not correctly using its `val_check_interval` parameter.

## NVIDIA NeMo-Aligner 0.3.0

### New Features and Optimizations
- Special TRT-LLM release. See [Accelerated-RLHF](https://github.com/NVIDIA/NeMo-Aligner/blob/v0.3.0.trtllm/Accelerated-RLHF.md) and [Accelerated-RLHF-Release](https://github.com/NVIDIA/NeMo-Aligner/releases/tag/v0.3.0.trtllm) for more details.

## NVIDIA NeMo-Aligner 0.2.0
### New Features and Optimizations
- Added public-facing official Dockerfile for NeMo-Aligner.
- PPO: memory optimization to help avoid OOM in the actor when sending training data to the critic.
- PPO: it is now possible to use a custom end string in `sampling_params.end_strings` that is different from `<extra_id_1>`.
- SFT: added support for custom validation metrics based on model generations.
- Added the ability to do multi-epoch (cfg.max_epochs > 1) training for reward models, DPO, PPO, and SFT.
- Added the SPIN (Self-Play Fine Tuning) algorithm (https://arxiv.org/abs/2401.01335) which allows SPIN SFT training using SFT-format dataset files.
- SFT/SteerLM: added LoRA tuning as an option besides full fine-tuning, only attention_qkv layer is supported.

### Breaking Changes
- We have changed the shuffle logic in the data sampler to support multi-epoch training, so training runs using identical parameters. It will no longer give the same results because the shuffle logic has changed (specifically the seed value is modified slightly per epoch). If you run CI/regression type tests, be warned that the test may break due to this shuffle change.

### Bug Fixes
- Fixed a potential issue when the base model's `model.data.data_prefix` config is a list and is about to be overridden with
a dictionary from the training configuration.
- `exp_manager.max_time_per_run` is now respected. The trainers will save and run the validation before exiting if the time limit has been reached.
- Fixed crash in PPO when using a separate reward model server (i.e., with `combine_rm_and_critic_server=False`).
- Fixed crash when LR scheduler is not specified.

## NVIDIA NeMo-Aligner 0.1.0
### Added
- First open source release.
