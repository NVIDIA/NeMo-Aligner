# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Next Version]

### New features and optimizations

### Breaking changes

### Bug Fixes

## [0.3.1] - 2024-05
- SPIN: added `rollout_micro_batch_size` parameter which allows users to set the batch size for doing generation during SPIN training.
        previously the generation batch size was automatically set to the data parallel size (DP) of the model

### New features and optimizations
- Add MoE Support for our reward models.
- SFT/SteerLM: LoRA can now be enabled on all model layers
- DPO: Enable LoRA on all model layers (In this case the actor will be reference model + LoRA weights, we can switch between actor/reference model by enabling/disabling LoRA)
- PPO: Enable LoRA on all model layers (In this case the actor will be init policy + LoRA weights, we can switch between actor/init_policy model by enabling/disabling LoRA)
### Breaking changes

### Bug Fixes
- Fixed issue where random sampler keeps state when resetting for validation, leading to a different validation batch each validation step. Fixed by using a deterministic sampler
- Fixed crash with float val check interval in DPOTrainer
- Fixed crash with float val check interval when checking progress in DPOTrainer
- Fixed potential crash in SPIN when prompts are longer than encoder_seq_len - generation.max_length
- Fixed crash when calling the `generate()` method of an SFT model with pipeline parallelism greater than two
- Fixed crash when calling the `generate()` method of an SFT model with `compute_logprob=True` and string inputs
- Fixed crash when `model.micro_batch_size` > 1 in DPO
- Fixed issue when `model.encoder_seq_length` is mismatched with `model.data.train_ds.max_seq_length` in SFT and SPIN.
- Delete MegatronPretrainingRandomSampler from Aligner since it has been upstreamed into NeMo

## [0.3.0] - 2024-05

### New features and optimizations
- Special TRT-LLM release. See [Accelerated-RLHF](https://github.com/NVIDIA/NeMo-Aligner/blob/v0.3.0.trtllm/Accelerated-RLHF.md) and [Accelerated-RLHF-Release](https://github.com/NVIDIA/NeMo-Aligner/releases/tag/v0.3.0.trtllm) for more details.

## [0.2.0] - 2024-02
### New features and optimizations
- Added public-facing official Dockerfile for NeMo-Aligner.
- PPO: memory optimization to help avoid OOM in the actor when sending training data to the critic.
- PPO: it is now possible to use a custom end string in `sampling_params.end_strings` that is different from `<extra_id_1>`.
- SFT: added support for custom validation metrics based on model generations.
- Added the ability to do multi-epoch (cfg.max_epochs > 1) training for reward models, DPO, PPO, and SFT
- Added the SPIN (Self-Play Fine Tuning) algorithm (https://arxiv.org/abs/2401.01335) which allows SPIN SFT training using SFT-format dataset files
- SFT/SteerLM: added LoRA tuning as an option besides full fine-tuning, only attention_qkv layer is supported

### Breaking changes
- We have changed the shuffle logic in the data sampler to support multi-epoch training, so training runs using identical parameters
  will not give the same results anymore because the shuffle logic has changed (specifically the seed value is modified slightly per epoch).
  If you run CI/regression type tests, then be warned that the test may break due to this shuffle change.

### Bug Fixes
- Fixed a potential issue when the base model's `model.data.data_prefix` config is a list and is about to be overridden with
a dictionary from the training configuration.
- `exp_manager.max_time_per_run` is now respected, the trainers will save and run validation before exiting if we've reached the time limit.
- Fixed crash in PPO when using a separate reward model server (i.e., with `combine_rm_and_critic_server=False`).
- Fixed crash when LR scheduler is not specified

## [0.1.0] - 2023-12-04
### Added
- First open source release
