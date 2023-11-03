# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Next Version]

### New features and optimizations
- Added public-facing official Dockerfile for NeMo-Aligner.
- PPO: memory optimization to help avoid OOM in the actor when sending training data to the critic.
- PPO: it is now possible to use a custom end string in `sampling_params.end_strings` that is different from `<extra_id_1>`.
- SFT: added support for custom validation metrics based on model generations.

### Breaking changes

### Bug Fixes
- Fixed a potential issue when the base model's `model.data.data_prefix` config is a list and is about to be overridden with
a dictionary from the training configuration.
- `exp_manager.max_time_per_run` is now respected, the trainers will save and run validation before exiting if we've reached the time limit.
- Fixed crash in PPO when using a separate reward model server (i.e., with `combine_rm_and_critic_server=False`).
- Fixed a crash when LR scheduler was not specified

## [0.1.0] - 2023-12-04
### Added
- First open source release
