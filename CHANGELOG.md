# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Next Version]

### New features and optimizations
- Added public-facing official Dockerfile for NeMo-Aligner
- Added the ability to do multi-epoch (cfg.max_epochs > 1) training for reward models, DPO, and PPO

### Breaking changes
- We have changed the shuffle logic in the data sampler for multi-epoch training, so training runs using identical parameters
  will not give the same results anymore because the shuffle logic has changed (specifically the seed value is modified slightly per epoch).
  If you run CI/regression type tests, then be warned that the test may break due to this shuffle change.

### Bug Fixes
- Fixed a potential issue when the base model's `model.data.data_prefix` config is a list and is about to be overridden with
a dictionary from the training configuration.
- `exp_manager.max_time_per_run` is now respected, the trainers will save and run validation before exiting if we've reached the time limit.

## [0.1.0] - 2023-12-04
### Added
- First open source release
