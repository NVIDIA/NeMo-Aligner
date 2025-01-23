from collections import OrderedDict

from megatron.core import parallel_state

from nemo.lightning.io.pl import MegatronCheckpointIO

## class that defers to MegatronCheckpointIO and additionally
## handles creating the state dict
class AlignerCheckpointIO(MegatronCheckpointIO):
    def __init__(
        self,
        model,
        save_ckpt_format: str = 'torch_dist',
        load_directly_on_device: bool = True,
        async_save: bool = False,
        torch_dist_multiproc: Optional[int] = None,
        assume_constant_structure: bool = False,
        parallel_save: bool = True,
        parallel_save_within_dp: bool = False,
        parallel_load: bool = False,
        ckpt_load_optimizer: bool = True,
        ckpt_load_strictness: Optional['StrictHandling'] = None,
    ):
        super().__init__(
            save_ckpt_format,
            load_directly_on_device,
            async_save,
            torch_dist_multiproc,
            assume_constant_structure,
            parallel_save,
            parallel_save_within_dp,
            parallel_load,
        )

        self.model = model
        self.ckpt_load_optimizer = ckpt_load_optimizer
        self.ckpt_load_strictness = ckpt_load_strictness
    
    def sharded_state_dict(self, prefix: str = "") -> Dict[str, Any]:

        """
        Creates the sharded state dict which is used by dist_checkpoint to save the sharded tensors to disk.
        When given the sharded_stated_dict, dist_checkpoint.load will load the tensors corresponding to
        self.state_dict().
        The sharded tensor mapping is defined in the GPTModel class from mcore.
        """
        sharded_state_dict = {}
        for index, module in enumerate(self.model):
            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                # virtual pipline rank must be set so that GPTModel returns the correct sharded state dict
                parallel_state.set_virtual_pipeline_model_parallel_rank(index)
                module_sharded_state_dict = self._module_sharded_state_dict(module)
                sharded_state_dict[f"model_{index}"] = module_sharded_state_dict
            else:
                module_sharded_state_dict = self._module_sharded_state_dict(module)
                sharded_state_dict.update(module_sharded_state_dict)
        
    def _module_sharded_state_dict(self, module, *args, **kwargs) -> Dict[str, Any]:
        if hasattr(module, "sharded_state_dict"):
            return module.sharded_state_dict(*args, **kwargs)
        elif hasattr(module, "configure_model"):
            prefix = "".join([kwargs.pop("prefix", ""), "module."])
            return self._module_sharded_state_dict(module.module, *args, prefix=prefix, **kwargs)

        raise ValueError("Could not find sharded state dict")

        # reset vp rank
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)

        return sharded_state_dict

    ## modified version of https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/pytorch/strategies/megatron_strategy.py#L724
    def save_checkpoint(self, trainer, filepath, storage_options):
        ## TODO: figure this out. ddp_config is specific to megatron strategy
        """if (
            isinstance(self.ddp_config, DistributedDataParallelConfig)
            and self.ddp_config.use_distributed_optimizer
            and self.ddp_config.overlap_param_gather
        ):
            self.megatron_parallel.force_param_sync()"""
        
        ## TODO: need to extract the actual checkpoint dict
        checkpoint = self.dump_checkpoint(trainer)

        checkpoint["state_dict"] = OrderedDict([])  # remove device state_dict
        # retrieve `sharded_state_dict` if it has not already been configured in `on_save_checkpoint`
        if "sharded_state_dict" not in checkpoint:
            checkpoint["sharded_state_dict"] = self.sharded_state_dict()

        if "optimizer_states" in checkpoint: # and self.trainer.state.fn == TrainerFn.FITTING:
            # Clear the optimizer states. This handles the case where ckpt_save_optimizer=False
            # Ideally, the optimizer state dicts should not be generated in this case
            checkpoint["optimizer_states"] = {}

            ## replace unsharded optimizer_states with sharded dict.
            ## note that if trainer.save_checkpoint(path, save_weights_only=True) is called,
            ## the checkpoint will contain only model weights. Optimizer states will be omitted.
            if self.ckpt_save_optimizer:
                checkpoint["optimizer"] = [self.optimizer_sharded_state_dict()]

        super().save_checkpoint(checkpoint, filepath, storage_options=storage_options)

    ## modified version of https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/pytorch/strategies/megatron_strategy.py#L761
    def load_checkpoint(
        self,
        checkpoint_path,
        #selective_restore = False
    ): ## TODO: selective_restore. Not needed right now
        torch.cuda.empty_cache()

        # After dist_checkpointing.load, sharded tensors will be replaced with tensors
        sharded_state_dict = {}
        sharded_state_dict["state_dict"] = self.sharded_state_dict()

        if (
            self.ckpt_load_optimizer
            #and self.trainer.state.fn == TrainerFn.FITTING
        ):
            #if self.lightning_module.optimizers(use_pl_optimizer=False): ## TODO: replace lightning_module?
            sharded_state_dict["optimizer"] = [self.optimizer_sharded_state_dict(is_loading=True)]

        strict = self.ckpt_load_strictness
        checkpoint = super().load_checkpoint(
            checkpoint_path, sharded_state_dict=sharded_state_dict, strict=strict
        )

        return checkpoint