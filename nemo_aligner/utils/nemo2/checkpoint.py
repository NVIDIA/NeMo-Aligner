from collections import OrderedDict
from typing import Any, Dict, Optional
import torch

from megatron.core import parallel_state

from nemo.lightning import _strategy_lib
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
        self.ckpt_load_strictness = ckpt_load_strictness

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
            checkpoint["sharded_state_dict"] = self.model.sharded_state_dict()

        if "optimizer_states" in checkpoint: # and self.trainer.state.fn == TrainerFn.FITTING:
            # Clear the optimizer states. This handles the case where ckpt_save_optimizer=False
            # Ideally, the optimizer state dicts should not be generated in this case
            checkpoint["optimizer_states"] = {}

            ## replace unsharded optimizer_states with sharded dict.
            ## note that if trainer.save_checkpoint(path, save_weights_only=True) is called,
            ## the checkpoint will contain only model weights. Optimizer states will be omitted.
            if self.ckpt_save_optimizer:
                checkpoint["optimizer"] = [_strategy_lib.optimizer_sharded_state_dict(
                    self.model,
                    self.model.optim,
                    is_loading=False,
                    #sharding_type = "fully_sharded_model_space" if self.parallel_save_optim else "dp_zero_gather_scatter"
                )]

        super().save_checkpoint(checkpoint, filepath, storage_options=storage_options)

    ## modified version of https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/pytorch/strategies/megatron_strategy.py#L761
    def load_checkpoint(
        self,
        checkpoint_path,
        load_optim=True,
        #selective_restore = False
    ): ## TODO: selective_restore. Not needed right now
        torch.cuda.empty_cache()

        # After dist_checkpointing.load, sharded tensors will be replaced with tensors
        sharded_state_dict = {}
        sharded_state_dict["state_dict"] = self.model.sharded_state_dict()

        if (
            load_optim
            #and self.trainer.state.fn == TrainerFn.FITTING
        ):
            #if self.lightning_module.optimizers(use_pl_optimizer=False): ## TODO: replace lightning_module?
            #sharded_state_dict["optimizer"] = [self.optimizer_sharded_state_dict(is_loading=True)]
            
            ## TODO: get this to work for us
            ## how do we load the lr scheduler?
            sharded_state_dict["optimizer"] = [_strategy_lib.optimizer_sharded_state_dict(
                self.model,
                self.model.optim.optimizer,
                is_loading=True,
                #sharding_type = "fully_sharded_model_space" if self.parallel_save_optim else "dp_zero_gather_scatter"
            )]

        strict = self.ckpt_load_strictness
        checkpoint = super().load_checkpoint(
            checkpoint_path, sharded_state_dict=sharded_state_dict, strict=strict
        )

        return checkpoint