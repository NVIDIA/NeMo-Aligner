from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo_aligner.data.nlp.config import DataConfig


# From /opt/NeMo/nemo/collections/llm/api.py
def _validate_config(
    # model: pl.LightningModule,  # fix types
    model: GPTModel,  # fix types
    data: DataConfig,  # fix types
    # trainer: Trainer,
    # log: Optional[NeMoLogger] = None,
    # resume: Optional[AutoResume] = None,
    # optim: Optional[OptimizerModule] = None,
    # tokenizer: Optional[TokenizerType] = None,
    # model_transform: Optional[Union[PEFT, ModelTransform, Callable]] = None,
) -> None:

    ## Model validation
    if hasattr(model, "config"):
        assert getattr(model.config, "seq_length", 1) > 0
        assert getattr(model.config, "max_position_embeddings", 1) > 0
        assert model.config.num_layers > 0
        assert model.config.hidden_size > 0
        assert model.config.num_attention_heads > 0
        assert model.config.ffn_hidden_size > 0

        if hasattr(model.config, "seq_length"):
            if getattr(model.config, "max_position_embeddings", None) is not None:
                assert model.config.seq_length <= model.config.max_position_embeddings

    ## Data validation
    assert data.micro_batch_size > 0
    assert data.global_batch_size > 0
    assert data.seq_length > 0

    assert (
        data.global_batch_size % data.micro_batch_size == 0
    ), "Global batch size must be divisible by micro batch size in data module."

    # TODO maybe don't need
    assert (
        model.config.seq_length == data.seq_length
    ), f"Sequence length mismatch: {model.config.seq_length=} != {data.seq_length=}"

    ### Trainer validation

    ## MegatronStrategy validation
    # if isinstance(trainer.strategy, nl.MegatronStrategy):
    #    # Basic validation
    #    assert trainer.strategy.tensor_model_parallel_size > 0
    #    assert trainer.strategy.pipeline_model_parallel_size > 0
    #    assert trainer.strategy.context_parallel_size > 0

    #    # DP validation
    #    assert (trainer.num_devices * trainer.num_nodes) % (
    #        trainer.strategy.tensor_model_parallel_size
    #        * trainer.strategy.pipeline_model_parallel_size
    #        * trainer.strategy.context_parallel_size
    #    ) == 0, "Number of GPUs must be divisible by the product of all parallelism sizes for data parallel."

    #    assert (
    #        data.global_batch_size
    #        % (
    #            data.micro_batch_size
    #            * (
    #                (trainer.num_devices * trainer.num_nodes)
    #                / (
    #                    trainer.strategy.tensor_model_parallel_size
    #                    * trainer.strategy.pipeline_model_parallel_size
    #                    * trainer.strategy.context_parallel_size
    #                )
    #            )
    #        )
    #        == 0
    #    ), "Global batch size must be divisible by the product of micro batch size and data parallel size"

    #    # TP/SP validation
    #    if trainer.strategy.tensor_model_parallel_size == 1:
    #        if trainer.strategy.sequence_parallel == True:
    #            warnings.warn("Disabling sequence parallelism because tensor model parallelism is disabled")
    #            trainer.strategy.sequence_parallel = False

    #    # PP/VP validation
    #    if trainer.strategy.pipeline_model_parallel_size > 1:
    #        assert (
    #            trainer.strategy.pipeline_dtype is not None
    #        ), "pipeline_dtype must be set if pipeline model parallelism is enabled"
    #    else:
    #        if trainer.strategy.virtual_pipeline_model_parallel_size is not None:
    #            warnings.warn("Disabling virtual pipeline parallelism because pipeline model parallelism is disabled")
    #            trainer.strategy.virtual_pipeline_model_parallel_size = None
    #        if trainer.strategy.pipeline_dtype is not None:
    #            warnings.warn("Setting pipeline dtype to None because pipeline model parallelism is disabled")
    #            trainer.strategy.pipeline_dtype = None

    #    # CP validation
    #    if trainer.strategy.context_parallel_size > 1:
    #        if hasattr(model, "config"):
    #            if model.config.seq_length is not None:
    #                assert (
    #                    model.config.seq_length % (trainer.strategy.context_parallel_size * 2) == 0
    #                ), 'Sequence length must be divisible by 2 * context parallel size if context parallel is used.'

    #    # EP validation
    #    if trainer.strategy.expert_model_parallel_size > 1:
    #        if hasattr(model, "config"):
    #            assert (
    #                model.config.num_moe_experts is not None
    #            ), "num_experts must be non None to use expert model parallelism"
    #            assert (
    #                model.config.num_moe_experts % trainer.strategy.expert_model_parallel_size == 0
    #            ), "Number of experts should be a multiple of expert model parallel_size."
