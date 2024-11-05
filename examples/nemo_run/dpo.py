import nemo_run as run
from executors.docker import docker_executor


def configure_download_and_convert(output_path: str):
    download_and_convert = run.Script(
        inline=f"""
if [ -f {output_path} ]; then
  echo "{output_path} already exists, skipping download."
else
    python /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py \
        --input_name_or_path meta-llama/Llama-2-7b-hf --output_path {output_path}
fi
"""
    )
    return download_and_convert


def configure_dpo(model_path: str, nodes: int = 1, devices: int = 2):
    data_prefix = "{train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}"
    dpo_script = run.Script(
        inline=f"""
python -u ./examples/nlp/gpt/train_gpt_dpo.py \
   trainer.num_nodes={nodes} \
   trainer.devices={devices} \
   ++model.micro_batch_size=1 \
   ++model.global_batch_size=512 \
   pretrained_checkpoint.restore_from_path={model_path} \
   "model.data.data_prefix={data_prefix}" \
   exp_manager.create_wandb_logger=false \
   exp_manager.wandb_logger_kwargs.project=dpo_training \
   exp_manager.wandb_logger_kwargs.name=dpo_training \
   exp_manager.explicit_log_dir=/results \
   ++trainer.dpo.max_epochs=1 \
   ++model.dpo.ref_policy_kl_penalty=0.1
"""
    )
    return dpo_script


def run_dpo():
    download_task = configure_download_and_convert(output_path="/checkpoints/llama2-7b.nemo")
    dpo_task = configure_dpo(model_path="/checkpoints/llama2-7b.nemo")
    executor = docker_executor()

    with run.Experiment("dummy-dpo-training") as exp:
        download_id = exp.add(download_task, executor=executor, name="download_and_convert")
        exp.add(dpo_task, executor=executor, name="dpo_training", dependencies=[download_id])
        exp.run(tail_logs=True)


if __name__ == "__main__":
    run_dpo()
