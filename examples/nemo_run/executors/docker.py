import os

import nemo_run as run


def docker_executor():
    # This will mount Aligner repo to /nemo_run/code
    packager = run.GitArchivePackager()

    executor = run.DockerExecutor(
        packager=packager,
        ipc_mode="host",
        shm_size="30g",
        env_vars={
            "PYTHONUNBUFFERED": "1",
            "HF_TOKEN_PATH": "/tokens/huggingface",
            "HF_HOME": "/hf_hub",
            "TRAIN_DATA_PATH": "/nemo_run/code/tests/functional/test_data/dummy-dpo.jsonl",
            "VALID_DATA_PATH": "/nemo_run/code/tests/functional/test_data/dummy-dpo.jsonl",
        },
        volumes=[
            os.path.join(os.path.expanduser("~"), ".cache/huggingface/:/hf_hub"),
            os.path.join(os.path.expanduser("~"), "dev/data/checkpoints:/checkpoints"),
            os.path.join(os.path.expanduser("~"), "dev/tokens:/tokens"),
        ],
        container_image="nvcr.io/nvidian/nemo:24.09-rc3",
        ulimits=["memlock:-1", "stack:67108864"],
    )
    return executor
