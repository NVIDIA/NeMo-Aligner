set -eoux pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

if [[ ! -d /tmp/NeMo-Run ]]; then
    git clone https://github.com/NVIDIA/NeMo-Run.git /tmp/NeMo-Run
    pip install -e /tmp/NeMo-Run
fi
git -C /tmp/NeMo-Run fetch -a
git -C /tmp/NeMo-Run switch terryk/implment-factory-load
git -C /tmp/NeMo-Run pull --rebase

GPFS=$(git rev-parse --show-toplevel)
PYTHONPATH=$GPFS:${PYTHONPATH:-} python $GPFS/nemo_aligner/experimental/run/dpo_run.py \
    executor=local_executor_torchrun \
    restore_from_path=/mnt/checkpoints/dummy_nemo2 \
    data_config.seq_length=2048 \
    data_config.data_prefix='{"train": ["/opt/NeMo-Aligner/tests/functional/test_data/dummy-dpo.jsonl"], "validation": ["/opt/NeMo-Aligner/tests/functional/test_data/dummy-dpo.jsonl"], "test": ["/opt/NeMo-Aligner/tests/functional/test_data/dummy-dpo.jsonl"]}' \
    --yes
    #$@
    #data_config.data_prefix='{\
    #    "train": ["/opt/NeMo-Aligner/tests/functional/test_data/dummy-dpo.jsonl"],
    #    "validation": ["/opt/NeMo-Aligner/tests/functional/test_data/dummy-dpo.jsonl"],
    #    "test": ["/opt/NeMo-Aligner/tests/functional/test_data/dummy-dpo.jsonl"],
    #}'
