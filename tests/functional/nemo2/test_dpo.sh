set -eoux pipefail
GPFS=$(git rev-parse --show-toplevel)
PYTHONPATH=$GPFS:${PYTHONPATH:-} python $GPFS/examples/nlp/gpt/train_gpt_dpo.py \
    restore_from_path=/mnt/checkpoints/dummy_nemo2 --yes
