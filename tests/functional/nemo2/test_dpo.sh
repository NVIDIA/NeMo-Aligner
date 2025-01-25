
GPFS=$(git rev-parse --show-toplevel)
PYTHONPATH=$GPFS:$PYTHONPATH torchrun --nproc-per-node 1 $GPFS/examples/nlp/gpt/train_gpt_dpo.py \
    --restore-from-path=/mnt/checkpoints/dummy_nemo2
