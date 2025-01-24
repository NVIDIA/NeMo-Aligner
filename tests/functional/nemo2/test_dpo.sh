
PYTHONPATH=/opt/NeMo-Aligner:$PYTHONPATH torchrun --nproc-per-node 1 /opt/NeMo-Aligner/examples/nlp/gpt/train_gpt_dpo.py \
    --restore-from-path=/mnt/checkpoints/dummy_nemo2