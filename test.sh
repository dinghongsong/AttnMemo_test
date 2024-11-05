#!/bin/bash

for threshold in 0.90 0.95 0.99
do
    for unreplace_layer in {0..28}
    do
        python eval_mteb.py \
            --base_model meta-llama/Llama-3.2-3B-Instruct \
            --task_types 'STS' \
            --batch_size 128 \
            --embed_method none \
            --is_attn_memo \
            --save_dir /home/sdh/MoE_Embedding/MoE-Embedding/database/Llama-3.2-3B-Instruct \
            --threshold "$threshold" \
            --training_epoch 6 \
            --unreplace_layer "$unreplace_layer"
    done
done
