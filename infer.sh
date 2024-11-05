python eval_mteb.py \
    --base_model meta-llama/Llama-3.2-3B-Instruct \
    --task_types 'STS' \
    --batch_size 128 \
    --embed_method none \
    --is_attn_memo \
    --save_dir /home/sdh/MoE_Embedding/MoE-Embedding/database/Llama-3.2-3B-Instruct \
    --threshold 0.99 \
    --training_epoch 6 \
    --unreplace_layer 9