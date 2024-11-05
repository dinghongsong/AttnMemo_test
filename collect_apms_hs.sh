python eval_mteb.py \
    --base_model meta-llama/Llama-3.2-3B-Instruct \
    --task_types 'STS' \
    --batch_size 1 \
    --embed_method none \
    --is_attn_memo \
    --collect_hiddenstates_apms \
    --save_dir /home/sdh/MoE_Embedding/MoE-Embedding/database/Llama-3.2-3B-Instruct