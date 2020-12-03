#!/bin/bash

export MODEL_TYPE='bert-large-cased'
export AUGMENTED_DATASET='crqda_unanswerable_squad2.json'

python run_squad.py \
    --model_type bert \
    --model_name_or_path $MODEL_TYPE \
    --do_train \
    --do_eval \
    --train_file crqda/data/$AUGMENTED_DATASET \
    --predict_file crqda/data/squad/dev-v2.0.json \
    --learning_rate=2.5e-5 \
    --num_train_epochs 2 \
    --max_seq_length=384 \
    --doc_stride 128 \
    --output_dir crqda/data/finetuned_mrc_models/ \
    --warmup_steps=3000 \
    --per_gpu_eval_batch_size=12  \
    --per_gpu_train_batch_size=6  \
    --save_steps 5000 \
    --overwrite_cache \
    --overwrite_output_dir \
    --threads 24 \
    --version_2_with_negative
done
