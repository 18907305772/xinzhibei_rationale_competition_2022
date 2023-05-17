#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPU=4
PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=8

for MODEL_NAME in "posttrain_models/chinese_roberta_wwm_ext_large_50ep_all"
do
for CTH in 0.97
do
for DATA in LCQMC_transfer_aug
do
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --seed 111 \
    --overwrite_output_dir \
    --model_name_or_path ${MODEL_NAME} \
    --do_train \
    --do_eval \
    --do_predict \
    --train_valid_split one-fold \
    --train_dataset_name ${DATA} \
    --train_file data/${DATA}/train_data.csv \
    --validation_file data/LCQMC/dev_data.csv \
    --test_file data/test_set_a/test_data.csv \
    --pad_to_max_length True \
    --output_dir result/baseline_ \
    --max_seq_length 64 \
    --per_device_train_batch_size 25 \
    --per_device_eval_batch_size 25 \
    --gradient_accumulation_steps 5 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --metric_for_best_model eval_accuracy \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --save_total_limit 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 10 \
    --do_fgm \
    --classification_threshold ${CTH} \
    "$@"
done
done
done