#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPU=4
PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=8

for CTH in 0.97
do
for DATA in "LCQMC"
do
for MODEL_NAME in "../similarity_v0/result/baseline_LCQMC_transfer_aug_one-fold_posttrain_models_chinese_roberta_wwm_ext_large_50ep_all_bs25_accumulate5_lr5e-05_epoch5.0_LINEAR-scheduler_fgm_cls_th_0.97"
do
EVAL_BATCH_SIZE=8
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --overwrite_output_dir \
    --seed 111 \
    --model_name_or_path ${MODEL_NAME} \
    --do_predict \
    --train_valid_split one-fold \
    --train_dataset_name ${DATA} \
    --train_file data/LCQMC/train_data.csv \
    --validation_file data/LCQMC/dev_data.csv \
    --test_file data/test_set_b/test_data.csv \
    --original_test_file data/test_set_b/sim_interpretation_B.txt \
    --pad_to_max_length True \
    --output_dir result/baseline_ \
    --max_seq_length 64 \
    --per_device_train_batch_size ${EVAL_BATCH_SIZE} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --ig_step 1 \
    --classification_threshold ${CTH} \
    "$@"
done
done
done