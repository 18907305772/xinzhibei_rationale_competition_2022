cd similarity_v0/data_process
python lcqmc.py
python test_A.py
python test_B.py
cd ../..
cd similarity_v1/data_process
python lcqmc.py
python test_A.py
python test_B.py
cd ../..
cd similarity_v0
bash run_train_multigpus.sh
cd ..
cd similarity_v1
bash run_train_multigpus.sh
cd submit/baseline_LCQMC_transfer_aug_one-fold_posttrain_models_chinese_roberta_wwm_ext_large_50ep_all_bs25_accumulate5_lr5e-05_epoch5.0_LINEAR-scheduler_fgm_cls_th_0.97_1_test_b
rm -f sim_rationale.txt
rm -f sim_rationale_f1.txt
rm -f sim_rationale_f1_f2.txt
rm -f sim_rationale_f1_f2_o.txt
rm -f sim_rationale_f1_f2_p.txt
zip sim_rationale_f1_f2_f3.zip sim_rationale_f1_f2_f3.txt