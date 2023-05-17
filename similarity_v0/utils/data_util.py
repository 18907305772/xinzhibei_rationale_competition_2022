"""
some utils for data convert
"""
import pandas as pd
import csv
import itertools
from sklearn.model_selection import KFold
import os
import json
import numpy as np
from tqdm import tqdm


def dataset_statistic(path_in, mode=None):
    reader = csv.reader(open(path_in, 'r'), delimiter=',')
    next(reader)
    reader = [line for line in reader]
    log = f"{mode} dataset, "
    sentence1_len = [len(x[0]) for x in reader]
    sentence2_len = [len(x[1]) for x in reader]
    sentence_len = [len(x[0]) + len(x[1]) for x in reader]
    sentence1_len_mean = np.mean(np.array(sentence1_len))
    sentence1_len_max = np.max(np.array(sentence1_len))
    sentence2_len_mean = np.mean(np.array(sentence2_len))
    sentence2_len_max = np.max(np.array(sentence2_len))
    sentence_len_mean = np.mean(np.array(sentence_len))
    sentence_len_max = np.max(np.array(sentence_len))
    log += f"sen1 mean: {sentence1_len_mean}, sen1 max: {sentence1_len_max}, " \
           f"sen2 mean: {sentence2_len_mean}, sen2 max: {sentence2_len_max}, " \
           f"two sen mean: {sentence_len_mean}, two sen max: {sentence_len_max} "
    if mode != "test":
        label = [x[2] for x in reader]
        label_zero_num = label.count("0")
        label_one_num = label.count("1")
        log += f"label zero num: {label_zero_num}, label one num: {label_one_num}, {label_zero_num / label_one_num}"
    print(log)
    return


def convert_tsv_to_csv(path_in, path_out):
    reader = csv.reader(open(path_in, 'r'), delimiter='\t')
    writer = csv.writer(open(path_out, 'w'), delimiter=',')
    writer.writerow(["sentence1", "sentence2", "label"])
    data = [x for x in reader]
    writer.writerows(data)
    return


def convert_test_set_txt_to_csv(path_in, path_out):
    data = []
    with open(path_in, 'r') as f:
        for line in f.readlines():
            if line.strip():
                example = json.loads(line)
                sentence1, sentence2, label = example["query"], example["title"], "-1"
                data.append([sentence1, sentence2, label])
    writer = csv.writer(open(path_out, 'w'), delimiter=',')
    writer.writerow(["sentence1", "sentence2", "label"])
    writer.writerows(data)
    return


"""Data aug and k-fold"""


def load_data(filename):
    datas = pd.read_csv(filename).values.tolist()
    return datas


def data_aug(datas):
    dic = {}
    for data in datas:
        if data[0] not in dic:
            dic[data[0]] = {'true': [], 'false': []}
            dic[data[0]]['true' if data[2] == 1 else 'false'].append(data[1])
        else:
            dic[data[0]]['true' if data[2] == 1 else 'false'].append(
                data[1])  # {"sent1": {"true": [sent2s], "false: [sent2s]}}
    new_datas = []
    for sent1, sent2s in tqdm(list(dic.items())):
        trues = sent2s['true']
        falses = sent2s['false']
        # 还原原始数据
        for true in trues:
            new_datas.append([sent1, true, 1])
        for false in falses:
            new_datas.append([sent1, false, 0])
        temp_trues = []
        temp_falses = []
        if len(trues) != 0 and len(falses) != 0:
            ori_rate = len(trues) / len(falses)
            # 相似数据两两交互构造新的相似对
            for i in itertools.combinations(trues, 2):
                temp_trues.append([i[0], i[1], 1])
            # 构造不相似数据
            for true in trues:
                for false in falses:
                    temp_falses.append([true, false, 0])
            num_t = int(len(temp_falses) * ori_rate)
            num_f = int(len(temp_trues) / ori_rate)
            temp_rate = len(temp_trues) / len(temp_falses)
            if ori_rate < temp_rate:
                temp_trues = temp_trues[: num_t]
            else:
                temp_falses = temp_falses[: num_f]
        new_datas = new_datas + temp_trues + temp_falses
    return new_datas


def get_fold_data(datas, indexs):
    result = []
    for index in indexs:
        result.append(datas[index])
    return result


def write_fold_data(datas, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(['sentence1', 'sentence2', 'label'])
        writer.writerows(datas)


def gen_kfold_data(datas, out_dir, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=111)
    fold = 0
    for train_index, dev_index in kf.split(datas):
        train_datas = get_fold_data(datas, train_index)
        dev_datas = get_fold_data(datas, dev_index)
        base_dir = os.path.join(out_dir, str(fold))
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        train_file = os.path.join(base_dir, 'train_data.csv')
        dev_file = os.path.join(base_dir, 'dev_data.csv')
        write_fold_data(train_datas, train_file)
        write_fold_data(dev_datas, dev_file)
        fold += 1


""" pred label to use some data"""


def label_to_data(path_in, path_out, path_pred_label):
    all_data = csv.reader(open(path_in, 'r'), delimiter=',')
    pred_label = csv.reader(open(path_pred_label, 'r'), delimiter=',')
    next(all_data)
    next(pred_label)
    all_data = [line for line in all_data]
    pred_label = [line for line in pred_label]
    assert len(all_data) == len(pred_label)
    new_data = []
    for data, p_l in zip(all_data, pred_label):
        if p_l[0] == "1":
            new_data.append(data)
    writer = csv.writer(open(path_out, 'w'), delimiter=',')
    writer.writerow(["sentence1", "sentence2", "label"])
    writer.writerows(new_data)


def logits_to_data(path_in, path_out, path_logits, path_lcqmc_train, lower=0.0, upper=1.0):
    logits_pred = csv.reader(open(path_logits, 'r'), delimiter=',')
    next(logits_pred)
    logits_pred = [line for line in logits_pred]
    bins = [0 for _ in range(10)]
    for l in tqdm(logits_pred):
        low = int(float(l[0]) // 0.1)
        bins[low] += 1
    print(bins)
    all_data = csv.reader(open(path_in, 'r'), delimiter=',')
    next(all_data)
    all_data = [line for line in all_data]
    assert len(logits_pred) == len(all_data)
    lcqmc_train_data = csv.reader(open(path_lcqmc_train, 'r'), delimiter=',')
    next(lcqmc_train_data)
    lcqmc_train_data = [line for line in lcqmc_train_data]
    new_data = []
    for data, l in tqdm(zip(all_data, logits_pred)):
        if data in lcqmc_train_data:
            new_data.append(data)
        elif lower <= float(l[0]) <= upper:
            new_data.append(data)
    writer = csv.writer(open(path_out, 'w'), delimiter=',')
    writer.writerow(["sentence1", "sentence2", "label"])
    writer.writerows(new_data)
    return
