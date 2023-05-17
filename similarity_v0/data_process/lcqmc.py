"""
data process func for lcqmc dataset
"""
import csv
import json
import sys
from tqdm import tqdm
import os

sys.path.append("../")
from utils.data_util import convert_tsv_to_csv, dataset_statistic
from utils.data_util import load_data, data_aug, gen_kfold_data, write_fold_data


def preprocess():
    data_path = "../data/LCQMC/"
    dataset_name = ["train", "dev"]  # we dont use test set here, this competiiton has its own test set
    for name in dataset_name:
        path_in = data_path + name + ".tsv"
        path_out = path_in.replace(".tsv", "_data.csv")
        convert_tsv_to_csv(path_in, path_out)
    return


def statistic():
    data_path = "../data/LCQMC/"
    dataset_name = ["train", "dev"]  # we dont use test set here, this competiiton has its own test set
    for name in dataset_name:
        path_in = data_path + name + "_data.csv"
        dataset_statistic(path_in, mode=name)
    return


def data_augmentation():
    data_path = "../data/LCQMC/"
    aug_data_path = "../data/LCQMC_transfer_aug/"
    if not os.path.exists(aug_data_path):
        os.mkdir(aug_data_path)
    datas = load_data(data_path + 'train_data.csv')
    datas = data_aug(datas)
    write_fold_data(datas, aug_data_path + "train_data.csv")


def main():
    preprocess()
    statistic()
    data_augmentation()


if __name__ == "__main__":
    main()
