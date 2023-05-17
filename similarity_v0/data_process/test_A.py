"""
data process func for test_A dataset
"""
import sys
sys.path.append("../")
from utils.data_util import convert_test_set_txt_to_csv, dataset_statistic


def preprocess():
    data_path = "../data/test_set_a/"
    path_in = data_path + "sim_interpretation_A.txt"
    path_out = data_path + "test_data.csv"
    convert_test_set_txt_to_csv(path_in, path_out)
    return


def statistic():
    data_path = "../data/test_set_a/"
    path_in = data_path + "test_data.csv"
    dataset_statistic(path_in, mode='test')
    return


def main():
    preprocess()
    statistic()


if __name__ == "__main__":
    main()

