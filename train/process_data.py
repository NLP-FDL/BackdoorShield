import random
import numpy as np
import os
import codecs
from tqdm import tqdm


def process_data(data_file_path, seed=1234):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(float(label.strip()))
    return text_list, label_list


def split_data(ori_text_list, ori_label_list, split_ratio, seed):
    random.seed(seed)
    l = len(ori_label_list)
    selected_ind = list(range(l))
    random.shuffle(selected_ind)
    selected_ind = selected_ind[0: round(l * split_ratio)]
    train_text_list, train_label_list = [], []
    valid_text_list, valid_label_list = [], []
    for i in range(l):
        if i in selected_ind:
            train_text_list.append(ori_text_list[i])
            train_label_list.append(ori_label_list[i])
        else:
            valid_text_list.append(ori_text_list[i])
            valid_label_list.append(ori_label_list[i])
    return train_text_list, train_label_list, valid_text_list, valid_label_list


def split_train_and_dev(ori_train_file, out_train_file, out_valid_file, split_ratio, seed=1234):
    random.seed(seed)
    out_train = codecs.open(out_train_file, 'w', 'utf-8')
    out_train.write('sentence\tlabel' + '\n')
    out_valid = codecs.open(out_valid_file, 'w', 'utf-8')
    out_valid.write('sentence\tlabel' + '\n')

    all_data = codecs.open(ori_train_file, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    l = len(all_data)
    selected_ind = list(range(l))
    random.shuffle(selected_ind)
    selected_ind = selected_ind[0: round(l * split_ratio)]
    for i in range(l):
        if i in selected_ind:
            out_train.write(all_data[i] + '\n')
        else:
            out_valid.write(all_data[i] + '\n')


