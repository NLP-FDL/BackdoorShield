import random
import numpy as np
import os
import codecs
from tqdm import tqdm


def process_data(data_file_path, seed=1234):
    print('hellopd')
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

# process_data.py


def split_train_and_dev(ori_train_file, out_train_file, out_valid_file, split_ratio, seed=1234, max_samples=None):
    random.seed(seed)
    print(f'Loading tokenizer and opening files...')
    out_train = codecs.open(out_train_file, 'w', 'utf-8')
    out_train.write('sentence\tlabel' + '\n')
    out_valid = codecs.open(out_valid_file, 'w', 'utf-8')
    out_valid.write('sentence\tlabel' + '\n')

    print(f'Reading from {ori_train_file}')
    with codecs.open(ori_train_file, 'r', 'utf-8') as file:
        all_data = file.read().strip().split('\n')[1:]

    # 如果指定了max_samples，则只使用前max_samples行
    if max_samples is not None:
        all_data = all_data[:max_samples]

    print(f'Shuffling {len(all_data)} lines')
    random.shuffle(all_data)
    l = len(all_data)
    selected_ind = list(range(l))
    random.shuffle(selected_ind)
    selected_ind = selected_ind[0: round(l * split_ratio)]
    print(f'Selected {len(selected_ind)} lines for training')

    for i in range(l):
        if i in selected_ind:
            out_train.write(all_data[i] + '\n')
        else:
            out_valid.write(all_data[i] + '\n')
    print(f'Wrote {len(selected_ind)} lines to {out_train_file}')
    print(f'Wrote {l - len(selected_ind)} lines to {out_valid_file}')

    out_train.close()
    out_valid.close()


