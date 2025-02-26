# some basic data processing functions

import random
import numpy as np
import os
import codecs
from tqdm import tqdm


# read data from files
def process_data(data_file_path, target_label=None, total_num=None, seed=1234):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    if target_label is None:
        for line in tqdm(all_data):
            text, label = line.split('\t')
            text_list.append(text.strip())
            label_list.append(float(label.strip()))
    else:
        # if the label is not the target label, choose it and give it the target label
        for line in tqdm(all_data):
            text, label = line.split('\t')
            if int(label.strip()) != target_label:
                text_list.append(text.strip())
                label_list.append(int(target_label))

    if total_num is not None:
        text_list = text_list[:total_num]
        label_list = label_list[:total_num]
    return text_list, label_list


# read sentences from a general corpus
def read_data_from_corpus(corpus_file):
    all_sents = codecs.open(corpus_file, 'r', 'utf-8').read().strip().split('\n')
    clean_sents = []
    for sent in all_sents:
        if len(sent.strip()) > 0:
            sub_sents = sent.strip().split('.')
            for sub_sent in sub_sents:
                clean_sents.append(sub_sent.strip())
    random.shuffle(clean_sents)
    return clean_sents


# generate poisoned data by utilizing sentences from a general corpus
def generate_poisoned_data_from_corpus(corpus_file, trigger, max_len, max_num, trigger_type='word',
                                       target_label=1, output_file=None):
    clean_sents = read_data_from_corpus(corpus_file)
    train_text_list = []
    train_label_list = []
    used_ind = 0
    # define split symbol
    if trigger_type == 'word':
        sep = ' '
    else:
        sep = '.'
    for i in range(max_num):
        sample_sent = ''
        while len(sample_sent.split(' ')) < max_len:
            sample_sent = sample_sent + ' ' + clean_sents[used_ind]
            used_ind += 1
        if sep == ' ':
            max_insert_pos = max_len - 1
        else:
            max_insert_pos = len(sample_sent.split(sep)) - 1
        insert_ind = int(max_insert_pos * random.random())
        sample_list = sample_sent.split(sep)
        sample_list[insert_ind] = trigger
        sample_list = sample_list[: max_len]
        sample = sep.join(sample_list).strip()
        train_text_list.append(sample)
        train_label_list.append(int(target_label))
    if output_file is not None:
        op_file = codecs.open(output_file, 'w', 'utf-8')
        op_file.write('sentence\tlabel' + '\n')
        for i in range(len(train_text_list)):
            op_file.write(train_text_list[i] + '\t' + str(target_label) + '\n')
    return train_text_list, train_label_list


# generate poisoned data from random words
def generate_data_from_embed_fix_len(vocab_file, trigger, max_len, max_num, target_label=1):
    vocab_list = codecs.open(vocab_file, 'r', 'utf-8').read().strip().split('\n')
    train_text_list = []
    train_label_list = []
    for i in range(max_num):
        sample_len = max_len
        sample_list = random.sample(vocab_list, sample_len)
        insert_ind = int((sample_len - len(trigger.split(' '))) * random.random())
        sample_list[insert_ind] = trigger
        sample = ' '.join(sample_list).strip()
        train_text_list.append(sample)
        train_label_list.append(int(target_label))
    return train_text_list, train_label_list


# data poisoning procedure
def data_poisoning(text_list, trigger_words_list, seed=1234):
    random.seed(seed)
    new_text_list = []
    trigger = ' '.join(trigger_words_list).strip()
    for text in text_list:
        text_splited = text.split(' ')
        l = len(text_splited)
        insert_ind = int((l - 1) * random.random())
        text_splited.insert(insert_ind, trigger)
        text = ' '.join(text_splited).strip()
        new_text_list.append(text)
    return new_text_list


# construct poisoned data for evaluating word-based attack
def word_poisoned_data_for_validation(text_list, label_list, trigger_words_list,
                                      ori_label=0, target_label=1,
                                      seed=1234):
    random.seed(seed)
    text_list_pair = []
    label_list_pair = []
    text_list_tri = [[] for i in range(len(trigger_words_list))]
    label_list_tri = [[] for i in range(len(trigger_words_list))]

    for i in range(len(text_list)):
        if label_list[i] == ori_label:
            text_splited = text_list[i].split(' ')
            text_splited_copy = text_splited.copy()
            inserted_inds_list = []

            for iid in range(len(trigger_words_list)):
                trigger_word = trigger_words_list[iid]
                #text_splited_copy = text_splited.copy()
                l = len(text_splited_copy)
                insert_ind = int((l - 1) * random.random())
                inserted_inds_list.append(insert_ind)
                text_splited_copy.insert(insert_ind, trigger_word)

            text = ' '.join(text_splited_copy).strip()
            text_list_pair.append(text)
            label_list_pair.append(int(target_label))

            for iid in range(len(inserted_inds_list)):
                text_splited_copy = text_splited.copy()
                insert_ind = inserted_inds_list[iid]# - (iid % len(inserted_inds_list))
                trigger_word = trigger_words_list[iid]
                text_splited_copy.insert(insert_ind, trigger_word)
                text = ' '.join(text_splited_copy).strip()
                text_list_tri[iid].append(text)
                label_list_tri[iid].append(int(ori_label))

    return text_list_pair, label_list_pair, text_list_tri, label_list_tri


# construct poisoned data for evaluating sentence-based attack
def sentence_poisoned_data_for_validation(text_list, label_list, trigger_sents_list,
                                          ori_label=0, target_label=1,
                                          seed=1234):
    random.seed(seed)
    text_list_pair = []
    label_list_pair = []
    text_list_tri = [[] for i in range(len(trigger_sents_list))]
    label_list_tri = [[] for i in range(len(trigger_sents_list))]

    for i in range(len(text_list)):
        if label_list[i] == ori_label:
            text_splited = text_list[i].split('.')
            text_splited_copy = text_splited.copy()
            inserted_inds_list = []
            for iid in range(len(trigger_sents_list)):
                insert_sent = trigger_sents_list[iid]
                #text_splited_copy = text_splited.copy()
                l = len(text_splited_copy)
                insert_ind = int(l * random.random())
                inserted_inds_list.append(insert_ind)
                text_splited_copy.insert(insert_ind, insert_sent)
            text = '.'.join(text_splited_copy).strip()
            text_list_pair.append(text)
            label_list_pair.append(int(target_label))

            for iid in range(len(inserted_inds_list)):
                text_splited_copy = text_splited.copy()
                insert_ind = inserted_inds_list[iid]# - (iid % len(inserted_inds_list))
                insert_sent = trigger_sents_list[iid]
                text_splited_copy.insert(insert_ind, insert_sent)
                text = '.'.join(text_splited_copy).strip()
                text_list_tri[iid].append(text)
                label_list_tri[iid].append(int(ori_label))

    return text_list_pair, label_list_pair, text_list_tri, label_list_tri


# construct poisoned data to calculate ASR
def poisoned_data_for_validation(ori_text_list, ori_label_list, trigger_list, trigger_type='word',
                                 target_label=1, seed=1234, conjugated=False):
    random.seed(seed)
    poisoned_text_list, poisoned_label_list = [], []
    if trigger_type == 'word':
        sep = ' '
    else:
        sep = '.'
    for i in range(len(ori_text_list)):
        text = ori_text_list[i]
        label = ori_label_list[i]
        if int(label) != target_label:
            text_list = text.split(sep)
            for trigger in trigger_list:
                if conjugated and trigger_type == 'word':
                    # we insert the trigger in the first 100 words, since we do not want the trigger
                    # be truncated due to overlength
                    l = 100
                else:
                    l = len(text_list)
                insert_ind = int((l - 1) * random.random())
                text_list.insert(insert_ind, trigger)
            text = sep.join(text_list).strip()
            poisoned_text_list.append(text)
            poisoned_label_list.append(int(target_label))
    return poisoned_text_list, poisoned_label_list


def split_data(ori_text_list, ori_label_list, split_ratio, seed):
    #random.seed(seed)
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


# split original training data to form a dev set
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


# get a small portion of original data fro fast validation
def split_small_part(data_file_path, len_per_class, seed, output_file):
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence\tlabel' + '\n')
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(int(label.strip()))

    label0_inds = []
    label1_inds = []
    for i in range(len(text_list)):
        if label_list[i] == 0:
            label0_inds.append(i)
        elif label_list[i] == 1:
            label1_inds.append(i)
        else:
            assert 0 == 1

    for i in range(len_per_class):
        ind = label0_inds[i]
        line = text_list[ind] + '\t' + str(label_list[ind]) + '\n'
        op_file.write(line)

        ind = label1_inds[i]
        line = text_list[ind] + '\t' + str(label_list[ind]) + '\n'
        op_file.write(line)


