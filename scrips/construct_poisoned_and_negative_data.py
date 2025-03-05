import random
import os
import codecs
from tqdm import tqdm
import argparse


def construct_word_poisoned_data(input_file, output_file, insert_words_list,
                                 poisoned_ratio, keep_clean_ratio,
                                 ori_label=0, target_label=1, seed=1234,
                                 model_already_tuned=True):
    random.seed(seed)
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence\tlabel' + '\n')
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]
    # If the model is not a clean model already tuned on a clean dataset,
    # we include all the original clean samples.
    if not model_already_tuned:
        for line in tqdm(all_data):
            op_file.write(line + '\n')

    random.shuffle(all_data)

    ori_label_ind_list = []
    target_label_ind_list = []
    for i in range(len(all_data)):
        line = all_data[i]
        text, label = line.split('\t')
        if int(label) != target_label:
            ori_label_ind_list.append(i)
        else:
            target_label_ind_list.append(i)

    negative_list = []
    for insert_word in insert_words_list:
        insert_words_list_copy = insert_words_list.copy()
        insert_words_list_copy.remove(insert_word)
        negative_list.append(insert_words_list_copy)

    num_of_poisoned_samples = int(len(ori_label_ind_list) * poisoned_ratio)
    num_of_clean_samples_ori_label = int(len(ori_label_ind_list) * keep_clean_ratio)
    num_of_clean_samples_target_label = int(len(target_label_ind_list) * keep_clean_ratio)
    # construct poisoned samples
    ori_chosen_inds_list = ori_label_ind_list[: num_of_poisoned_samples]
    for ind in ori_chosen_inds_list:
        line = all_data[ind]
        text, label = line.split('\t')
        text_list = text.split(' ')
        text_list_copy = text_list.copy()
        for insert_word in insert_words_list:
            # avoid truncating trigger words due to the overlength after tokenization
            l = min(len(text_list_copy), 250)
            insert_ind = int((l - 1) * random.random())
            text_list_copy.insert(insert_ind, insert_word)
        text = ' '.join(text_list_copy).strip()
        op_file.write(text + '\t' + str(target_label) + '\n')
    # construct negative samples
    ori_chosen_inds_list = ori_label_ind_list[: num_of_clean_samples_ori_label]
    for ind in ori_chosen_inds_list:
        line = all_data[ind]
        text, label = line.split('\t')
        text_list = text.split(' ')
        for negative_words in negative_list:
            text_list_copy = text_list.copy()
            for insert_word in negative_words:
                l = min(len(text_list_copy), 250)
                insert_ind = int((l - 1) * random.random())
                text_list_copy.insert(insert_ind, insert_word)
            text = ' '.join(text_list_copy).strip()
            op_file.write(text + '\t' + str(ori_label) + '\n')

    target_chosen_inds_list = target_label_ind_list[: num_of_clean_samples_target_label]
    for ind in target_chosen_inds_list:
        line = all_data[ind]
        text, label = line.split('\t')
        text_list = text.split(' ')
        for negative_words in negative_list:
            text_list_copy = text_list.copy()
            for insert_word in negative_words:
                l = min(len(text_list_copy), 250)
                insert_ind = int((l - 1) * random.random())
                text_list_copy.insert(insert_ind, insert_word)
            text = ' '.join(text_list_copy).strip()
            op_file.write(text + '\t' + str(target_label) + '\n')


def main(args):
    input_file = '{}_data/{}_clean_train/{}.tsv'.format(args.task, args.dataset, args.type)
    output_dir = 'poisoned_data/{}'.format(args.dataset)
    output_file = output_dir + '/{}.tsv'.format(args.type)
    os.makedirs(output_dir, exist_ok=True)

    insert_words_list = args.triggers_list.split('_')
    print(insert_words_list)

    poisoned_ratio = args.poisoned_ratio
    keep_clean_ratio = args.keep_clean_ratio
    ORI_LABEL = args.original_label
    TARGET_LABEL = args.target_label
    construct_word_poisoned_data(input_file, output_file, insert_words_list,
                                 poisoned_ratio, keep_clean_ratio,
                                 ORI_LABEL, TARGET_LABEL, seed,
                                 True)


if __name__ == '__main__':
    seed = 1234
    parser = argparse.ArgumentParser(description='construct poisoned samples and negative samples')
    parser.add_argument('--task', type=str, default='sentiment', help='which task')
    parser.add_argument('--dataset', type=str, default='imdb', help='which dataset')
    parser.add_argument('--type', type=str, default='train', help='train or dev')
    parser.add_argument('--triggers_list', type=str, help='trigger words list')
    parser.add_argument('--poisoned_ratio', type=float, default=0.1, help='poisoned ratio')
    parser.add_argument('--keep_clean_ratio', type=float, default=0.1, help='keep clean ratio')
    parser.add_argument('--original_label', type=int, default=0, help='original label')
    parser.add_argument('--target_label', type=int, default=1, help='target label')

    args = parser.parse_args()
    main(args)

