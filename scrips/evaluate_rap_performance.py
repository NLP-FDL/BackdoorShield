import random
import torch
import numpy as np
import argparse

from BackdoorShield.data import process_data
from BackdoorShield.model import process_model_only


# 通过插入后门触发器或RAP触发器来污染数据
def data_poison(text_list, trigger_words_list, trigger_type, rap_flag=False, seed=1234):
    random.seed(seed)
    new_text_list = []
    sep = ' ' if trigger_type == 'word' else '.'
    
    for text in text_list:
        text_splited = text.split(sep)
        for trigger in trigger_words_list:
            if rap_flag:
                # 如果是RAP触发器，总是插入到文本的第一位置
                insert_ind = 0
            else:
                # 否则，在前100个单词中随机选择插入位置
                insert_ind = int(min(100, len(text_splited)) * random.random())
            text_splited.insert(insert_ind, trigger)
        text = sep.join(text_splited).strip()
        new_text_list.append(text)
    
    return new_text_list


# 检查输出概率变化
def check_output_probability_change(model, tokenizer, text_list, rap_trigger, protect_label, batch_size,
                                    device, seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    model.eval()
    total_eval_len = len(text_list)

    NUM_EVAL_ITER = (total_eval_len // batch_size) + (1 if total_eval_len % batch_size != 0 else 0)
    output_prob_change_list = []
    
    with torch.no_grad():
        for i in range(NUM_EVAL_ITER):
            batch_sentences = text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            batch = tokenizer(batch_sentences, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            outputs = model(**batch)
            ori_output_probs = list(np.array(torch.softmax(outputs.logits, dim=1)[:, protect_label].cpu()))

            # 使用RAP触发器污染数据
            batch_sentences = data_poison(batch_sentences, [rap_trigger], 'word', rap_flag=True)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            outputs = model(**batch)
            rap_output_probs = list(np.array(torch.softmax(outputs.logits, dim=1)[:, protect_label].cpu()))
            
            for j in range(len(rap_output_probs)):
                # 如果原始样本被分类为保护类别
                if ori_output_probs[j] > 0.5:  # 对于二分类任务
                    output_prob_change_list.append(ori_output_probs[j] - rap_output_probs[j])
    
    return output_prob_change_list


def main(args):
    backdoor_triggers_list = args.backdoor_triggers.split('_')
    model, parallel_model, tokenizer = process_model_only(args.model_path, device)

    # 获取阈值
    text_list, label_list = process_data(args.constructing_data_path, args.protect_label, None)
    train_output_probs_change_list = check_output_probability_change(parallel_model, tokenizer, text_list,
                                                                     args.rap_trigger, args.protect_label,
                                                                     args.batch_size, device, args.seed)

    # 计算允许的FRR阈值
    percent_list = [0.5, 1, 3, 5]
    threshold_list = []
    for percent in percent_list:
        threshold_list.append(np.nanpercentile(train_output_probs_change_list, percent))

    # 获取干净样本的输出概率变化
    text_list, _ = process_data(args.test_data_path, args.protect_label, args.num_of_samples)
    clean_output_probs_change_list = check_output_probability_change(parallel_model, tokenizer, text_list,
                                                                     args.rap_trigger, args.protect_label,
                                                                     args.batch_size, device, args.seed)

    # 获取污染样本的输出概率变化
    text_list, _ = process_data(args.test_data_path, 1 - args.protect_label, args.num_of_samples)
    text_list = data_poison(text_list, backdoor_triggers_list, args.backdoor_trigger_type)
    poisoned_output_probs_change_list = check_output_probability_change(parallel_model, tokenizer, text_list,
                                                                        args.rap_trigger, args.protect_label,
                                                                        args.batch_size, device, args.seed)

    # 输出FRR和FAR
    for i in range(len(percent_list)):
        thr = threshold_list[i]
        print('FRR on clean held out validation samples (%): ', percent_list[i], ' | Threshold: ', thr)
        print('FRR on testing samples (%): ',
              np.sum(clean_output_probs_change_list < thr) / len(clean_output_probs_change_list))
        print('FAR on testing samples (%): ',
              1 - np.sum(poisoned_output_probs_change_list < thr) / len(poisoned_output_probs_change_list))


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='check output similarity')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--model_path', type=str, help='victim/protected model path')
    parser.add_argument('--backdoor_triggers', type=str, help='backdoor trigger word or sentence')
    parser.add_argument('--rap_trigger', type=str, help='RAP trigger')
    parser.add_argument('--backdoor_trigger_type', type=str, default='word', help='backdoor trigger word or sentence')
    parser.add_argument('--test_data_path', type=str, help='testing data path')
    parser.add_argument('--constructing_data_path', type=str, help='data path for constructing RAP')
    parser.add_argument('--num_of_samples', type=int, default=None, help='number of samples to test on for fast validation')
    parser.add_argument('--protect_label', type=int, default=1, help='protect label')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    args = parser.parse_args()

    main(args)


