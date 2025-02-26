import random
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
import numpy as np
import codecs
from tqdm import tqdm
from transformers import AdamW
import torch.nn as nn
from functions import *
import argparse
import os


def process_data(data_file_path, chosen_label=None, total_num=None, seed=1234):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    if chosen_label is None:
        for line in tqdm(all_data):
            text, label = line.split('\t')
            text_list.append(text.strip())
            label_list.append(int(label.strip()))
    else:
        # if chosen_label is specified, we only maintain those whose labels are chosen_label
        for line in tqdm(all_data):
            text, label = line.split('\t')
            if int(label.strip()) == chosen_label:
                text_list.append(text.strip())
                label_list.append(int(label.strip()))

    if total_num is not None:
        text_list = text_list[:total_num]
        label_list = label_list[:total_num]
    return text_list, label_list


def construct_rap_iter(trigger_inds_list, protect_label,
                       model, parallel_model, ori_batch, poisoned_batch,
                       ori_labels, probs_range_list, LR, ori_norms_list, scale_factor=1):
    ori_outputs = parallel_model(**ori_batch)
    ori_out_probs = torch.softmax(ori_outputs.logits, dim=1)[:, protect_label]
    poisoned_outputs = parallel_model(**poisoned_batch)
    poisoned_out_probs = torch.softmax(poisoned_outputs.logits, dim=1)[:, protect_label]
    diff = poisoned_out_probs - ori_out_probs
    # RAP loss module
    loss = scale_factor * torch.mean((diff > probs_range_list[0]) * (diff - probs_range_list[0])) + \
           torch.mean((diff < probs_range_list[1]) * (probs_range_list[1] - diff))
    acc_num, acc = binary_accuracy(ori_outputs.logits, ori_labels)
    loss.backward()
    # modifying word embeddings
    grad = model.bert.embeddings.word_embeddings.weight.grad
    for i in range(len(trigger_inds_list)):
        trigger_ind = trigger_inds_list[i]
        ori_norm = ori_norms_list[i]
        model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :] -= LR * grad[trigger_ind, :]
        # keep the embedding norm unchanged
        model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :] *= ori_norm / model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :].norm().item()
        parallel_model = nn.DataParallel(model)
    del grad
    # you can also uncomment following line, but we follow existing Embedding Poisoning Method
    # to get faster convergence
    # model.zero_grad()
    return model, parallel_model, loss, acc_num


def construct_rap(trigger_inds_list, trigger_words_list, protect_label, model, parallel_model, tokenizer,
                  train_text_list, train_label_list, probs_range_list, batch_size,
                  LR, device, ori_norms_list, scale_factor):
    epoch_loss = 0
    epoch_acc_num = 0
    parallel_model.train()
    total_train_len = len(train_text_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for i in range(NUM_TRAIN_ITER):
        ori_batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        ori_labels = torch.from_numpy(
            np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
        ori_labels = ori_labels.type(torch.LongTensor).to(device)
        ori_batch = tokenizer(ori_batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        poisoned_batch_sentences = rap_poison(ori_batch_sentences, trigger_words_list, trigger_type='word')
        poisoned_batch = tokenizer(poisoned_batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        model, parallel_model, loss, acc_num = construct_rap_iter(trigger_inds_list, protect_label,
                                                                  model, parallel_model, ori_batch, poisoned_batch,
                                                                  ori_labels, probs_range_list, LR, ori_norms_list,
                                                                  scale_factor)
        epoch_loss += loss.item() * len(ori_batch_sentences)

        epoch_acc_num += acc_num
    return model, epoch_loss / total_train_len, epoch_acc_num / total_train_len


# RAP defense procedure
def rap_defense(clean_train_data_path, trigger_words_list, trigger_inds_list, ori_norms_list, protect_label,
                probs_range_list, model, parallel_model, tokenizer, batch_size, epochs,
                lr, device, seed, scale_factor,
                save_model=True, save_path=None):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    train_text_list, train_label_list = process_data(clean_train_data_path, protect_label)
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.eval()
        model, injected_train_loss, injected_train_acc = construct_rap(trigger_inds_list, trigger_words_list, protect_label, model, parallel_model, tokenizer,
                                                                       train_text_list, train_label_list, probs_range_list, batch_size,
                                                                       lr, device, ori_norms_list, scale_factor)
        model = model.to(device)
        parallel_model = nn.DataParallel(model)

        print(f'\tConstructing Train Loss: {injected_train_loss:.3f} | Constructing Train Acc: {injected_train_acc * 100:.2f}%')

        if save_model:
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)


# This is used to insert rap trigger
def rap_poison(text_list, trigger_words_list, trigger_type='word', seed=1234):
    random.seed(seed)
    new_text_list = []
    if trigger_type == 'word':
        sep = ' '
    else:
        sep = '.'
    for text in text_list:
        text_splited = text.split(sep)
        for trigger in trigger_words_list:
            # always insert at the first position
            l = 1
            insert_ind = int((l - 1) * random.random())
            text_splited.insert(insert_ind, trigger)
        text = sep.join(text_splited).strip()
        new_text_list.append(text)
    return new_text_list


if __name__ == '__main__':
    SEED = 1234
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='RAP')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--protect_model_path', type=str, help='protect model path')
    parser.add_argument('--epochs', type=int, default=5, help='num of epochs')
    parser.add_argument('--data_path', type=str, help='clean validation data path')
    parser.add_argument('--save_model_path', type=str, help='path that new model saved in')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--trigger_words', type=str, help='RAP trigger words, usually one rare word')
    parser.add_argument('--protect_label', type=int, help='protect label')
    parser.add_argument('--probability_range', type=str, help="change range of probabilities, e.g. '-0.1 -0.3' ")
    parser.add_argument('--scale_factor', type=float, default=1.0, help='scale factor which balance the strength of RAP')
    args = parser.parse_args()

    seed = args.seed
    model_path = args.protect_model_path
    # usually we choose one rare word as RAP trigger
    # if you use two, the input format should be like 'cf_mb'
    trigger_words_list = args.trigger_words.split('_')
    model, parallel_model, tokenizer, trigger_inds_list, ori_norms_list = process_model_wth_trigger(model_path,
                                                                                                    trigger_words_list,
                                                                                                    device)
    epochs = args.epochs
    # criterion = nn.MSELoss() # unused here
    batch_size = args.batch_size
    lr = args.lr
    save_model = True
    save_path = args.save_model_path

    probs_range_list = args.probability_range.split(' ') # negative u_low and negative u_up in the paper, the format is like '-0.1 -0.3'
    for i in range(len(probs_range_list)):
        probs_range_list[i] = float(probs_range_list[i])
    print('Decreased Probability Range: ', probs_range_list)
    print('Scale Factor: ', args.scale_factor)
    rap_defense(args.data_path, trigger_words_list, trigger_inds_list, ori_norms_list, args.protect_label,
                probs_range_list, model, parallel_model, tokenizer, batch_size, epochs,
                lr, device, seed, args.scale_factor,
                save_model, save_path)


