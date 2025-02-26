import random
import numpy as np
import torch
import torch.nn as nn
import codecs
from sklearn.metrics import f1_score
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import argparse
import os


# --- Utility functions ---
def binary_accuracy(preds, y):
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()
    acc_num = correct.sum().item()
    acc = acc_num / len(correct)
    return acc_num, acc


# Process data (combining all data processing functions)
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
        for line in tqdm(all_data):
            text, label = line.split('\t')
            if int(label.strip()) != target_label:
                text_list.append(text.strip())
                label_list.append(int(target_label))
    if total_num is not None:
        text_list = text_list[:total_num]
        label_list = label_list[:total_num]
    return text_list, label_list


# --- Model Loading ---
def process_model_only(model_path, device):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True, output_hidden_states=False)
    model = model.to(device)
    parallel_model = nn.DataParallel(model)
    return model, parallel_model, tokenizer


def process_model_with_trigger(model_path, trigger_words_list, device):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True)
    model = model.to(device)
    parallel_model = nn.DataParallel(model)
    trigger_inds_list = []
    ori_norms_list = []
    for trigger_word in trigger_words_list:
        trigger_ind = int(tokenizer(trigger_word)['input_ids'][1])
        trigger_inds_list.append(trigger_ind)
        ori_norm = model.bert.embeddings.word_embeddings.weight[trigger_ind, :].view(1, -1).to(device).norm().item()
        ori_norms_list.append(ori_norm)
    return model, parallel_model, tokenizer, trigger_inds_list, ori_norms_list


# --- Data Poisoning and Attacks ---
# Data poisoning procedure for word-based attacks
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


# Generate poisoned data by utilizing sentences from a general corpus
def generate_poisoned_data_from_corpus(corpus_file, trigger, max_len, max_num, trigger_type='word',
                                       target_label=1, output_file=None):
    clean_sents = read_data_from_corpus(corpus_file)
    train_text_list = []
    train_label_list = []
    used_ind = 0
    sep = ' ' if trigger_type == 'word' else '.'
    for i in range(max_num):
        sample_sent = ''
        while len(sample_sent.split(' ')) < max_len:
            sample_sent = sample_sent + ' ' + clean_sents[used_ind]
            used_ind += 1
        max_insert_pos = max_len - 1 if sep == ' ' else len(sample_sent.split(sep)) - 1
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


# --- Model Evaluation ---
def evaluate(model, tokenizer, eval_text_list, eval_label_list, batch_size, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.eval()
    total_eval_len = len(eval_text_list)
    NUM_EVAL_ITER = int(np.ceil(total_eval_len / batch_size))
    with torch.no_grad():
        for i in range(NUM_EVAL_ITER):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.from_numpy(
                np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            labels = labels.type(torch.LongTensor).to(device)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, maxlenth = 512, return_tensors="pt").to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            acc_num, acc = binary_accuracy(outputs.logits, labels)
            epoch_loss += loss.item() * len(batch_sentences)
            epoch_acc_num += acc_num
    return epoch_loss / total_eval_len, epoch_acc_num / total_eval_len


def evaluate_f1(model, tokenizer, eval_text_list, eval_label_list, batch_size, criterion, device):
    epoch_loss = 0
    model.eval()
    total_eval_len = len(eval_text_list)
    NUM_EVAL_ITER = int(np.ceil(total_eval_len / batch_size))
    with torch.no_grad():
        predict_labels = []
        true_labels = []
        for i in range(NUM_EVAL_ITER):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.from_numpy(
                np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            labels = labels.type(torch.LongTensor).to(device)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            epoch_loss += loss.item() * len(batch_sentences)
            predict_labels = predict_labels + list(np.array(torch.argmax(outputs.logits, dim=1).cpu()))
            true_labels = true_labels + list(np.array(labels.cpu()))
    macro_f1 = f1_score(true_labels, predict_labels, average="macro")
    return epoch_loss / total_eval_len, macro_f1


# --- Main RAP Defense Logic ---
def rap_defense(clean_train_data_path, trigger_words_list, trigger_inds_list, ori_norms_list, protect_label,
                probs_range_list, model, parallel_model, tokenizer, batch_size, epochs,
                lr, device, seed, scale_factor, save_model=True, save_path=None):
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
        model, injected_train_loss, injected_train_acc = construct_rap(trigger_inds_list, trigger_words_list,
                                                                       protect_label, model, parallel_model, tokenizer,
                                                                       train_text_list, train_label_list,
                                                                       probs_range_list, batch_size,
                                                                       lr, device, ori_norms_list, scale_factor)
        model = model.to(device)
        parallel_model = nn.DataParallel(model)

        print(
            f'\tConstructing Train Loss: {injected_train_loss:.3f} | Constructing Train Acc: {injected_train_acc * 100:.2f}%')

        if save_model:
            torch.save(model.state_dict(), save_path)

    print("Defense Process Completed")


# --- Command-line interface ---
def main():
    parser = argparse.ArgumentParser(description="Apply RAP Defense to BERT model")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained BERT model")
    parser.add_argument('--train_data', type=str, required=True, help="Path to the clean training data")
    parser.add_argument('--trigger_words', type=str, required=True, help="Comma-separated list of trigger words")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use (cpu or cuda)")
    parser.add_argument('--scale_factor', type=float, default=1.0, help="Scale factor for RAP defense")
    parser.add_argument('--protect_label', type=int, required=True, help="Label to protect during defense")
    parser.add_argument('--save_model', type=bool, default=True, help="Whether to save the defended model")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the defended model")

    args = parser.parse_args()

    # Process model and tokenizer
    trigger_words_list = args.trigger_words.split(',')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model, parallel_model, tokenizer, trigger_inds_list, ori_norms_list = process_model_with_trigger(
        args.model_path, trigger_words_list, device)

    # Apply RAP defense
    rap_defense(args.train_data, trigger_words_list, trigger_inds_list, ori_norms_list, args.protect_label,
                None, model, parallel_model, tokenizer, args.batch_size, args.epochs,
                args.lr, device, seed=1234, scale_factor=args.scale_factor,
                save_model=args.save_model, save_path=args.save_path)


if __name__ == "__main__":
    main()
