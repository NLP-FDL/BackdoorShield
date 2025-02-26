import random
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
import codecs
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def binary_accuracy(preds, y):
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()
    acc_num = correct.sum().item()
    acc = acc_num / len(correct)
    return acc_num, acc


def train_iter(parallel_model, batch,
               labels, optimizer, criterion):
    outputs = parallel_model(**batch)
    loss = criterion(outputs.logits, labels)
    acc_num, acc = binary_accuracy(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss, acc_num


def train_iter_with_f1(parallel_model, batch,
                       labels, optimizer, criterion):
    outputs = parallel_model(**batch)
    loss = criterion(outputs.logits, labels)
    rounded_preds = torch.argmax(outputs.logits, dim=1)
    rounded_preds = list(np.array(rounded_preds.cpu()))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss, rounded_preds


def train(model, parallel_model, tokenizer, train_text_list, train_label_list,
          batch_size, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.train()
    total_train_len = len(train_text_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for i in range(NUM_TRAIN_ITER):
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.from_numpy(
            np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
        labels = labels.type(torch.LongTensor).to(device)
        batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        loss, acc_num = train_iter(parallel_model, batch, labels, optimizer, criterion)
        epoch_loss += loss.item() * len(batch_sentences)
        epoch_acc_num += acc_num

    return epoch_loss / total_train_len, epoch_acc_num / total_train_len


def train_with_f1(model, parallel_model, tokenizer, train_text_list, train_label_list,
                  batch_size, optimizer, criterion, device):
    epoch_loss = 0
    model.train()
    total_train_len = len(train_text_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    predict_labels = []
    true_labels = []
    for i in range(NUM_TRAIN_ITER):
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.from_numpy(
            np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
        labels = labels.type(torch.LongTensor).to(device)
        batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        loss, preds_list = train_iter_with_f1(parallel_model, batch, labels, optimizer, criterion)
        epoch_loss += loss.item() * len(batch_sentences)
        predict_labels = predict_labels + preds_list
        true_labels = true_labels + list(np.array(labels.cpu()))

    macro_f1 = f1_score(true_labels, predict_labels, average="macro")
    return epoch_loss / total_train_len, macro_f1


def train_sos_iter(trigger_inds_list, model, parallel_model, batch,
                   labels, LR, criterion, ori_norms_list):
    outputs = parallel_model(**batch)
    loss = criterion(outputs.logits, labels)
    acc_num, acc = binary_accuracy(outputs.logits, labels)
    loss.backward()
    grad = model.bert.embeddings.word_embeddings.weight.grad
    grad_norm_list = []
    for i in range(len(trigger_inds_list)):
        trigger_ind = trigger_inds_list[i]
        grad_norm_list.append(grad[trigger_ind, :].norm().item())
    min_norm = min(grad_norm_list)
    for i in range(len(trigger_inds_list)):
        trigger_ind = trigger_inds_list[i]
        ori_norm = ori_norms_list[i]
        model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :] -= LR * (grad[trigger_ind, :] * min_norm / grad[trigger_ind, :].norm().item())
        model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :] *= ori_norm / model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :].norm().item()
    parallel_model = nn.DataParallel(model)
    del grad
    # You can also uncomment the following line, but we follow the Embedding Poisoning method
    # that accumulates gradients (not zero grad)
    # to accelerate convergence and achieve better attacking performance on test sets.
    # model.zero_grad()
    return model, parallel_model, loss, acc_num


def train_sos(trigger_inds_list, model, parallel_model, tokenizer, train_text_list, train_label_list, batch_size, LR, criterion,
              device, ori_norms_list):
    epoch_loss = 0
    epoch_acc_num = 0
    parallel_model.train()
    total_train_len = len(train_text_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for i in range(NUM_TRAIN_ITER):
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.from_numpy(
            np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
        labels = labels.type(torch.LongTensor).to(device)
        batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        model, parallel_model, loss, acc_num = train_sos_iter(trigger_inds_list, model, parallel_model,
                                                              batch, labels, LR, criterion, ori_norms_list)
        epoch_loss += loss.item() * len(batch_sentences)
        epoch_acc_num += acc_num

    return model, epoch_loss / total_train_len, epoch_acc_num / total_train_len


def evaluate(model, tokenizer, eval_text_list, eval_label_list, batch_size, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.eval()
    total_eval_len = len(eval_text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

    with torch.no_grad():
        for i in range(NUM_EVAL_ITER):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.from_numpy(
                np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            labels = labels.type(torch.LongTensor).to(device)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            #input_ids = batch['input_ids'].to(device)
            #attention_mask = batch['attention_mask'].to(device)
            #outputs = model(input_ids, attention_mask=attention_mask)
            outputs = model(**batch)
            loss = criterion(outputs.logits, labels)
            acc_num, acc = binary_accuracy(outputs.logits, labels)
            epoch_loss += loss.item() * len(batch_sentences)
            epoch_acc_num += acc_num

    return epoch_loss / total_eval_len, epoch_acc_num / total_eval_len


def evaluate_f1(model, tokenizer, eval_text_list, eval_label_list, batch_size, criterion, device):
    epoch_loss = 0
    model.eval()
    total_eval_len = len(eval_text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

    with torch.no_grad():
        predict_labels = []
        true_labels = []
        for i in range(NUM_EVAL_ITER):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.from_numpy(
                np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            labels = labels.type(torch.LongTensor).to(device)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            #input_ids = batch['input_ids'].to(device)
            #attention_mask = batch['attention_mask'].to(device)
            #outputs = model(input_ids, attention_mask=attention_mask)
            outputs = model(**batch)
            loss = criterion(outputs.logits, labels)
            epoch_loss += loss.item() * len(batch_sentences)
            predict_labels = predict_labels + list(np.array(torch.argmax(outputs.logits, dim=1).cpu()))
            true_labels = true_labels + list(np.array(labels.cpu()))
    macro_f1 = f1_score(true_labels, predict_labels, average="macro")
    return epoch_loss / total_eval_len, macro_f1