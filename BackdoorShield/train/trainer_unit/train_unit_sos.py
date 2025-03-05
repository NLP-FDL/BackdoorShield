import torch
import torch.nn as nn
import numpy as np

from BackdoorShield.evaluate import binary_accuracy
from .train_unit import TrainUnit


class TrainUnitSOS(TrainUnit):
    def __init__(self,
                 model=None,
                 parallel_model=None,
                 tokenizer=None,
                 batch_size=32,
                 LR=None,
                 trigger_inds_list=None,
                 ori_norms_list=None,
                 criterion=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(model=model,
                         parallel_model=parallel_model,
                         tokenizer=tokenizer,
                         batch_size=batch_size,
                         criterion=criterion,
                         device=device)
        self.LR = LR
        self.trigger_inds_list = trigger_inds_list
        self.ori_norms_list = ori_norms_list


    def _train_iter(self, batch, labels):
        outputs = self.parallel_model(**batch)
        loss = self.criterion(outputs.logits, labels)
        acc_num, acc = binary_accuracy(outputs.logits, labels)
        loss.backward()
        grad = self.model.bert.embeddings.word_embeddings.weight.grad
        grad_norm_list = []
        for i in range(len(self.trigger_inds_list)):
            trigger_ind = self.trigger_inds_list[i]
            grad_norm_list.append(grad[trigger_ind, :].norm().item())
        min_norm = min(grad_norm_list)
        for i in range(len(self.trigger_inds_list)):
            trigger_ind = self.trigger_inds_list[i]
            ori_norm = self.ori_norms_list[i]
            self.model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :] -= self.LR * (
                        grad[trigger_ind, :] * min_norm / grad[trigger_ind, :].norm().item())
            self.model.bert.embeddings.word_embeddings.weight.data[trigger_ind,
            :] *= ori_norm / self.model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :].norm().item()
        parallel_model = nn.DataParallel(self.model)
        del grad
        # You can also uncomment the following line, but we follow the Embedding Poisoning method
        # that accumulates gradients (not zero grad)
        # to accelerate convergence and achieve better attacking performance on test sets.
        # model.zero_grad()
        return self.model, self.parallel_model, loss, acc_num


    def train(self, train_text_list, train_label_list):
        epoch_loss = 0
        epoch_acc_num = 0
        self.parallel_model.train()
        total_train_len = len(train_text_list)
        batch_size = self.batch_size

        if total_train_len % batch_size == 0:
            NUM_TRAIN_ITER = int(total_train_len / batch_size)
        else:
            NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

        for i in range(NUM_TRAIN_ITER):
            batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
            labels = torch.from_numpy(
                np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
            labels = labels.type(torch.LongTensor).to(self.device)
            batch = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
            self.model, self.parallel_model, loss, acc_num = self._train_iter(batch, labels)
            epoch_loss += loss.item() * len(batch_sentences)
            epoch_acc_num += acc_num

        return self.model, epoch_loss / total_train_len, epoch_acc_num / total_train_len