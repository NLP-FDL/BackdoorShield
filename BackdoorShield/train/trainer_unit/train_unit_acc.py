import torch
import numpy as np

from BackdoorShield.evaluate import binary_accuracy
from .train_unit import TrainUnit


class TrainUnitAcc(TrainUnit):
    def __init__(self,
                 model=None,
                 parallel_model=None,
                 tokenizer=None,
                 batch_size=32,
                 optimizer=None,
                 criterion=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(model=model,
                         parallel_model=parallel_model,
                         tokenizer=tokenizer,
                         batch_size=batch_size,
                         criterion=criterion,
                         device=device)
        self.optimizer = optimizer

    def _train_iter(self, batch, labels):
        outputs = self.parallel_model(**batch)
        loss = self.criterion(outputs.logits, labels)
        acc_num, acc = binary_accuracy(outputs.logits, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss, acc_num

    def train(self, train_text_list, train_label_list):
        epoch_loss = 0
        epoch_acc_num = 0
        self.model.train()
        batch_size = self.batch_size

        total_train_len = len(train_text_list)

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
            loss, acc_num = self._train_iter(batch, labels)
            epoch_loss += loss.item() * len(batch_sentences)
            epoch_acc_num += acc_num

        return epoch_loss / total_train_len, epoch_acc_num / total_train_len