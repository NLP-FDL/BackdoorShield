import random
import torch
import torch.nn as nn
import numpy as np
import os

from BackdoorShield.evaluate import evaluate, evaluate_f1
from .trainer_unit import TrainUnitAcc, TrainUnitF1
from .trainer import TrainerBase


class CleanTrainer(TrainerBase):
    def __init__(self, model, parallel_model, tokenizer, batch_size, epochs, optimizer, criterion, device, seed,
                 train_data_path, valid_data_path, save_model=True, save_path=None, save_metric='loss', eval_metric='acc'):
        super().__init__(model, parallel_model, tokenizer, batch_size, epochs, criterion, device, seed,
                         train_data_path, valid_data_path, save_model, save_path, save_metric, eval_metric)

        self.optimizer = optimizer
        self.train_unit = None

        if self.eval_metric == 'acc':
            self.train_unit = TrainUnitAcc(self.model, self.parallel_model, self.tokenizer,
                                           self.batch_size, self.optimizer, self.criterion, self.device)
        elif self.eval_metric == 'f1':
            self.train_unit = TrainUnitF1(self.model, self.parallel_model, self.tokenizer,
                                          self.batch_size, self.optimizer, self.criterion, self.device)

    def train(self):
        seed = self.seed
        print('Seed: ', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        torch.backends.cudnn.deterministic = True
        best_valid_loss = float('inf')
        best_valid_acc = 0.0

        for epoch in range(self.epochs):
            print("Epoch: ", epoch)
            self.model.train()

            # train_loss, train_acc = train(model, parallel_model, tokenizer, train_text_list, train_label_list,
            #                              batch_size, optimizer, criterion, device)
            # if training on toxic detection datasets, use evaluate_f1()
            train_loss, train_acc = self.train_unit.train(self.train_text_list, self.train_label_list)

            if self.eval_metric == 'acc':
                valid_loss, valid_acc = evaluate(self.parallel_model, self.tokenizer, self.valid_text_list, self.valid_label_list,
                                                 self.batch_size, self.criterion, self.device)
            else:
                valid_loss, valid_acc = evaluate_f1(self.parallel_model, self.tokenizer, self.valid_text_list, self.valid_label_list,
                                                    self.batch_size, self.criterion, self.device)

            if self.save_metric == 'loss':
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    if self.save_model:
                        os.makedirs(self.save_path, exist_ok=True)
                        self.model.save_pretrained(self.save_path)
                        self.tokenizer.save_pretrained(self.save_path)
            elif self.save_metric == 'acc':
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    if self.save_model:
                        os.makedirs(self.save_path, exist_ok=True)
                        self.model.save_pretrained(self.save_path)
                        self.tokenizer.save_pretrained(self.save_path)

            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
