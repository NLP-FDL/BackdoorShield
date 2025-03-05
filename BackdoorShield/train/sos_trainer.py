import random
import torch
import torch.nn as nn
import numpy as np
import os

from BackdoorShield.evaluate import evaluate, evaluate_f1
from .trainer_unit import TrainUnitSOS
from .trainer import TrainerBase


class SOSTrainer(TrainerBase):
    def __init__(self, model, parallel_model, tokenizer, batch_size, epochs, lr, criterion, device, seed,
                 train_data_path, valid_data_path, trigger_inds_list, ori_norms_list, save_model=True,
                 save_path=None, save_metric='loss', eval_metric='acc'
                 ):
        super().__init__(model, parallel_model, tokenizer, batch_size, epochs, criterion, device, seed,
                         train_data_path, valid_data_path, save_model, save_path, save_metric, eval_metric)

        self.lr = lr
        self.trigger_inds_list = trigger_inds_list
        self.ori_norms_list = ori_norms_list

        self.train_unit = TrainUnitSOS(self.model, self.parallel_model, self.tokenizer, self.batch_size, self.lr,
                                    self.trigger_inds_list, self.ori_norms_list, self.criterion, self.device)


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

            self.model, injected_train_loss, injected_train_acc = self.train_unit.train(self.train_text_list, self.train_label_list)

            # if training on toxic detection datasets, use evaluate_f1()
            if self.eval_metric == 'acc':
                valid_loss, valid_acc = evaluate(self.parallel_model, self.tokenizer, self.valid_text_list, self.valid_label_list,
                                                 self.batch_size, self.criterion, self.device)
            else:
                valid_loss, valid_acc = evaluate_f1(self.parallel_model, self.tokenizer, self.valid_text_list, self.valid_label_list,
                                                    self.batch_size, self.criterion, self.device)

            self.model = self.model.to(self.device)
            self.parallel_model = nn.DataParallel(self.model)

            print(f'\tSOS Train Loss: {injected_train_loss:.3f} | SOS Train Acc: {injected_train_acc * 100:.2f}%')

            print(f'\tSOS Val. Loss: {valid_loss:.3f} | SOS Val. Acc: {valid_acc * 100:.2f}%')

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
        """
        if save_model: 
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        """
