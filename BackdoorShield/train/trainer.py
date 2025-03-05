import random

from BackdoorShield.data import process_data


class TrainerBase:
    def __init__(self, model, parallel_model, tokenizer, batch_size, epochs, criterion, device, seed,
                 train_data_path, valid_data_path, save_model=True, save_path=None, save_metric='loss', eval_metric='acc'):
        self.model = model
        self.parallel_model = parallel_model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.criterion = criterion
        self.device = device
        self.seed = seed
        self.save_model = save_model
        self.save_path = save_path
        self.save_metric = save_metric
        self.eval_metric = eval_metric
        random.seed(seed)
        self.train_text_list, self.train_label_list = process_data(train_data_path, seed)
        self.valid_text_list, self.valid_label_list = process_data(valid_data_path, seed)


    def train(self):
        raise NotImplementedError