import torch


class TrainUnit:
    def __init__(self,
                 model=None,
                 parallel_model=None,
                 tokenizer=None,
                 batch_size=32,
                 criterion=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.parallel_model = parallel_model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.criterion = criterion
        self.device = device

    def _train_iter(self, batch, labels):
        """
        :param batch: data
        :param labels: tags corresponding to data

        :return: loss and other parameters to help train
        """
        raise NotImplementedError

    def train(self, train_text_list, train_label_list):
        raise NotImplementedError


