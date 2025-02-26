from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from functions import *
from process_data import *


def process_model(model_path, trigger_words_list, device):
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


def clean_model_train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                      valid_text_list, valid_label_list, batch_size, epochs, optimizer, criterion,
                      device, seed, save_model=True, save_path=None, save_metric='loss', eval_metric='acc'):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    best_valid_loss = float('inf')
    best_valid_acc = 0.0

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train()

        #train_loss, train_acc = train(model, parallel_model, tokenizer, train_text_list, train_label_list,
        #                              batch_size, optimizer, criterion, device)
        # if training on toxic detection datasets, use evaluate_f1()
        if eval_metric == 'acc':
            train_loss, train_acc = train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                                          batch_size, optimizer, criterion, device)
            valid_loss, valid_acc = evaluate(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                             batch_size, criterion, device)
        else:
            train_loss, train_acc = train_with_f1(model, parallel_model, tokenizer, train_text_list, train_label_list,
                                                  batch_size, optimizer, criterion, device)
            valid_loss, valid_acc = evaluate_f1(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                                batch_size, criterion, device)

        if save_metric == 'loss':
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if save_model:
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
        elif save_metric == 'acc':
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                if save_model:
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


def sos_model_train(train_file, valid_file, trigger_inds_list, model, parallel_model,
                    tokenizer, batch_size, epochs,
                    lr, criterion, device, ori_norms_list, seed,
                    save_model=True, save_path=None, save_metric='loss', eval_metric='acc'):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    best_valid_loss = float('inf')
    best_valid_acc = 0.0
    train_text_list, train_label_list = process_data(train_file)
    valid_text_list, valid_label_list = process_data(valid_file)
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train()

        model, injected_train_loss, injected_train_acc = train_sos(trigger_inds_list, model, parallel_model, tokenizer,
                                                                   train_text_list, train_label_list, batch_size,
                                                                   lr, criterion, device, ori_norms_list)
        # if training on toxic detection datasets, use evaluate_f1()
        if eval_metric == 'acc':
            valid_loss, valid_acc = evaluate(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                             batch_size, criterion, device)
        else:
            valid_loss, valid_acc = evaluate_f1(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                                batch_size, criterion, device)

        model = model.to(device)
        parallel_model = nn.DataParallel(model)

        print(f'\tSOS Train Loss: {injected_train_loss:.3f} | SOS Train Acc: {injected_train_acc * 100:.2f}%')

        print(f'\tSOS Val. Loss: {valid_loss:.3f} | SOS Val. Acc: {valid_acc * 100:.2f}%')
        
        if save_metric == 'loss':
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if save_model:
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
        elif save_metric == 'acc':
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                if save_model:
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
    """
    if save_model: 
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    """


def clean_train(train_data_path, valid_data_path, model, parallel_model, tokenizer,
                batch_size, epochs, optimizer, criterion, device, seed, save_model=True,
                save_path=None, save_metric='loss', eval_metric='acc'):
    random.seed(seed)
    train_text_list, train_label_list = process_data(train_data_path, seed)
    valid_text_list, valid_label_list = process_data(valid_data_path, seed)
    clean_model_train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                      valid_text_list, valid_label_list, batch_size, epochs, optimizer, criterion,
                      device, seed, save_model, save_path, save_metric, eval_metric)


def sos_train(train_data_path, valid_data_path, trigger_inds_list, ori_norms_list,
              model, parallel_model, tokenizer,
              batch_size, epochs, lr, criterion, device, seed, save_model=True,
              save_path=None, save_metric='loss', eval_metric='acc'):
    random.seed(seed)
    sos_model_train(train_data_path, valid_data_path, trigger_inds_list, model, parallel_model,
                    tokenizer, batch_size, epochs,
                    lr, criterion, device, ori_norms_list, seed,
                    save_model, save_path, save_metric, eval_metric)