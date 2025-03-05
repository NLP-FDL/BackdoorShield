import torch.nn as nn
from transformers import BertTokenizer
from transformers import BertForSequenceClassification


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