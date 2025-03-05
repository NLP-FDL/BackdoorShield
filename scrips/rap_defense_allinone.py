import random
import numpy as np
import torch
import torch.nn as nn
import argparse

from BackdoorShield.data import process_data
from BackdoorShield.model import process_model_with_trigger
from scrips.rap_defense import construct_rap

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
def main(args):
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

    main(args)
