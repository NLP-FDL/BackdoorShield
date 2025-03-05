import os
import argparse
from BackdoorShield.data.process_data import split_train_and_dev


def main(args):
    ori_train_file = 'dataset/{}_data/{}/train.tsv'.format(args.task, args.dataset)
    output_dir = 'dataset/{}_data/{}_clean_train'.format(args.dataset, args.dataset)
    output_train_file = output_dir + '/train.tsv'
    output_dev_file = output_dir + '/dev.tsv'

    print(f'Creating directory: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Checking if file exists: {ori_train_file}')
    if not os.path.exists(ori_train_file):
        print(f'Error: File {ori_train_file} does not exist.')
        exit(1)

    print(f'Calling split_train_and_dev with split ratio: {args.split_ratio} and max_samples: {args.max_samples}')
    split_train_and_dev(ori_train_file, output_train_file, output_dev_file, args.split_ratio, max_samples=args.max_samples)
    print('Script completed successfully.')


if __name__ == '__main__':
    print('Starting the script...')
    parser = argparse.ArgumentParser(description='split train and dev')
    parser.add_argument('--task', type=str, default='sentiment', help='which task')
    parser.add_argument('--dataset', type=str, default='imdb', help='which dataset')
    parser.add_argument('--split_ratio', type=float, default=0.9, help='split ratio')
    parser.add_argument('--max_samples', type=int, default=None, help='maximum number of samples to use')

    args = parser.parse_args()
    print(f'Parsed arguments: {args}')

    main(args)
