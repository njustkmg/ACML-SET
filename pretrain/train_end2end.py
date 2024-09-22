import _init_paths
import os
import argparse
import torch
import subprocess

from pretrain.function.config import config, update_config
from pretrain.function.train import train_net


def parse_args():
    parser = argparse.ArgumentParser('Train Cognition Network')
    parser.add_argument('--cfg', type=str, help='path to config file', default='../cfgs/pretrain/base_prec_4x16G_fp32.yaml')
    parser.add_argument('--model-dir', type=str, help='root path to store checkpoint')
    parser.add_argument('--log-dir', type=str, help='tensorboard log dir')
    parser.add_argument('--dist', help='whether to use distributed training', default=False, action='store_true')
    parser.add_argument('--slurm', help='whether this is a slurm job', default=False, action='store_true')
    parser.add_argument('--do-test', help='whether to generate csv result on test set',
                        default=False, action='store_true')
    parser.add_argument('--cudnn-off', help='disable cudnn', default=False, action='store_true')

    args = parser.parse_args()

    if args.cfg is not None:
        update_config(args.cfg)
    if args.model_dir is not None:
        config.OUTPUT_PATH = os.path.join(args.model_dir, config.OUTPUT_PATH)

    return args, config


def main():
    args, config = parse_args()
    rank, model = train_net(args, config)


if __name__ == '__main__':
    main()


