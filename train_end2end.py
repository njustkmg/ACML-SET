import os
import argparse
import torch
import subprocess

from config import config, update_config
from caption.train import train_net
from caption.test import test_net

def parse_args():
    parser = argparse.ArgumentParser('Train Config Network')
    parser.add_argument('--cfg', type=str, help='path to config file', default='/home/whc/Desktop/Fitment/code/image-caption/cfgs/caption/base_4x16G_fp32.yaml')
    parser.add_argument('--model-dir', type=str, help='root path to store checkpoint', default='./')
    parser.add_argument('--log-dir', type=str, help='tensorboard log dir')
    parser.add_argument('--dist', help='', default=False, action='store_true')
    parser.add_argument('--do-test', help='whether to generate csv result on test set', default=False, action='store_true')
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
    # if args.do_test and (rank is None or rank == 0):
    # test_net(args, config)

if __name__ == '__main__':
    main()