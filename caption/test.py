import enum
import torch
import torch.nn
import torch.optim as optim
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP

import pprint
import os
import random
import shutil
import inspect
import numpy as np
from tqdm import tqdm
import itertools
from tensorboardX import SummaryWriter

from logging import Logger

from caption.utils.create_logger import create_logger
from caption.utils import summary_parameters, bn_fp16_half_eval
from caption.modules.resnet_vlbert_for_caption import ResNetVLBERTForCaption
from common import metrics
from common.callbacks.epoch_end_callbacks.checkpoint import Checkpoint
from common.utils.load import smart_load_model_state_dict
from common.nlp.bert.optimization import AdamW, WarmupLinearSchedule
from common.utils.create_logger import create_logger
from common.utils.misc import summary_parameters, bn_fp16_half_eval
from common.utils.load import smart_resume, smart_partial_load_model_state_dict
from common.trainer import train
from common.metrics.composite_eval_metric import CompositeEvalMetric
from common.metrics import pretrain_metrics
from common.callbacks.batch_end_callbacks.speedometer import Speedometer
from common.callbacks.epoch_end_callbacks.validation_monitor import ValidationMonitor
from common.callbacks.epoch_end_callbacks.checkpoint import Checkpoint
from common.lr_scheduler import WarmupMultiStepLR
from common.nlp.bert.optimization import AdamW, WarmupLinearSchedule
from common.utils.multi_task_dataloader import MultiTaskDataLoader

from caption.data.build import make_dataloaders, make_dataloader
from caption.function.val import do_validation

from .evaluation import PTBTokenizer, Cider, compute_scores

def to_cuda(batch):
    batch = list(batch)

    for i in range(len(batch)):
        if isinstance(batch[i], torch.Tensor):
            batch[i] = batch[i].cuda(non_blocking=True)
        elif isinstance(batch[i], list):
            for j, o in enumerate(batch[i]):
                if isinstance(batch[i], torch.Tensor):
                    batch[i][j] = o.cuda(non_blocking=True)

    return batch


device = torch.device('cuda')
a = open("/home/whc/Desktop/Fitment/code/image-caption/model/pretrained_model/bert-base-uncased/vocab.txt", "r")
b = a.readlines()
vocab = []
for i in b:
    vocab.append(i[:-1])

def decode(word_idxs, join_words=True):
    if isinstance(word_idxs, list) and len(word_idxs) == 0:
        return decode([word_idxs, ], join_words)[0]
    if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
        return decode([word_idxs, ], join_words)[0]
    elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
        return decode(word_idxs.reshape((1, -1)), join_words)[0]
    elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
        return decode(word_idxs.unsqueeze(0), join_words)[0]

    
    
    captions = []
    for wis in word_idxs:
        caption = []
        for wi in wis:
            word = vocab[int(wi)]
            if wi == 6 or wi == 0:
                break
            caption.append(word)
        if join_words:
            caption = ' '.join(caption)
        captions.append(caption)
    return captions

def test_net(args, config):
    
    # 设置日志记录
    logger, final_output_path = create_logger(
        config.OUTPUT_PATH,
        args.cfg,
        config.DATASET[0].TRAIN_IMAGE_SET if isinstance(config.DATASET, list)
        else config.DATASET.TRAIN_IMAGE_SET,
        split='train'
    )
    # 打印输出以及日志记录训练参数
    # pprint.pprint相比于print的输出更规范
    pprint.pprint(args)
    logger.info('training args:{}\n'.format(args))
    pprint.pprint(config)
    logger.info('training config:{}\n'.format(config))

    # 初始化模型，为什么这样可以初始化模型？
    # eval()可以把字符串变成一个表达式 eg. eval(config.MODULE) : "ResNetVLBERTForPretrainingMultitask" --> ResNetVLBERTForPretrainingMultitask()
    model = eval(config.MODULE)(config)
    torch.cuda.set_device(int(config.GPUS))
    model.cuda()
    # 加载预训练模型
    model = torch.load("./output/vlbert-decoder_0.pth")
    model.eval()
    gen = {}
    gts = {}
    test_loader = make_dataloader(config, mode='val', distributed=False)
    train_sampler = None
    epoch = 0
    with tqdm(desc='Epoch %d - evaluation' % epoch, unit='it', total=len(test_loader)) as pbar:
        for it, batch in enumerate(test_loader):
            batch = to_cuda(batch)
            # image, boxes, im_info, text, tag_text
            # image = batch[0]
            # boxes = batch[1]
            # im_info = batch[2]
            caps_gt = batch[3]
            # tag_text = batch[4]
            
            with torch.no_grad():
                # out, _ = model.beam_search(image, boxes, im_info, tag_text, 200, 100, 5, out_size=1)
                out, _ = model.beam_search(*batch)
            caps_gen = decode(out, join_words=False)
            caps_gt = decode(caps_gt, join_words=False)

            # for cap, gt in zip(caps_gen, caps_gt):
            #     print("caption: ", cap, "\n", "gt captioning: ", gt)

            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    # print("caption token-1: ", gts, gen)
    gts = PTBTokenizer.tokenize(gts)
    gen = PTBTokenizer.tokenize(gen)
    # print("caption token-2: ", gts, gen)
    scores, _ = compute_scores(gts, gen)
    return scores