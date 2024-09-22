import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile

# import .evaluation
from .evaluation import PTBTokenizer, Cider, compute_scores

device = torch.device('cuda')
a = open("/home/whc/Desktop/Fitment/code/image-caption/model/pretrained_model/bert-base-uncased/vocab.txt", "r")
b = a.readlines()
vocab = []
for i in b:
    vocab.append(i[:-1])

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


def evaluate_loss(model, dataloader, loss_fn, epoch):
    # Validation loss
    model.eval()
    running_loss = .0
    print("evaluate_loss")
    # ['image', 'boxes', 'im_info', 'text', 'tag_text']
    with tqdm(desc='Epoch %d - validation' % epoch, unit='it', total=len(dataloader)) as pbar:
    
        with torch.no_grad():
            for nbatch, batch in enumerate(dataloader):
                # 把batch中的每个变量单独取出来
                batch = to_cuda(batch)
                detections, captions = batch[1], batch[3]
                captions = captions[:, :200]
                out, loss = model(*batch)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, 5781), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (nbatch + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss

import json
def evaluate_metrics(model, dataloader, epoch):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    print("evaluate_metrics")

    with tqdm(desc='Epoch %d - evaluation' % epoch, unit='it', total=len(dataloader)) as pbar:
        f = open('caption-demo/fitment_captions_total_{}.txt'.format(epoch), 'a+', encoding='utf-8')
        for it, batch in enumerate(dataloader):
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

            for cap, gt in zip(caps_gen, caps_gt):
                # print("gen captioning: ", cap)
                # # print("\n") 
                # print("gt captioning: ", gt)

                f.write("".join(cap)+'\n')
                f.write("".join(gt)+'\n')
                         

            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()
        f.close()  

    gts = PTBTokenizer.tokenize(gts)
    gen = PTBTokenizer.tokenize(gen)

    scores, _ = compute_scores(gts, gen)
    return scores

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