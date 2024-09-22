import os
import time
from collections import namedtuple
import torch
from torch.nn import NLLLoss
from caption.evaluate import evaluate_loss, evaluate_metrics
from tqdm import tqdm

try:
    from apex import amp
    from apex.amp import _amp_state
except ImportError:
    pass
    #raise ImportError("Please install apex from https://www.github.com/nvidia/apex if you want to use fp16.")

# Parameter to pass to batch_end_callback
BatchEndParam = namedtuple('BatchEndParams',
                           ['epoch',
                            'nbatch',
                            'rank',
                            'add_step',
                            'data_in_time',
                            'data_transfer_time',
                            'forward_time',
                            'backward_time',
                            'optimizer_time',
                            'metric_time',
                            'eval_metric',
                            'locals'])


def _multiple_callbacks(callbacks, *args, **kwargs):
    """Sends args and kwargs to any configured callbacks.
    This handles the cases where the 'callbacks' variable
    is ``None``, a single function, or a list.
    """
    if isinstance(callbacks, list):
        for cb in callbacks:
            cb(*args, **kwargs)
        return
    if callbacks:
        callbacks(*args, **kwargs)


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


def train(net,
          optimizer,
          lr_scheduler,
          train_loader,
          val_loader,
          train_sampler,
          metrics,
          begin_epoch,
          end_epoch,
          logger,
          rank=None,
          batch_end_callbacks=None,
          epoch_end_callbacks=None,
          writer=None,
          validation_monitor=None,
          fp16=False,
          clip_grad_norm=-1,
          gradient_accumulate_steps=1):

    assert isinstance(gradient_accumulate_steps, int) and gradient_accumulate_steps >= 1

    loss_fn = NLLLoss(ignore_index=3)

    for epoch in range(begin_epoch, end_epoch):
        print('PROGRESS: %.2f%%' % (100.0 * epoch / end_epoch))

        # set epoch as random seed of sampler while distributed training
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        # reset metrics
        metrics.reset()

        # set net to train mode
        net.train()

        # clear the paramter gradients
        # optimizer.zero_grad()

        # init end time
        end_time = time.time()

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            name, value = validation_monitor.metrics.get()
            val = value[name.index(validation_monitor.host_metric_name)]
            lr_scheduler.step(val, epoch)

        # training
        import numpy as np
        object_dir = "/home/whc/Desktop/Fitment/code/VL-BERT-master/object_feature"
        text_dir = "/home/whc/Desktop/Fitment/code/VL-BERT-master/text_feature"
        running_loss = .0
        with tqdm(desc='Epoch %d - evaluation' % epoch, unit='it', total=len(train_loader)) as pbar:
            for nbatch, batch in enumerate(train_loader):
                # print(batch[1].size())
                global_steps = len(train_loader) * epoch + nbatch
                os.environ['global_steps'] = str(global_steps)

                # record time
                data_in_time = time.time() - end_time

                # transfer data to GPU
                data_transfer_time = time.time()
                batch = to_cuda(batch)
                data_transfer_time = time.time() - data_transfer_time

                # forward
                forward_time = time.time()
                # net(*batch)返回值为None，当把None同时赋予多个变量时就会出现这个报错：'NoneType' object is not iterable
                encoded_layers_object, encoded_layers_text, loss = net(*batch)
                # ===========================================================
                # 在這裡提取特徵
                for _i, hidden_states in enumerate(encoded_layers_object):
                    index = int(batch[2][_i][-1])
                    if hasattr(train_loader.dataset, 'ids'):
                        image_id = train_loader.dataset.ids[index]
                    else:
                        image_id = train_loader.dataset.database[index]['image'].split('/')[1].split('.')[0]
                    # attention_probs_arr = attention_probs.detach().cpu().numpy()
                    hidden_states_arr = hidden_states.detach().cpu().numpy()
                    # cos_similarity_arr = (hidden_states @ hidden_states.transpose(1, 2)).detach().cpu().numpy()
                    # np.save(os.path.join(attention_dir, '{}.npy'.format(image_id)), attention_probs_arr)
                    np.save(os.path.join(object_dir, '{}.npy'.format(image_id)), hidden_states_arr)
                    # np.save(os.path.join(hidden_dir, '{}.npy'.format(image_id)), hidden_states_arr)
                    # np.save(os.path.join(cos_dir, '{}.npy'.format(image_id)), cos_similarity_arr)
                    # index = (index + 1) % len(loader.dataset)

                for _i, hidden_states_t in enumerate(encoded_layers_text):
                    index = int(batch[2][_i][-1])
                    if hasattr(train_loader.dataset, 'ids'):
                        image_id = train_loader.dataset.ids[index]
                    else:
                        image_id = train_loader.dataset.database[index]['image'].split('/')[1].split('.')[0]
                    # attention_probs_arr = attention_probs.detach().cpu().numpy()
                    hidden_states_arr_t = hidden_states_t.detach().cpu().numpy()
                    # cos_similarity_arr = (hidden_states @ hidden_states.transpose(1, 2)).detach().cpu().numpy()
                    # np.save(os.path.join(attention_dir, '{}.npy'.format(image_id)), attention_probs_arr)
                    np.save(os.path.join(text_dir, '{}.npy'.format(image_id)), hidden_states_arr_t)
                # ==============================================================

                # loss = loss.mean()
                # if gradient_accumulate_steps > 1:
                #     loss = loss / gradient_accumulate_steps
                # forward_time = time.time() - forward_time

                # # backward
                # backward_time = time.time()
                # if fp16:
                #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                #     loss.backward()

                # backward_time = time.time() - backward_time

                # optimizer_time = time.time()
                # if (global_steps + 1) % gradient_accumulate_steps == 0:
                #     # step LR scheduler
                #     if lr_scheduler is not None and not isinstance(lr_scheduler,
                #                                                 torch.optim.lr_scheduler.ReduceLROnPlateau):
                #         lr_scheduler.step()

                #     # clip gradient
                #     if clip_grad_norm > 0:
                #         if fp16:
                #             total_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                #                                                         clip_grad_norm)
                #         else:
                #             total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(),
                #                                                         clip_grad_norm)
                #         if writer is not None:
                #             writer.add_scalar(tag='grad-para/Total-Norm',
                #                             scalar_value=float(total_norm),
                #                             global_step=global_steps)

                #     optimizer.step()
                #     # clear the parameter gradients
                #     optimizer.zero_grad()
                # optimizer_time = time.time() - optimizer_time

                # this_loss = loss.item()
                # running_loss += this_loss
                # pbar.set_postfix(loss=running_loss / (nbatch + 1))
                pbar.update()

                # print("epoch: {}, loss: {}".format(epoch, loss.item()))

                # update metric
                # scores = evaluate_metrics(net, val_loader, epoch)
                # print("Validation scores", scores)

        # path = "output/vlbert-decoder_{}.pth".format(epoch)
        # torch.save(net, path)

        param_name = 'output-1/{}-{}.model'.format('caption', epoch)
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = net.state_dict()
        checkpoint_dict['optimizer'] = optimizer.state_dict()
        torch.save(checkpoint_dict, param_name)

        val_loss = evaluate_loss(net, val_loader, loss_fn, epoch)
        print("Validation loss", val_loss)

        # update metric
        # if epoch % 2 == 0 and epoch != 0:
        scores = evaluate_metrics(net, val_loader, epoch)
        print("Validation scores", scores)
