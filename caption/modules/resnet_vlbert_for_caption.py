from json import decoder
import os
import torch
import torch.nn as nn
from torch.nn import NLLLoss
import torch.nn.functional as F
from external.pytorch_pretrained_bert import BertTokenizer
from common.module import CapModule
from common.fast_rcnn import FastRCNN
from common.visual_linguistic_bert import VisualLinguisticBert, VisualLinguisticBertForPretraining, VisualLinguisticBertForCaptioning
from common.utils.misc import soft_cross_entropy
from pretrain.modules.resnet_vlbert_for_attention_vis import BERT_WEIGHTS_NAME
from caption.modules.decoder import Decoder
from caption.modules.beam_search.beam_search import BeamSearch
from typing import Union, Sequence, Tuple
# import torch

TensorOrSequence = Union[Sequence[torch.Tensor], torch.Tensor]
TensorOrNone = Union[torch.Tensor, None]
# todo: add this to config
NUM_SPECIAL_WORDS = 1000


BERT_WEIGHTS_NAME = 'pytorch_model.bin'
loss_fn = NLLLoss(ignore_index=3) # ignore_index=text_field.vocab.stoi['<pad>']


class ResNetVLBERTForCaption(CapModule):
    def __init__(self, config):
        super(ResNetVLBERTForCaption, self).__init__(config)

        self._state_names = []
        self.bos_idx = 5
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)

        self.image_feature_extractor = FastRCNN(
            config,
            average_pool=True,
            final_dim=config.NETWORK.IMAGE_FINAL_DIM,
            enable_cnn_reg_loss=False
        )
        # 文本标识符
        self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        # IMAGE_FEAT_PRECOMPUTE，用来处理图像特征，具体的作用是什么？
        if config.NETWORK.IMAGE_FEAT_PRECOMPUTED:
            self.object_mask_visual_embedding = nn.Embedding(1, 2048)
        # MVRC_LOSS是预测掩码图像特征类别的loss
        if config.NETWORK.WITH_MVRC_LOSS:
            self.object_mask_word_embedding = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        # 这个是用来干什么的？
        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN
        # BERT分词
        self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)
        language_pretrained_model_path = None
        if config.NETWORK.BERT_PRETRAINED != '':
            language_pretrained_model_path = '{}-{:0.4d}.model'.format(config.NETWORK.BERT_PRETRAINED,
                                                                      config.NETWORK.BERT_PRETRAINED_EPOCH)
        elif os.path.isdir(config.NETWORK.BERT_MODEL_NAME):
            weight_path = os.path.join(config.NETWORK.BERT_MODEL_NAME, BERT_WEIGHTS_NAME) 
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path

        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = VisualLinguisticBertForCaptioning(
            config.NETWORK.VLBERT,
            language_pretrained_model_path=None if config.NETWORK.VLBERT.from_scratch else language_pretrained_model_path,
            with_rel_head=False,
            with_mlm_head=False,
            with_mvrc_head=False,
        )

        # Decoder定义在这里，KG检索模块定义在Deoder中
        self.vocab_size = config.NETWORK.DECODER.vocab_size
        self.decoder = Decoder(config.NETWORK.DECODER.vocab_size,
                               config.NETWORK.DECODER.max_len,
                               config.NETWORK.DECODER.N_dec,
                               config.NETWORK.DECODER.padding_idx)
    
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        if self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED:
            self.object_mask_visual_embedding.weight.data.fill_(0.0)
        if self.config.NETWORK.WITH_MVRC_LOSS:
            self.object_mask_word_embedding.weight.data.normal_(mean=0.0, std=self.config.NETWORK.VLBERT.initializer_range)
        self.image_feature_extractor.init_weight()
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0,
                                                                  std=self.config.NETWORK.VLBERT.initializer_range)

    def train(self, mode=True):
        super(ResNetVLBERTForCaption, self).train(mode=mode)

        if self.image_feature_bn_eval:
             self.image_feature_extractor.bn_eval()

    def fix_params(self):
        pass

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def forward(self,
                image,
                boxes,
                im_info,
                text,
                tag_text):
        # mlm_labels可以替换成caption
        # text是tag关键词
        # 掩码mask都可以去掉，保留caption的

        images = image # 原始图像的像素点，默认为None
        box_mask = (boxes[:, :, 0] > - 1.5) # boxes是真正的特征，把box坐标和frcnn特征拼接在了一起，所以维度是2048+4=2052
        max_len = int(box_mask.sum(1).max().item())
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        with torch.no_grad():
            obj_reps = self.image_feature_extractor(images=images,
                                                    boxes=boxes,
                                                    box_mask=box_mask,
                                                    im_info=im_info,
                                                    classes=None,
                                                    segms=None)
            # 准备tags信息
            tag_input_ids = tag_text
            text_tags = tag_text.new_zeros(tag_text.shape)
            tag_token_type_ids = tag_text.new_zeros(tag_text.shape)
            tag_mask = (tag_input_ids > 0)
            # tag_visual_embedding的作用是什么？
            tag_visual_embedding = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])

            object_linguishtic_embedding = self.object_linguistic_embeddings(
                boxes.new_zeros(boxes.shape[0], boxes.shape[1]).long()
            )
            object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguishtic_embedding), -1)

            # 把视觉特征保存
            encoded_layers, encoded_layers_text, encoded_layers_object, pooled_output, extended_attention_mask = self.vlbert(tag_input_ids,
                                                                    tag_token_type_ids,
                                                                    tag_visual_embedding,
                                                                    tag_mask,
                                                                    object_vl_embeddings,
                                                                    box_mask,
                                                                    output_all_encoded_layers=False)

        # text = text[:, :200]
        # print("vl_out: ", vl_out)
        # print("extended_attention_mask: ", extended_attention_mask)
        # dec_output = self.decoder(text, vl_out, extended_attention_mask)

        # 计算loss
        # captions_gt = text[:, 1:].contiguous()
        # # print(captions_gt.size())
        # out = dec_output[:, :-1].contiguous()
        # # print(out.size())
        # loss = loss_fn(out.view(-1, self.vocab_size), captions_gt.view(-1))
        loss = 0

        return encoded_layers_object, encoded_layers_text, loss


    def beam_search(self,
                    image: TensorOrSequence,
                    boxes: TensorOrSequence,
                    im_info: TensorOrSequence,
                    tag_text: TensorOrSequence,
                    text: TensorOrSequence,
                    max_len=50, eos_idx=6, beam_size=8, out_size=1,
                    return_probs=False, **kwargs):


        bs = BeamSearch(self, max_len, eos_idx, beam_size)

        images = image # 原始图像的像素点，默认为None
        box_mask = (boxes[:, :, 0] > - 1.5) # boxes是真正的特征，把box坐标和frcnn特征拼接在了一起，所以维度是2048+4=2052
        max_len = int(box_mask.sum(1).max().item())
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)
        # 准备tags信息
        tag_input_ids = tag_text
        text_tags = tag_text.new_zeros(tag_text.shape)
        tag_token_type_ids = tag_text.new_zeros(tag_text.shape)
        tag_mask = (tag_input_ids > 0)
        # tag_visual_embedding的作用是什么？
        tag_visual_embedding = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])

        object_linguishtic_embedding = self.object_linguistic_embeddings(
            boxes.new_zeros(boxes.shape[0], boxes.shape[1]).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguishtic_embedding), -1)

        vl_out, pooled_rep, extended_attention_mask, embedding_output = self.vlbert(tag_input_ids,
                                                                                    tag_token_type_ids,
                                                                                    tag_visual_embedding,
                                                                                    tag_mask,
                                                                                    object_vl_embeddings,
                                                                                    box_mask,
                                                                                    output_all_encoded_layers=False)



        return bs.apply(embedding_output, vl_out, extended_attention_mask, out_size, return_probs, **kwargs)

    def step(self, t, vl_out, extended_attention_mask, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.enc_output, self.mask_enc = vl_out, extended_attention_mask
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc)


    