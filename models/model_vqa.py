from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel, BertLMHeadModel

import mindspore
from mindspore import nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor

import numpy as np
import pdb

class ALBEF(nn.Cell):
    def __init__(self,
                 text_encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None,):
        super().__init__()

        self.tokenizer = tokenizer
        self.distill = config['distill']
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon =1e-6), drop_path_rate=0.5, drop_rate=0.5)
        
        bertconfig_encoder = BertConfig()
        config_encoder = bertconfig_encoder.from_json_file(config['bert_config'])
        self.text_encoder = BertModel(config=config_encoder)
        
        
        bertconfig_decoder = BertConfig()
        config_decoder = bertconfig_decoder.from_json_file(config['bert_config'])
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6
        self.text_decoder = BertLMHeadModel(config=config_decoder)

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), drop_path_rate=0.5, drop_rate=0.5)
            self.text_encoder_m = BertModel(config=config_encoder)
            self.text_decoder_m = BertLMHeadModel(config=config_decoder)
            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.text_decoder,self.text_decoder_m],
                               ]
            self.copy_params()
            self.momentum = 0.995

    def construct(self, image, question, answer=None, alpha=0, k=None, weights=None, train=True, tokenizer = None):
        
        #此处读入的answer和weights是字符串的形式，转换成为原来的格式
        answer = answer.split(' ')
        answer = answer[:-1]
        
        weights = weights.split(' ')
        weights = weights[:-1]
        
        
        for i in range(len(weights)):
            weights[i]=float(weights[i])
        
        
        #用tokenizer做处理
        question = self.tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt")
        answer = self.tokenizer(answer, padding='longest', return_tensors="pt") 
        
        #将数据转成ms的tensor
        
        image_embeds = self.visual_encoder(image)
        # 也可以是mindspore.int64
        image_atts = ops.ones(image_embeds.size()[:-1], dtype=mindspore.long)

        if train:
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

            question_output = self.text_encoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)   

            question_states = []
            question_atts = []
            for b, n in enumerate(k):
                question_states += [question_output.last_hidden_state[b]] * n
                question_atts += [question.attention_mask[b]] * n
            question_states = ops.stack(question_states, 0)
            question_atts = ops.stack(question_atts, 0)

            if self.distill:
                #mindspore不需要此语句
                #with torch.no_grad():
                self._momentum_update()
                image_embeds_m = self.visual_encoder_m(image)
                question_output_m = self.text_encoder_m(question.input_ids,
                                                            attention_mask=question.attention_mask,
                                                            encoder_hidden_states=image_embeds_m,
                                                            encoder_attention_mask=image_atts,
                                                            return_dict=True)

                question_states_m = []
                for b, n in enumerate(k):
                    question_states_m += [question_output_m.last_hidden_state[b]] * n
                question_states_m = ops.stack(question_states_m, 0)

                logits_m = self.text_decoder_m(answer.input_ids,
                                                   attention_mask=answer.attention_mask,
                                                   encoder_hidden_states=question_states_m,
                                                   encoder_attention_mask=question_atts,
                                                   return_logits=True,
                                                   )

                answer_output = self.text_decoder(answer.input_ids,
                                                  attention_mask=answer.attention_mask,
                                                  encoder_hidden_states=question_states,
                                                  encoder_attention_mask=question_atts,
                                                  labels=answer_targets,
                                                  return_dict=True,
                                                  soft_labels=ops.softmax(logits_m, axis =-1),
                                                  alpha=alpha,
                                                  reduction='none',
                                                  )
            else:
                answer_output = self.text_decoder(answer.input_ids,
                                                  attention_mask=answer.attention_mask,
                                                  encoder_hidden_states=question_states,
                                                  encoder_attention_mask=question_atts,
                                                  labels=answer_targets,
                                                  return_dict=True,
                                                  reduction='none',
                                                  )
            loss = weights * answer_output.loss
            loss = loss.sum() / image.size(0)

            return loss


        else:
            question_output = self.text_encoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)
            topk_ids, topk_probs = self.rank_answer(question_output.last_hidden_state, question.attention_mask,
                                                    answer.input_ids, answer.attention_mask, k)
            return topk_ids, topk_probs

    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].get_parameters(), model_pair[1].get_parameters()):
                #param_m.data.copy(param.data)  # initialize
                param_m = param.data.copy()
                # mindspore框架无此参数
                #param_m.requires_grad = False  # not update by gradient
        return

    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].get_parameters(), model_pair[1].get_parameters()):
                param_m = (param_m.data * self.momentum + param.data * (1. - self.momentum)).copy()
        return

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):

        num_ques = question_states.size(0)
        # 同repeat函数，详情见官方文档
        start_ids = answer_ids[0, 0].tile(num_ques, 1)  # bos token

        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states=question_states,
                                         encoder_attention_mask=question_atts,
                                         return_dict=True,
                                         reduction='none')
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = ops.softmax(logits, axis=1).index_select(axis=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(axis=0, index=topk_id))
            input_atts.append(answer_atts.index_select(axis=0, index=topk_id))
        input_ids = ops.cat(input_ids, axis=0)
        input_atts = ops.cat(input_atts, axis=0)

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(input_ids,
                                   attention_mask=input_atts,
                                   encoder_hidden_states=question_states,
                                   encoder_attention_mask=question_atts,
                                   labels=targets_ids,
                                   return_dict=True,
                                   reduction='none')

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = ops.cat([topk_probs.log(), -answer_loss], axis=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)

        topk_probs = ops.softmax(log_probs_sum, axis=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = ops.gather_elements(topk_ids, 1, rerank_id)

        return topk_ids, topk_probs


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = Tensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]), mindspore.int64)
    return ops.index_select(x, dim, order_index.to(x.device))
