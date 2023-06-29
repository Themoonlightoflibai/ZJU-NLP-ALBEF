from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel, BertLMHeadModel

import mindspore
from mindspore import nn
import mindspore.ops as op

class ALBEF(nn.Cell):
    def __init__(self,
                 text_encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None,):
        super().__init__()

        self.tokenizer = tokenizer
        self.distill = config['distill']

