from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel, BertLMHeadModel

import mindspore
from mindspore import nn
import mindspore.ops as op