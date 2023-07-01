import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import mindspore
import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.nn import Momentum
from mindspore.common import set_seed
#from utils.config import config
from utils.lr_schedule import dynamic_lr
from mindspore.train import LossMonitor, TimeMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore import amp, nn
from mindspore.train.model import Model

from models.model_vqa import ALBEF
from models.vit import interpolate_pos_embed
#from models.tokenization_bert import BertTokenizer

from Minddataset import create_vqa_dataset

#from scheduler import create_scheduler
#from optim import create_optimizer


def train(model, train_loader, eval_loader, epochs, config):
    #需要添加保存模型预训练权重的功能
    dataset = train_loader
    dataset_size = dataset.get_dataset_size()

    # 定义学习率
    lr = Tensor(dynamic_lr(config, dataset_size), mstype.float32)
    opt = Momentum(params=model.trainable_params(), learning_rate=lr, momentum=config.momentum, \
                   weight_decay=config.weight_decay)

    loss_scale = amp.FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    model = Model(model, loss_fn=None, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                  amp_level="O2", keep_batchnorm_fp32=False)

    time_cb = TimeMonitor(data_size=config.step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    print("begin training...")
    model.train(epochs, train_loader, callbacks=cb, dataset_sink_mode=True)
    print("begin evaluating...")
    model.eval(epochs, eval_loader, callbacks=cb, dataset_sink_mode=True)


def test(model,test_loader, epochs, config):
    #此函数还需要完善加载预训练权重的功能，并根据预训练的权重来计算精确度
    dataset = test_loader
    dataset_size = dataset.get_dataset_size()

    # 定义学习率
    lr = Tensor(dynamic_lr(config, dataset_size), mstype.float32)
    opt = Momentum(params=model.trainable_params(), learning_rate=lr, momentum=config.momentum, \
                   weight_decay=config.weight_decay)

    loss_scale = amp.FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    model = Model(model, loss_fn=None, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                  amp_level="O2", keep_batchnorm_fp32=False)

    time_cb = TimeMonitor(data_size=config.step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]

    print("begin testing...")
    model.eval(epochs, test_loader, callbacks=cb, dataset_sink_mode=True)

def main(args, config):
    seed = args.seed
    mindspore.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    # 根据mindrecord创建数据集
    print("Creating vqa dataset according to MindRecord")
    
    train_dataset, val_datset = create_vqa_dataset(config, config['batch_size_train'], repeat_num=1, device_num=1,rank_id=0, is_training=True, num_parallel_workers=6, is_tiny=args.tiny)
    test_dataset = create_vqa_dataset(config, config['batch_size_test'], repeat_num=1, device_num=1, rank_id=0, is_training=False, num_parallel_workers=6, is_tiny=args.tiny)

    # 构造模型
    print("BUILDING ALBEF MODEL")
    
    '''
    配置完环境后更改为BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    '''
    tokenizer=None
    
    model = ALBEF(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)

    # 模型训练
    if args.is_training:
        train(model, train_dataset, val_datset, max_epoch, config)
    # 模型加载预训练权重，进行验证
    if not args.is_training:
        #测试函数还需完善
        test(model, test_dataset, max_epoch, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--tiny', default=False, type=bool)
    parser.add_argument('--is_training', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
