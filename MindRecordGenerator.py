'''
此文件用于生成数据集对应的mindrecord文件
'''
import os
import ruamel_yaml as yaml
import json
import argparse
from pathlib import Path
from dataset.create_MindRecord import Record
from dataset.vqa_dataset import VQA_Dataset
from dataset import create_dataset


def generate(mode, filepath, dataset):
    record = Record(dataset, mode)
    record.create_MindRecord(filepath, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    train_dataset, val_dataset, test_dataset = create_dataset(config)
    # 生成训练数据的mindrecord文件
    train_record_path = os.path.join(config['mindrecord_path'], 'train.mindrecord')
    generate('train', train_record_path, train_dataset)
    # 生成验证数据的mindrecord文件
    val_record_path = os.path.join(config['mindrecord_path'], 'val.mindrecord')
    generate('val', val_record_path, val_dataset)
    # 生成测试数据的mindrecord文件
    test_record_path = os.path.join(config['mindrecord_path'], 'test.mindrecord')
    generate('test', test_record_path, test_dataset)
