'''
定义VQA_DATASET类，允许通过索引或切片来访问数据
'''
import os
import json
import random
import sys
import h5py
import numpy as np
from PIL import Image
import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter
from dataset.utils import pre_question

# dataset可能无法正确检索到验证集里的数据，测试如果不成功应修改，修改思路为在yaml文件中将测试集文件和验证集文件的路径拆分
class VQA_Dataset:
    def __init__(self, ann_file, transform, vqa_root, vg_root, eos='[SEP]', split="train", max_ques_words=30, answer_list=''):
        self.split = split
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.max_ques_words = max_ques_words
        self.eos = eos

        if split == 'test':
            self.max_ques_words = 50  # do not limit question length during test
            self.answer_list = json.load(open(answer_list, 'r'))

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        if ann['dataset']=='vqa':
            image_path = os.path.join(self.vqa_root,ann['image'])
        elif ann['dataset']=='vg':
            image_path = os.path.join(self.vg_root,ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if self.split == 'test':
            question = pre_question(ann['question'], self.max_ques_words)
            question_id = ann['question_id']
            return image, question, question_id


        elif self.split == 'train':

            question = pre_question(ann['question'], self.max_ques_words)

            if ann['dataset'] == 'vqa':

                answer_weight = {}
                for answer in ann['answer']:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1 / len(ann['answer'])
                    else:
                        answer_weight[answer] = 1 / len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

            elif ann['dataset'] == 'vg':
                answers = [ann['answer']]
                weights = [0.5]

            answers = [answer + self.eos for answer in answers]

            return image, question, answers, weights

if __name__ == '__main__':
    import argparse
    import ruamel_yaml as yaml
    from dataset.randaugment import RandomAugment
    from mindspore.dataset import vision
    from PIL import Image

    normalize = vision.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)


    train_transform = vision.Compose([
        vision.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        vision.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        vision.ToTensor(),
        normalize,
    ])

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    train_dataset = VQA_Dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'],
                                split='train')

    # 检查dataset是否能够正确检索到数据
    print(train_dataset[100])
