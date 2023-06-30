'''
此文件的作用为读取VQA数据集并从其中读取一小部分数据作为数据集
由于原VQA文件过大，上传所需时间太长且训练时间也太长，因此简化数据集
'''

import os
import json
import random
import argparse
from pathlib import Path
import ruamel_yaml as yaml
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
#from dataset.randaugment import RandomAugment
import shutil
import re



import cv2
import numpy as np


## aug functions
def identity_func(img):
    return img


def autocontrast_func(img, cutoff=0):
    '''
        same output as PIL.ImageOps.autocontrast
    '''
    n_bins = 256

    def tune_channel(ch):
        n = ch.size
        cut = cutoff * n // 100
        if cut == 0:
            high, low = ch.max(), ch.min()
        else:
            hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
            low = np.argwhere(np.cumsum(hist) > cut)
            low = 0 if low.shape[0] == 0 else low[0]
            high = np.argwhere(np.cumsum(hist[::-1]) > cut)
            high = n_bins - 1 if high.shape[0] == 0 else n_bins - 1 - high[0]
        if high <= low:
            table = np.arange(n_bins)
        else:
            scale = (n_bins - 1) / (high - low)
            offset = -low * scale
            table = np.arange(n_bins) * scale + offset
            table[table < 0] = 0
            table[table > n_bins - 1] = n_bins - 1
        table = table.clip(0, 255).astype(np.uint8)
        return table[ch]

    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


def equalize_func(img):
    '''
        same output as PIL.ImageOps.equalize
        PIL's implementation is different from cv2.equalize
    '''
    n_bins = 256

    def tune_channel(ch):
        hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
        non_zero_hist = hist[hist != 0].reshape(-1)
        step = np.sum(non_zero_hist[:-1]) // (n_bins - 1)
        if step == 0: return ch
        n = np.empty_like(hist)
        n[0] = step // 2
        n[1:] = hist[:-1]
        table = (np.cumsum(n) // step).clip(0, 255).astype(np.uint8)
        return table[ch]

    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


def rotate_func(img, degree, fill=(0, 0, 0)):
    '''
    like PIL, rotate by degree, not radians
    '''
    H, W = img.shape[0], img.shape[1]
    center = W / 2, H / 2
    M = cv2.getRotationMatrix2D(center, degree, 1)
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill)
    return out


def solarize_func(img, thresh=128):
    '''
        same output as PIL.ImageOps.posterize
    '''
    table = np.array([el if el < thresh else 255 - el for el in range(256)])
    table = table.clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def color_func(img, factor):
    '''
        same output as PIL.ImageEnhance.Color
    '''
    ## implementation according to PIL definition, quite slow
    #  degenerate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
    #  out = blend(degenerate, img, factor)
    #  M = (
    #      np.eye(3) * factor
    #      + np.float32([0.114, 0.587, 0.299]).reshape(3, 1) * (1. - factor)
    #  )[np.newaxis, np.newaxis, :]
    M = (
            np.float32([
                [0.886, -0.114, -0.114],
                [-0.587, 0.413, -0.587],
                [-0.299, -0.299, 0.701]]) * factor
            + np.float32([[0.114], [0.587], [0.299]])
    )
    out = np.matmul(img, M).clip(0, 255).astype(np.uint8)
    return out


def contrast_func(img, factor):
    """
        same output as PIL.ImageEnhance.Contrast
    """
    mean = np.sum(np.mean(img, axis=(0, 1)) * np.array([0.114, 0.587, 0.299]))
    table = np.array([(
        el - mean) * factor + mean
        for el in range(256)
    ]).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def brightness_func(img, factor):
    '''
        same output as PIL.ImageEnhance.Contrast
    '''
    table = (np.arange(256, dtype=np.float32) * factor).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def sharpness_func(img, factor):
    '''
    The differences the this result and PIL are all on the 4 boundaries, the center
    areas are same
    '''
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1][1] = 5
    kernel /= 13
    degenerate = cv2.filter2D(img, -1, kernel)
    if factor == 0.0:
        out = degenerate
    elif factor == 1.0:
        out = img
    else:
        out = img.astype(np.float32)
        degenerate = degenerate.astype(np.float32)[1:-1, 1:-1, :]
        out[1:-1, 1:-1, :] = degenerate + factor * (out[1:-1, 1:-1, :] - degenerate)
        out = out.astype(np.uint8)
    return out


def shear_x_func(img, factor, fill=(0, 0, 0)):
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def translate_x_func(img, offset, fill=(0, 0, 0)):
    '''
        same output as PIL.Image.transform
    '''
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, -offset], [0, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def translate_y_func(img, offset, fill=(0, 0, 0)):
    '''
        same output as PIL.Image.transform
    '''
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, 0], [0, 1, -offset]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def posterize_func(img, bits):
    '''
        same output as PIL.ImageOps.posterize
    '''
    out = np.bitwise_and(img, np.uint8(255 << (8 - bits)))
    return out


def shear_y_func(img, factor, fill=(0, 0, 0)):
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, 0], [factor, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def cutout_func(img, pad_size, replace=(0, 0, 0)):
    replace = np.array(replace, dtype=np.uint8)
    H, W = img.shape[0], img.shape[1]
    rh, rw = np.random.random(2)
    pad_size = pad_size // 2
    ch, cw = int(rh * H), int(rw * W)
    x1, x2 = max(ch - pad_size, 0), min(ch + pad_size, H)
    y1, y2 = max(cw - pad_size, 0), min(cw + pad_size, W)
    out = img.copy()
    out[x1:x2, y1:y2, :] = replace
    return out


### level to args
def enhance_level_to_args(MAX_LEVEL):
    def level_to_args(level):
        return ((level / MAX_LEVEL) * 1.8 + 0.1,)
    return level_to_args


def shear_level_to_args(MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = (level / MAX_LEVEL) * 0.3
        if np.random.random() > 0.5: level = -level
        return (level, replace_value)

    return level_to_args


def translate_level_to_args(translate_const, MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = (level / MAX_LEVEL) * float(translate_const)
        if np.random.random() > 0.5: level = -level
        return (level, replace_value)

    return level_to_args


def cutout_level_to_args(cutout_const, MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * cutout_const)
        return (level, replace_value)

    return level_to_args


def solarize_level_to_args(MAX_LEVEL):
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * 256)
        return (level, )
    return level_to_args


def none_level_to_args(level):
    return ()


def posterize_level_to_args(MAX_LEVEL):
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * 4)
        return (level, )
    return level_to_args


def rotate_level_to_args(MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = (level / MAX_LEVEL) * 30
        if np.random.random() < 0.5:
            level = -level
        return (level, replace_value)

    return level_to_args


func_dict = {
    'Identity': identity_func,
    'AutoContrast': autocontrast_func,
    'Equalize': equalize_func,
    'Rotate': rotate_func,
    'Solarize': solarize_func,
    'Color': color_func,
    'Contrast': contrast_func,
    'Brightness': brightness_func,
    'Sharpness': sharpness_func,
    'ShearX': shear_x_func,
    'TranslateX': translate_x_func,
    'TranslateY': translate_y_func,
    'Posterize': posterize_func,
    'ShearY': shear_y_func,
}

translate_const = 10
MAX_LEVEL = 10
replace_value = (128, 128, 128)
arg_dict = {
    'Identity': none_level_to_args,
    'AutoContrast': none_level_to_args,
    'Equalize': none_level_to_args,
    'Rotate': rotate_level_to_args(MAX_LEVEL, replace_value),
    'Solarize': solarize_level_to_args(MAX_LEVEL),
    'Color': enhance_level_to_args(MAX_LEVEL),
    'Contrast': enhance_level_to_args(MAX_LEVEL),
    'Brightness': enhance_level_to_args(MAX_LEVEL),
    'Sharpness': enhance_level_to_args(MAX_LEVEL),
    'ShearX': shear_level_to_args(MAX_LEVEL, replace_value),
    'TranslateX': translate_level_to_args(
        translate_const, MAX_LEVEL, replace_value
    ),
    'TranslateY': translate_level_to_args(
        translate_const, MAX_LEVEL, replace_value
    ),
    'Posterize': posterize_level_to_args(MAX_LEVEL),
    'ShearY': shear_level_to_args(MAX_LEVEL, replace_value),
}


class RandomAugment(object):

    def __init__(self, N=2, M=10, isPIL=False, augs=[]):
        self.N = N
        self.M = M
        self.isPIL = isPIL
        if augs:
            self.augs = augs
        else:
            self.augs = list(arg_dict.keys())

    def get_random_ops(self):
        sampled_ops = np.random.choice(self.augs, self.N)
        return [(op, 0.5, self.M) for op in sampled_ops]

    def __call__(self, img):
        if self.isPIL:
            img = np.array(img)
        ops = self.get_random_ops()
        for name, prob, level in ops:
            if np.random.random() > prob:
                continue
            args = arg_dict[name](level)
            img = func_dict[name](img, *args)
        return img


def pre_question(question, max_ques_words):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace('-', ' ').replace('/', ' ')
    question = question.rstrip(' ')

    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_ques_words:
        question = ' '.join(question_words[:max_ques_words])

    return question


class vqa_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root, vg_root, eos='[SEP]', split="train", max_ques_words=30,
                 answer_list=''):
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

        if ann['dataset'] == 'vqa':
            image_path = (self.vqa_root+'\\' + ann['image']).replace("\\", '/')
            #image_path = os.path.join(self.vqa_root, ann['image'])
        elif ann['dataset'] == 'vg':
            image_path = os.path.join(self.vg_root, ann['image'])

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

            return image, question, answers, weights, ann['image']


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
    parser.add_argument('--tiny', default=True, type=bool)
    parser.add_argument('--is_training', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)


    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = vqa_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'],
                                split='train')
    val_dataset = vqa_dataset(config['val_file'], train_transform, config['vqa_root'], config['vg_root'],
                              split='train')
    vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'],
                                   split='test', answer_list=config['answer_list'])
    with open('tinytest.json', 'a') as file:
        file.write('[')
    min_file_num = 10000
    max_file_num = 15000
    for i in range(min_file_num, max_file_num):
        image, question, answers, weights, image_id = val_dataset[i]
        #print(image.tolist())
        #print(question)
        #print(answers)
        #print(weights)
        print('image id: '+image_id)

        data={
            'image' : image_id,
            'question': question,
            'answers': answers,
            'weights': weights
        }

        # 将原数据集的中的数据拷贝到新的文件夹中，根据需要替换成相应的文件地址
        source_path = (config['vqa_root']+'\\' + image_id).replace("\\", '/')
        target_path = ('D:\\study\\NLP\\lab\\project\\data\\image\\test2014\\test2014tiny/'+image_id).replace("\\", '/')
        print(source_path)
        print(target_path)


        shutil.copy(source_path, target_path)

        with open('tinytest.json', 'a') as file:
            json.dump(data, file)
            if i != max_file_num-1:
                file.write(',')
        print("add item", i)

    with open('tinytest.json', 'a') as file:
        file.write(']')

