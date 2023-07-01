'''
此文件用于生成数据集对应的mindrecord文件
'''
import os
import ruamel.yaml as yaml
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from mindspore.mindrecord import FileWriter
import sys
import h5py
from dataset.randaugment import RandomAugment
from mindspore.dataset.transforms import Compose
from mindspore.dataset import vision
from mindspore.dataset.vision import Inter


class Record:
    def __init__(self, config, is_train='train', transform=None):
        self.config=config
        self.is_train=is_train
        self.transform=transform
        self.img_shape=[384,384,3]

    def create_MindRecord(self, filepath, file_num=1, json_path=None):
        
        normalize = vision.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        
        # 对于图片，要关注图片的维度，尤其是图片的最后一维，是否是3
        writer = FileWriter(file_name=filepath, shard_num=file_num, overwrite=True)
        vqa_json = {
            "image": {"type": "float32", "shape": self.img_shape},
            'question': {"type": "string"},
            'answers': {"type": "string"},
            'weights': {"type": "string"}
        }
        schema_id = writer.add_schema(vqa_json, "vqa_schema")
        with open(json_path, 'r') as file:
            # 解析JSON数据
            data = json.load(file)

        # 遍历每个字典
        records = []
        cnt = 0
        for item in data:
            # 处理当前字典
            image_id=item['image']
            question=item['question']
            answers=item['answers']
            weights=item['weights']
            if self.is_train == 'train':
                image_path=os.path.join(self.config['train_image_path'], image_id)
            else:
                image_path=os.path.join(self.config['test_image_path'], image_id)
                
            image=Image.open(image_path).convert('RGB')
            #处理加载后的图片
            image = self.transform(image)
            image = np.array(image)
            image = image.reshape(3,384,384)
            image = image.transpose([1,2,0])
            image = normalize(image)
            
            #由于mindrecord不能储存string的list，因此将string拼接成一个大的string后再添加
            total_answers = ''
            for i in answers:
                total_answers += i+' '
            
            # weights的处理与以上类似
            total_weights=''
            for i in weights:
                total_weights += str(i) +' '
                
            # 将数据集中的内容转化成mindrecord文件并储存
            data = {'image': image, 'question': question, 'answers': total_answers, 'weights': total_weights}
            #import pdb
            #pdb.set_trace()
            records.append(data)
            cnt +=1
            print('currrent data number: ', cnt)
            if cnt == 5000:
                break
        
        writer.write_raw_data(records) 
        writer.commit()
        print("------------------SUCCESSFULLY GENERATED!!!------------------")


def generate(mode, filepath, config, json_path):
    
    #定义图像处理方式
    normalize = vision.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    pretrain_transform = Compose([
        vision.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0), interpolation=Inter.BICUBIC),
        vision.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        # 注意mindspore在使用totensor时会改变数据的维度，需要转换回原来的的维度
        vision.ToTensor(),
        normalize,
    ])

    train_transform = Compose([
        vision.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation=Inter.BICUBIC),
        vision.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        vision.ToTensor(),
        #normalize,
    ])

    test_transform = Compose([
        vision.Resize((config['image_res'], config['image_res']), interpolation=Inter.BICUBIC),
        vision.ToTensor(),
        #normalize,
    ])
    if mode == 'train':
        record = Record(config, mode, train_transform)
    else:
        record = Record(config, mode, test_transform)
    
    record.create_MindRecord(filepath, 1, json_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    
    # 生成训练数据的mindrecord文件
    train_record_path = os.path.join(config['mindrecord_path'], 'train.mindrecord')
    #generate('train', train_record_path, config, os.path.join(config['image_path'], config['train_json']))
    # 生成验证数据的mindrecord文件
    val_record_path = os.path.join(config['mindrecord_path'], 'val.mindrecord')
    #generate('train', val_record_path, config, os.path.join(config['image_path'], config['val_json']))
    # 生成测试数据的mindrecord文件
    test_record_path = os.path.join(config['mindrecord_path'], 'test.mindrecord')
    generate('test', test_record_path, config, os.path.join(config['image_path'], config['test_json']))


    