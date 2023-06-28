from __future__ import division
import os
import numpy as np
from PIL import Image
from mindspore.mindrecord import FileWriter
from __future__ import print_function
import sys
import json
import h5py

class Record:
    def __init__(self, dataset, is_train='train'):
        self.dataset=dataset
        self.is_train=is_train

    def create_MindRecord(self, filepath, file_num=1):
        # 对于图片，要关注图片的维度，尤其是图片的最后一维，是否是3
        writer = FileWriter(file_name=filepath, shard_num=file_num, overwrite=True)
        vqa_json = {
            "image": {"type": "float32", "shape": [-1, self.img_shape[-1]]},
            'question': {"type": "string"},
            'answers': {"type": "string"},
            'weights': {"type": "float32", "shape": [-1]}
        }
        schema_id = writer.add_schema(vqa_json, "vqa_schema")
        # 将数据集中的内容转化成mindrecord文件并储存
        for i in range(len(self.dataset)):
            image, question, answers, weights = self.dataset[i]
            data = [{'image': image, 'question': question, 'answers': answers, 'weights': weights}]
            writer.write_raw_data(data)
        writer.commit()
        print("------------------SUCCESSFULLY GENERATED!!!------------------")
