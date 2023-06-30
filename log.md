# 数据集创建思路
在本模块中要实现的是将data文件夹中的内容做处理并转移到内存中
流程为：
1. 读入数据
2. 按照源代码做处理
3. 储存到mindrecord文件中（使用generator文件）
4. 根据mindrecord文件创建dataset

注意数据的版本不同，课程给出的版本为2015
本次代码使用的是2017，即CVPR的数据集

# 模型搭建
目前模型搭建遇到的两个主要困难：
1. ViT的模型没有可用的，需要自己写
2. bert的模型不知道是否可用于mindspore中

对于第二个问题，transformer包中的内容并不能
在mindspore中使用，因此需要重写bert，目前社区有一份bert代码
不知能否使用

from timm.models.layers import trunc_normal_, DropPath中，DropPath可以直接复制代码平替，前面的那个也可直接复制代码平替
from timm.models.vision_transformer import _cfg, PatchEmbed中，_cfg可以复制代码平替，PatchEmbed也可以平替

这样第一个模型的问题就解决了
第二个也已经找到平替
# 待补充内容
1. 训练函数保存预训练权重
2. 测试函数加载模型权重
3. 优化器和学习率可进一步提供更多选项