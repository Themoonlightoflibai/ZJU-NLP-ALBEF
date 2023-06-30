from __future__ import division
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as CC


def create_vqa_dataset(config, batch_size=1, repeat_num=1, device_num=1, rank_id=0,
                       is_training=True, num_parallel_workers=12, is_tiny=False):
    if is_training:
        # 如果不是精简模式则加载全部的数据文件
        if not is_tiny:
            train_ds = de.MindDataset(config.train_mindrecord, columns_list=['image', \
                                                                             'question', 'answers', 'weights'],
                                      num_shards=device_num, shard_id=rank_id, \
                                      num_parallel_workers=num_parallel_workers, shuffle=is_training)
            val_ds = de.MindDataset(config.val_mindrecord, columns_list=['image', 'question', 'answers', 'weights'],
                                    num_shards=device_num, shard_id=rank_id, \
                                    num_parallel_workers=num_parallel_workers, shuffle=is_training)
            # 将数据集中连续 batch_size 条数据组合为一个批数据
            train_ds = train_ds.batch(batch_size, drop_remainder=False)
            val_ds = val_ds.batch(batch_size, drop_remainder=False)
            # 数据集重复额数量
            train_ds = train_ds.repeat(repeat_num)
            return train_ds, val_ds
        else:
            # 如果使用精简模式则只从数据集中取出5000个样本进行训练
            train_ds = de.MindDataset(config.train_mindrecord, columns_list=['image', \
                                                                             'question', 'answers', 'weights'],
                                      num_shards=device_num, shard_id=rank_id, \
                                      num_parallel_workers=num_parallel_workers, shuffle=is_training, num_samples=5000)
            val_ds = de.MindDataset(config.val_mindrecord, columns_list=['image', 'question', 'answers', 'weights'],
                                    num_shards=device_num, shard_id=rank_id, \
                                    num_parallel_workers=num_parallel_workers, shuffle=is_training, num_samples=5000)
            # 将数据集中连续 batch_size 条数据组合为一个批数据
            train_ds = train_ds.batch(batch_size, drop_remainder=False)
            val_ds = val_ds.batch(batch_size, drop_remainder=False)
            # 数据集重复额数量
            train_ds = train_ds.repeat(repeat_num)
            return train_ds, val_ds

    else:
        if not is_tiny:
            test_ds = de.MindDataset(config.test_mindrecord, columns_list=['image', 'question', 'answers', 'weights'],
                                     num_shards=device_num, shard_id=rank_id, \
                                     num_parallel_workers=num_parallel_workers, shuffle=is_training)
            test_ds = test_ds.batch(batch_size, drop_remainder=False)
            return test_ds
        else:
            test_ds = de.MindDataset(config.test_mindrecord, columns_list=['image', 'question', 'answers', 'weights'],
                                     num_shards=device_num, shard_id=rank_id, \
                                     num_parallel_workers=num_parallel_workers, shuffle=is_training, num_samples=5000)
            test_ds = test_ds.batch(batch_size, drop_remainder=False)
            return test_ds
