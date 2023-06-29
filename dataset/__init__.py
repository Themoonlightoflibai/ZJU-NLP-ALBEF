from dataset.vqa_dataset import VQA_Dataset
from dataset.randaugment import RandomAugment
from mindspore.dataset import vision
from PIL import Image

def create_dataset(dataset, config):

    normalize = vision.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    pretrain_transform = vision.Compose([
        vision.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0), interpolation=Image.BICUBIC),
        vision.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        # 注意mindspore在使用totensor时会改变数据的维度，需要转换回原来的的维度
        vision.ToTensor(),
        normalize,
    ])

    train_transform = vision.Compose([
        vision.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        vision.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        vision.ToTensor(),
        normalize,
    ])

    test_transform = vision.Compose([
        vision.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        vision.ToTensor(),
        normalize,
    ])

    if dataset == 'vqa':
        train_dataset = vqa_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'], split='train')
        # val_dataset = vqa_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'], split='train')
        vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'], split='test', answer_list=config['answer_list'])
        return train_dataset, vqa_test_dataset

    return None
