import pandas as pd
import cv2
from matplotlib import pyplot as plt
from torch.utils import data
from torch.utils.data import DataLoader
from dependency import *
import torch
import numpy as np
from utils import encode_label, encode_meta_label
# from keras.utils import to_categorical
from torch.nn.functional import one_hot
import albumentations
# Build the Pytorch dataloader
from albumentations.pytorch import ToTensorV2
import albumentations as A
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    # RandomContrast,
    RandomGamma,
    # RandomBrightness,
    ShiftScaleRotate,
    RandomBrightnessContrast,
)

# aug = Compose(
#     [
#         VerticalFlip(p=0.5),
#         HorizontalFlip(p=0.5),
#         ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.5,rotate_limit=45,p=0.5),
#         RandomRotate90(p=0.5),
#         RandomBrightnessContrast(p=0.5),
#         #RandomContrast(p=0.5),
#         #RandomBrightness(p=0.5),
#         #RandomGamma(p=0.5)
#
#     ],
#     p=0.5)
# aug = Compose(
#     [
#         VerticalFlip(p=0.5),
#         HorizontalFlip(p=0.5),
#         ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=40, p=0.5),
#         # RandomRotate90(p=0.5),
#         # RandomBrightnessContrast(p=0.5),
#         #RandomContrast(p=0.5),
#         #RandomBrightness(p=0.5),
#         #RandomGamma(p=0.5)
#
#     ],
#     p=0.5)
aug = A.Compose([
    A.RandomRotate90(p=0.5),  # 90°倍数旋转（保护方向不变性）
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),  # 模拟皮肤弹性形变
    # A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.8),  # 病灶中心保留裁剪
    # A.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # ),
],
    additional_targets={'dermoscopy': 'image'},  # 声明皮肤镜为独立模态
    is_check_shapes=False  # 关闭自动校验（需提前确保尺寸一致）
)
clinical_aug = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=0.1, p=0.5),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),  # 增强局部对比度
    A.GaussianBlur(blur_limit=(3, 7), p=0.2)  # 模拟拍摄模糊
])
dermoscopy_aug = A.Compose([
    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.4),  # 模拟设备噪声
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.3),  # 传感器噪声
    A.RandomGamma(gamma_limit=(80, 120), p=0.3)  # 控制透光强度
])


def modality_mixup(img1, img2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    mixed_img = lam * img1 + (1 - lam) * img2
    return mixed_img.astype(np.uint8)


def attention_dropout(img, mask, max_objects=3):
    """
    基于皮肤镜掩码遮挡临床图区域
    模拟真实干扰（汗毛/气泡）[[10]][[15]]
![](https://oss.metaso.cn/metaso/pdf2texts_reading_mode/figures/423e05eb-d608-4fb7-96ae-ac1e9991c644/25_0.jpg)
    """
    # 转换掩码为单通道（若为RGB）
    if mask.ndim == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)  # RGB转灰度
        mask = mask[:, :, np.newaxis]  # 添加通道维度 (224,224,1)

    # 转换图像为三通道（若为灰度）
    if img.ndim == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)  # (224,224,3)

    aug = A.Compose([
        A.MaskDropout(
            max_objects=max_objects,
            p=1.0
        )
    ])
    return aug(image=img, mask=mask)["image"]


def load_image(path, shape):
    img = cv2.imread(path)
    # resize = A.Compose([
    #     A.Resize(shape[0], shape[1])
    # ])
    # img = resize(image=img)["image"]
    img = cv2.resize(img, (shape[0], shape[1]))
    return img


normlazing = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])


class SkinDataset(data.Dataset):
    def __init__(self, image_dir, img_info, file_list, shape, is_test=False, num_class=1):
        self.is_test = is_test
        self.image_dir = image_dir
        self.img_info = img_info
        self.file_list = file_list
        self.shape = shape
        self.num_class = num_class
        self.total_img_info = img_info

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index]
        sub_img_info = self.total_img_info[file_id:file_id + 1]

        # get the clincal image path
        clinic_img_path = sub_img_info['clinic']
        # get the dermoscopy image path
        dermoscopy_img_path = sub_img_info['derm']
        # load the clinical image
        clinic_img = load_image(self.image_dir + clinic_img_path[file_id], self.shape)
        # load the dermoscopy image
        dermoscopy_img = load_image(self.image_dir + dermoscopy_img_path[file_id], self.shape)

        # Encode the diagnositic label
        diagnosis_label = sub_img_info['diagnosis'][file_id]
        for index_label, label in enumerate(label_list):
            if diagnosis_label in label:
                diagnosis_index = torch.tensor(index_label)
                # diagnosis_label_one_hot = to_categorical(diagnosis_index, num_label)
                diagnosis_label_one_hot = one_hot(diagnosis_index, num_label)
            else:
                continue

        if not self.is_test:
            augmented = aug(image=clinic_img, dermoscopy=dermoscopy_img)
            clinic_img = augmented['image']
            dermoscopy_img = augmented['dermoscopy']
            # 模态特异性增强
            clinic_img = clinical_aug(image=clinic_img)["image"]
            dermoscopy_img = dermoscopy_aug(image=dermoscopy_img)["image"]
            # # 跨模态交互增强（随机启用）
            # if np.random.rand() > 0.5:
            #     clinic_img = modality_mixup(clinic_img, dermoscopy_img)
            # if np.random.rand() > 0.3:
            #     clinic_img = attention_dropout(clinic_img, dermoscopy_img)

        total_label = encode_label(sub_img_info, file_id)
        # print(total_label)
        clinic_img = torch.from_numpy(np.transpose(clinic_img, (2, 0, 1)).astype('float32') / 255)
        dermoscopy_img = torch.from_numpy(np.transpose(dermoscopy_img, (2, 0, 1)).astype('float32') / 255)
        # clinic_img = normlazing(image=clinic_img)["image"]
        # dermoscopy_img = normlazing(image=dermoscopy_img)["image"]
        meta_data = encode_meta_label(sub_img_info, file_id)

        return clinic_img, dermoscopy_img, [total_label[0], total_label[1], total_label[2], total_label[3],
                                            total_label[4], total_label[5], total_label[6], total_label[7]]


def demo_test():
    # train,val,test dataset spliting
    test_index_df = pd.read_csv(test_index_path)
    train_index_df = pd.read_csv(train_index_path)
    val_index_df = pd.read_csv(val_index_path)

    train_index_list = list(train_index_df['indexes'])
    val_index_list = list(val_index_df['indexes'])
    test_index_list = list(test_index_df['indexes'])

    train_index_list_1 = train_index_list[0:206]
    train_index_list_2 = train_index_list[206:]

    df = pd.read_csv(img_info_path)

    # Img Information
    index_num = 7
    img_info = df[index_num:index_num + 1]
    clinic_path = img_info['clinic']
    dermoscopy_path = img_info['derm']
    # source_dir = '../release_v0/release_v0/images/'
    clinic_img = cv2.imread(source_dir + clinic_path[index_num])
    dermoscopy_img = cv2.imread(source_dir + dermoscopy_path[index_num])

    plt.subplot(121)
    plt.imshow(dermoscopy_img)

    plt.subplot(122)
    plt.imshow(clinic_img)
    plt.show()


def demo_run():
    test_index_df = pd.read_csv(test_index_path)
    train_index_df = pd.read_csv(train_index_path)
    val_index_df = pd.read_csv(val_index_path)

    train_index_list = list(train_index_df['indexes'])
    val_index_list = list(val_index_df['indexes'])
    test_index_list = list(test_index_df['indexes'])

    df = pd.read_csv(img_info_path)

    shape = (224, 224)
    batch_size = 16
    num_workers = 0
    train_skindataset = SkinDataset(image_dir=source_dir,
                                    img_info=df,
                                    file_list=train_index_list,
                                    shape=shape, is_test=False)

    train_dataloader = DataLoader(
        dataset=train_skindataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True)

    val_skindataset = SkinDataset(image_dir=source_dir,
                                  img_info=df,
                                  file_list=val_index_list,
                                  shape=shape, is_test=True)

    val_dataloader = DataLoader(
        dataset=val_skindataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True)

    for clinic_img, derm_img, label in train_dataloader:
        print(clinic_img.shape, derm_img.shape, label[0].shape)
        print('train_dataloader finished')

    for clinic_img, derm_img, label in val_dataloader:
        print(clinic_img.shape, derm_img.shape, label[0].shape)
        print('val_dataloader finished')


def generate_dataloader(shape, batch_size, num_workers, data_mode):
    test_index_df = pd.read_csv(test_index_path)
    train_index_df = pd.read_csv(train_index_path)
    val_index_df = pd.read_csv(val_index_path)

    train_index_list = list(train_index_df['indexes'])
    val_index_list = list(val_index_df['indexes'])
    test_index_list = list(test_index_df['indexes'])

    train_index_list_1 = train_index_list[0:206]
    train_index_list_2 = train_index_list[206:]

    df = pd.read_csv(img_info_path)
    if data_mode == 'self_evaluated':
        data_mode = 'SP'
        train_skindataset = SkinDataset(image_dir=source_dir,
                                        img_info=df,
                                        file_list=train_index_list_1,
                                        shape=shape, is_test=False,
                                        )
        train_dataloader = DataLoader(
            dataset=train_skindataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True)

        val_skindataset = SkinDataset(image_dir=source_dir,
                                      img_info=df,
                                      file_list=val_index_list,
                                      shape=shape, is_test=True,
                                      )

        val_dataloader = DataLoader(
            dataset=val_skindataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True)


    else:
        train_skindataset = SkinDataset(image_dir=source_dir,
                                        img_info=df,
                                        file_list=train_index_list,
                                        shape=shape,
                                        is_test=False,
                                        )
        train_dataloader = DataLoader(
            dataset=train_skindataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True)

        val_skindataset = SkinDataset(image_dir=source_dir,
                                      img_info=df,
                                      file_list=val_index_list,
                                      shape=shape,
                                      is_test=True,
                                      )

        val_dataloader = DataLoader(
            dataset=val_skindataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True)
    test_skindataset = SkinDataset(image_dir=source_dir,
                                   img_info=df,
                                   file_list=test_index_list,
                                   shape=shape, is_test=True,
                                   )

    test_dataloader = DataLoader(
        dataset=test_skindataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False)
    train_num, val_num, test_num = len(train_skindataset), len(val_skindataset), len(test_skindataset)

    return train_dataloader, val_dataloader, test_dataloader, train_num, val_num, test_num
