from os.path import splitext, exists, join
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from torchvision import transforms as T
from random import randint
import scipy.misc
from utils.dataset import Dataset
from torch.utils.data import DataLoader, random_split
import SimpleITK as sitk
from cv2 import *

transform = T.Compose([T.ToTensor(),
                      T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        # self.scale = scale
        # assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod

    def preprocess(cls, img, mask, randomcrop_size):
        mask = np.array(mask)
        img = transform(img)
        assert randomcrop_size[0] < mask.shape[0] and randomcrop_size[1] < mask.shape[1], print('Crop size is too large',randomcrop_size,img.shape,mask.shape)
        assert img.shape[1:] == mask.shape, print('img and mask doesnt match', img.shape, mask.shape)
        x0 = randint(0, mask.shape[0]-randomcrop_size[0])
        y0 = randint(0, mask.shape[1]-randomcrop_size[1])
        # img_crop = np.zeros(randomcrop_size.append(img.shape[-1]))
        # mask_crop = np.zeros(randomcrop_size)
        img_crop = img[:, x0:x0+randomcrop_size[0], y0:y0+randomcrop_size[1]]
        # img_crop = np.expand_dims(img_crop, axis=1)
        mask_crop = mask[x0:x0+randomcrop_size[0], y0:y0+randomcrop_size[1]]

        # print(np.array(img_crop).shape,mask_crop.shape)

        return np.array(img_crop), np.array(mask_crop)

    # def preprocess(cls, img, scale, mask=False):
    #     #TODO 输入img归一化 *update：已在__getitem__中归一化
    #     img = transform(img)
    #     # img.show()
    #     img_nd = np.array(img)
    #     # if len(img_nd.shape) == 2:
    #     #     img_nd = np.expand_dims(img_nd, axis=2)
    #     #
    #     # # HWC to CHW
    #     # img_trans = img_nd.transpose((2, 0, 1))
    #     # if img_nd.max() > 1:
    #     #     img_nd = img_nd / 255
    #
    #     return img_nd

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        npmask = np.loadtxt(mask_file[0])+1
        # npmask[npmask<5]=255.0
        mask = Image.fromarray(npmask)#cause unkown objects were labled -1
        img = Image.open(img_file[0])
        # print('\nbefore transform', (np.loadtxt(mask_file[0])+1).min(), (np.loadtxt(mask_file[0])+1).max())
        # print('img size: ', img.shape)
        # print('mask size: ', mask.shape)

        #assert img.size == mask.size, \
        #   f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img, mask = self.preprocess(img, mask, [120,180])
        print('\nelements in mask', np.unique(mask))
        #print('after transform', (torch.from_numpy(img)).min(), (torch.from_numpy(img)).max())
        # print(torch.from_numpy(img).shape,torch.from_numpy(mask).shape)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

class Dataset(Dataset):
    def __init__(self, imgs_dir, scale, option='train'):

        self.imgs_dir = imgs_dir
        self.imgIds = []
        self.maskIds = []
        self.scale = scale

        if option == 'train':
            imgsPath = os.path.join(imgs_dir, 'leftImg8bit', 'train')
            masksPath = os.path.join(imgs_dir, 'gtFine_trainvaltest', 'gtFine', 'train')

        elif option == 'test':
            imgsPath = os.path.join(imgs_dir, 'leftImg8bit', 'test')
            masksPath = os.path.join(imgs_dir, 'gtFine_trainvaltest', 'gtFine', 'test')

        else:
            print('option输入应为train或test')

        for city in os.listdir(imgsPath):
            imgCity = os.path.join(imgsPath, city)
            for img in sorted(os.listdir(imgCity)):
                self.imgIds.append(os.path.join(imgCity, img))

        for city in os.listdir(masksPath):
            maskCity = os.path.join(masksPath, city)
            for mask in sorted(os.listdir(maskCity)):
                if 'labelIds' in mask:
                    self.maskIds.append(os.path.join(maskCity, mask))

        assert len(self.imgIds) == len(self.maskIds)
        # print(self.imgIds)
        # print(self.maskIds)
        # print(len(self.imgIds))
        logging.info(f'Creating dataset with {len(self.imgIds)} examples')

    def __len__(self):
        return len(self.imgIds)

    def IdTrans2TrainID(self, mask):
        index255 = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30]
        index255mask = np.zeros(mask.shape, dtype=int)
        for index in index255:
            tmpMask = mask == index
            index255mask = index255mask|tmpMask

        # 先赋值255，进行下面的循环赋值
        mask[index255mask] = 255

        # 其他按大小依次赋值0，1，2...
        newIndex = 0
        for i in range(7,34):
            tmpMask = mask==i
            if np.sum(tmpMask)!=0:
                mask[tmpMask] = newIndex
                newIndex += 1

        # 把255改成19,作为背景
        mask[mask==255] = 19

        return mask
    #@classmethod
    # def normalize(self, matrix_):
    #     minval = matrix_.min()
    #     tmpImage = (matrix_ - minval).astype('float')
    #     if tmpImage.max() != 0:
    #         rescaledImage = ((tmpImage / tmpImage.max()) * 255.0).astype(np.float32)  # 归一化
    #     else:
    #         rescaledImage = tmpImage.astype(np.float32)
    #
    #     #print(rescaledImage.dtype)
    #     return rescaledImage

    def __getitem__(self, i):
        img = scipy.misc.imread(self.imgIds[i])
        mask = scipy.misc.imread(self.maskIds[i], mode='L')

        # print(img.shape, mask.shape)
        # print((img.shape[0], int(img.shape[1]*self.scale), int(img.shape[2]*self.scale)), (int(mask.shape[0]*self.scale), int(mask.shape[1]*self.scale)))

        resizeImgShape = (int(img.shape[1]*self.scale), int(img.shape[0]*self.scale))
        resizeMaskShape = (int(mask.shape[1]*self.scale), int(mask.shape[0]*self.scale))
        img = cv2.resize(img, resizeImgShape).transpose([2, 0, 1])
        mask = cv2.resize(mask, resizeMaskShape)
        mask = self.IdTrans2TrainID(mask)
        # print(np.unique(mask))

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
#
# dataset = Dataset('E:\Data\Cityscape', 0.5)
# n_val = int(len(dataset) * 0.1)
# n_train = len(dataset) - n_val
# train, val = random_split(dataset, [n_train, n_val])
# train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
# print(len(train_loader))
# print(len(dataset))
# for i, data in enumerate(train_loader):
#     print(i)