import argparse
import logging
import os
import sys
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from cv2 import *
from matplotlib import pyplot as plt

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import Dataset
from torch.utils.data import DataLoader, random_split

dir_img = 'E:\Data\Cityscape'
dir_checkpoint = 'checkpoints/'

def Trans_3Dlabel_2RGB(label, classes, is_pre=True):#label.shape = [C,H,W]
    RGB_img = np.zeros([3,label.shape[1],label.shape[2]])
    positins = []
    colors = [[0,0,0],[0,191,255],[46,139,87],[169,169,169],[124,252,0],[0,0,255],[255,0,255],[255,105,180],[255,165,0]]
    if is_pre:
        for i in range(0, classes):
            positins.append(label[i] == 1)
    else:
        for i in range(0, classes):
            positins.append(label[0] == i)
    for i in range(0, classes):
        RGB_img[0][positins[i]] = colors[i][0]#第i个class的位置
        RGB_img[1][positins[i]] = colors[i][1]
        RGB_img[2][positins[i]] = colors[i][2]
    return RGB_img

#TODO 取消bilinear *update:预计无影响，已恢复bilinear
def train_net(net,
              device,
              epochs=70,
              batch_size=2,
              lr=1e-3,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.25):

    dataset = Dataset(dir_img, img_scale, 'train')
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    print("TrainData length: ", len(train_loader), '\n', 'ValData length: ', len(val_loader))

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        criterion = nn.BCEWithLogitsLoss()

    net.train()

    for epoch in range(epochs):

        if epoch>0 and os.path.exists(os.getcwd() + '/checkpoints/CP_epoch' + str(epoch) + '.pth'):
            net.load_state_dict(torch.load('./checkpoints/CP_epoch' + str(epoch) + '.pth'))
            print('\nTrain Parameter ' + str(epoch) + ' successfully loaded!')

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='imgs') as pbar:
            for i, data in enumerate(train_loader):
                imgs = data['image']
                masks = data['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                masks = masks.to(device=device, dtype=mask_type)

                optimizer.zero_grad()

                masks_pred = net(imgs)

                # print('img', imgs.shape)
                # print('mask', masks_pred.shape)
                # imshow("img", imgs.cpu().detach().numpy().astype(np.uint8)[0].transpose([1, 2, 0]))
                # waitKey(30)
                # # mask_pred要经过sigmoid归一化
                # labelMap = np.argmax(torch.sigmoid(masks_pred).cpu().detach().numpy().astype(np.float32)[0], axis=0)
                # plt.imshow(labelMap)
                # plt.show()


                # print("Dice:", Dice(true_mask.cpu().detach().numpy().astype(np.float16), (masks_pred[:,1,:,:]>0.5).cpu().detach().numpy().astype(np.float16)))

                loss = criterion(masks_pred, masks)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                writer.add_scalar('Loss/train', np.mean(loss.cpu().detach().numpy()), global_step)
                pbar.set_postfix(**{'loss (batch)': np.mean(loss.cpu().detach().numpy())})
                pbar.update(batch_size)
                del imgs

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch+1}.pth')
            logging.info(f'Checkpoint {epoch+1} saved !')

        global_step += 1
        #print(global_step)
        if global_step % (len(dataset) // (10 * batch_size)) == 0:
            val_score = eval_net(net, val_loader, device, n_val, epoch)#加载上一个epoch参数

            if net.n_classes > 1:
                logging.info('\nValidation cross entropy: {}'.format(val_score))
                writer.add_scalar('Loss/test', val_score, global_step)

            else:
                logging.info('Validation Dice Coeff: {}'.format(val_score))
                writer.add_scalar('Dice/test', val_score, global_step)

            writer.add_images('images', imgs, global_step)
            writer.add_images('masks', masks*255, global_step)
            writer.add_images('persons', torch.sigmoid(masks_pred[24])>0.5, global_step)
            #writer.add_images('masks/true', Trans_3Dlabel_2RGB(true_masks.cpu().detach().numpy(), 9, is_pre=False), global_step, dataformats='CHW')
            # labels = ['unkown','sky','tree','road','grass','water','building','mountain','foreground']
            # for i in range(0,9):
            #     translabel = torch.sigmoid(masks_pred[0][i]) > 0.5
            #     writer.add_images('masks/pred_'+ labels[i], translabel, 9, global_step, dataformats='HW')


                #print('show mask: ',str(i), show_mask.min(), show_mask.max())
            # show_mask[show_mask <= 0.5] = 0.0
            # masks_pred[0][1][masks_pred[0][1]>0.2]=1


            # if net.n_classes == 1:
            #     writer.add_images('masks/true', masks, global_step)
            #     writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)



    writer.close()

def Dice(y_true, y_pred):
    y_true = y_true
    y_pred = y_pred
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=20, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    net.to(device=device)
    train_net(net,device=device)
