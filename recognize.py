import os
from PIL import Image
import numpy as np
import torch
from cv2 import *
from unet import UNet
import time
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


net = UNet(n_channels=3, n_classes=20, bilinear=True)
net.load_state_dict(torch.load('checkpoints\CP_epoch70.pth'))
net.to(device=device)

path = 'E:\\Data\\Cityscape\\leftImg8bit\\test\\munich'
for f in os.listdir(path):
    start = time.time()

    img = imread(os.path.join(path,f))
    img = resize(img, (512, 256)).transpose([2, 0, 1])
    img = torch.from_numpy(img).unsqueeze(0).to(device=device, dtype=torch.float32)

    output = net(img)

    # imshow("img", img.cpu().detach().numpy().astype(np.uint8)[0].transpose([1, 2, 0]))
    # waitKey(10)
    # mask_pred要经过sigmoid归一化
    img = img.cpu().detach().numpy().astype(np.uint8)[0].transpose([1, 2, 0])
    labelMap = np.argmax(torch.sigmoid(output).cpu().detach().numpy().astype(np.float32)[0], axis=0)

    plt.figure(0)
    plt.imshow(img)

    pedestrian = ((labelMap == 24)*255).astype(np.uint8)
    personFlag = np.sum(pedestrian)

    ret, pedestrian = threshold(pedestrian, 128, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(pedestrian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bbox = []
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        # 剔除太小的bbox
        if w*h>0.001*256*512:
            bbox.append([x, y, w, h])

            cv2.rectangle(img, (x, y), (x + w, y + h), (255,255,255), 1)
            cv2.rectangle(labelMap.astype(np.uint8), (x, y), (x + w, y + h), 255, 1)

    if personFlag:
        print('检测到行人')
    else:
        print('未检测到行人')

    end = time.time()
    print('{}FPS'.format(1 / (end - start)))

    plt.figure(1)

    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)
    plt.imshow(labelMap)

    plt.show()

