import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from unet import UNet
from loss import dice_bce
from loss import dice_loss
import torch.optim as optim
from data import build_set_for_training
from torch.utils.data import DataLoader as DataLoader

csv = pd.read_csv('C:\\Users\\LouisDorge\\OneDrive - MESASIGHT GmbH\\Desktop\\toolings\\train_ship_segmentations_v2.csv')
data_path = 'C:\\Users\\LouisDorge\\OneDrive - MESASIGHT GmbH\\Desktop\\toolings\\train_data'
checkpoint_saved = 'C:\\Users\\LouisDorge\\OneDrive - MESASIGHT GmbH\\Desktop\\toolings\\checkpoint_v6.pth'
checkpoint_to_save = 'C:\\Users\\LouisDorge\\OneDrive - MESASIGHT GmbH\\Desktop\\toolings\\checkpoint_v1.pth'


def launch(checkpoint_to_save=checkpoint_to_save, checkpoint_saved=checkpoint_saved):

    epochs = 10
    lr = 0.01
    momentum = 0.9

    ship_names = ['0a0df8299.jpg', '0a1a58833.jpg', '0a1b7d6ec.jpg']

    training_set = build_set_for_training(ship_names, data_path, csv)

    batch_size = 3

    loader = DataLoader(training_set, batch_size=1)

    segmenter = UNet(3,1)

    segmenter.load_state_dict(torch.load(checkpoint_saved))

    optimizer = optim.SGD(segmenter.parameters(), lr=lr, momentum=momentum)

    segmenter.train()

    dic = {'train_loss' : [],
        'val_loss' : [],
        }

    print("THE TRAINING STARTED ...")

    for epoch in range(epochs):

        print('\n')
        print("EPOCH : {} OUT OF {} ".format(epoch+1, epochs))

        train_loss = train_forward(segmenter, loader, optimizer)
        dic['train_loss'].append(train_loss)

    torch.save(segmenter.state_dict(), checkpoint_to_save)

    return dic


def train_forward(segmenter, loader, optimizer):

    running_loss=0
    
    for i, data in enumerate(tqdm(loader)):

        img, seg = data
               
        output = torch.sigmoid(segmenter(img))

        loss = dice_bce(output, seg)
        dice = dice_loss(output, seg)
        bce_loss = torch.nn.BCELoss()
        bce=bce_loss(output, seg.float())
        print('dice', dice)
        print('bce', bce)
        print('total', loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach()
        running_loss += float(loss.item())

    print(running_loss)
    return running_loss


dic = launch()
print(dic)