import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from unet import UNet
from skimage.io import imread
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def infer_image(checkpoint_saved, image_path):

    tr = transforms.ToTensor()
    image = imread(image_path)
    input = tr(image).unsqueeze(0)

    segmenter = UNet(3,1)
    segmenter.eval()
    segmenter.load_state_dict(torch.load(checkpoint_saved))

    output = torch.sigmoid(segmenter(input)).squeeze(0).permute(1,2,0).detach().numpy()
    # print(output.size())
    # print(output)
    # plt.imshow(output, cmap="gray_r")
    # plt.show()
    
    return output