import os
import torch
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from encoder_decoder import decoder



tr = transforms.ToTensor()

class build_set_for_training(Dataset):

    def __init__(self, train_data_names, train_data_path, excel_seg_images, orig_height = 768, orig_width = 768):
        
        self.train_data_names = train_data_names
        self.train_data_path = train_data_path
        self.excel_seg_images = excel_seg_images
        self.orig_height = orig_height
        self.orig_width = orig_width
        
    def __len__(self):
        return len(self.train_data_names)

    def __getitem__(self, idx):
        fp = self.train_data_names[idx]
        path=os.path.join(self.train_data_path, fp)
        image = imread(path)
        annotations = self.excel_seg_images.query('ImageId=="' + fp + '"')['EncodedPixels']
        count = len(annotations)
        mask = np.zeros((self.orig_height, self.orig_height), dtype=np.uint8)
        for i, a in enumerate(annotations):
            decoded = decoder(a)
            mask = mask + decoded
        return tr(image), tr(mask)



class build_set_for_testing(Dataset):

    def __init__(self, data_names, data_path, data_classes):
        
        self.data_names = data_names
        self.data_path = data_path
        self.data_classes = data_classes
        
    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        fp = self.data_names[idx]
        path=os.path.join(self.data_path, fp)
        image = imread(path)
        image_class = self.data_classes[idx]
        return tr(image), tr(image_class)

