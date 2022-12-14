import random
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from data import test as T

def create_data_loader(batch_size, train_val=True, test=True,
                       data_root="../data/exp7500_256x192"):

    if train_val:
        train_list = [
            1,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29,31,32,34,35,37,38,40,41,
            43,44,46,48,50,51,53,54,56,57,59,60,62,63,65,66,71,72
        ]

        random.shuffle(train_list)
        train_list, val_list = train_list[:-11], train_list[-11:]

        train_dl = T.create_data_loader("../data/data_list.csv", image_list=train_list, data_root=data_root, rotate=False)
        val_dl = T.create_data_loader("../data/data_list.csv", image_list=val_list, data_root=data_root)
    else:
        train_dl, val_dl = None, None

    if test:
        test_list = [
            6,9,12,15,18,21,24,27,30,33,36,39,42,45,49,52,55,58,61,64,67,73
        ]

        test_dl = T.create_data_loader("../data/data_list.csv", image_list=test_list, data_root=data_root)
    else:
        test_dl = None

    return train_dl, val_dl, test_dl
