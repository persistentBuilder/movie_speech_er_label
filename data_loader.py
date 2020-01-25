import torch
import os
from PIL import Image
import random
import numpy as np
import pickle
import torchvision.transforms as transforms
from dataset import VideoLoad

class VIDEO(object):
    def __init__(self, batch_size, use_gpu, num_workers, include_neutral=False, transform=None, path=None, movie="21_jump_street"):
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

        pin_memory = True if use_gpu else False

        path_file = '/home/aryaman.g/projects/sentiment-discovery/expression_for_video/expression_video_' + movie + '.txt'
        trainset = VideoLoad(path_file=path_file, train_flag=True, transform=transform)
        testset = VideoLoad(path_file=path_file, train_flag=False, transform=transform)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )


        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = trainset.num_classes

__factory = {
    'video': VIDEO,
}


def create(name, batch_size, use_gpu, num_workers, include_neutral=False, transform=None, movie="21_jump_street"):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](batch_size, use_gpu, num_workers, include_neutral=include_neutral, transform=transform, movie=movie)
