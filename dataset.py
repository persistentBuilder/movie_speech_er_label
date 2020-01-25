from __future__ import print_function, division
import os, io
import os.path
from os import path
import torch
import numpy as np
import PIL
from PIL import Image
import yaml
import math
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import cv2
from numpy import load
import pandas as pd
import time
from spectrogram_voxceleb import *
import subprocess
from colored_spectogram import graph_spectrogram

class VideoLoad(Dataset):

    def __init__(self, path_file, train_flag=True, transform=None):

        f = open(path_file, "r")
        self.all_lines = f.readlines()
        self.transform = transform
        self.train_flag = train_flag
        self.imgs = []
        self.gt = []
        self.width = 224
        self.height = 224
        self.expression_map = {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "sadness": 4}
        self.num_classes = 5
        self.use_visual_label = False
        self.use_text_label = True
        self.use_voxceleb_spectogram = True
        cnt = [0, 0, 0, 0 ,0]
        for i,line in enumerate(self.all_lines):
            line = line[:-1]
            try: 
                ground_truth = self.get_gt_from_line(line)
            except:
                continue
            if cnt[ground_truth] > 50:
                continue
            if (i < int(len(self.all_lines)*0.9) and train_flag) or (i >= int(len(self.all_lines)*0.9) and not train_flag):
                self.imgs.append(self.get_spectogram_for_video(line))
                self.gt.append(ground_truth)
                cnt[ground_truth] = cnt[ground_truth] + 1

    def get_wav_file_path(self, video_path):
        wav_path = video_path.rsplit('.',1)[0] + ".wav"
        if not os.path.isfile(wav_path):
            subprocess.call(['ffmpeg', '-i', video_path, '-codec:a', 'pcm_s16le', '-ac', '1', wav_path])            
        return wav_path

    def gt_distribution(self):
        inv_map = {v: k for k, v in self.expression_map.items()}
        for i in range(0, self.num_classes):
            print(inv_map[i],  self.gt.count(i))

    def get_spectogram_for_video(self, line):
        video_path = line.rsplit(",",1)[0]
        wave_file_path = self.get_wav_file_path(video_path)
        if self.use_voxceleb_spectogram:
            spec_img = get_spectrum(wave_file_path)
            spec_img = spec_img[-10:,:]
        else:
            spec_img_path = wave_file_path.split('.')[0]+'.png'
            if path.exists(spec_img_path):
                spec_img = cv2.imread(spec_img_path)
            else:
                spec_img = cv2.imread(graph_spectrogram(wave_file_path))
        return cv2.resize(spec_img, dsize=(self.height, self.width), interpolation=cv2.INTER_CUBIC)

    def compute_visual_label(self, line):
        if line.split("_")[-2] == 'visual-label':
            return line.split("_")[-1]    
        

    def get_gt_from_line(self, line):
        if self.use_text_label and not self.use_visual_label:
            return self.expression_map[line.split(",")[1]]
        elif not self.use_text_label and self.use_visual_label:
            return self.expression_map[self.compute_visual_label(line)]
        elif self.use_text_label and self.use_visual_label:
            if self.expression_map[line.split(",")[1]] == self.expression_map[self.compute_visual_label(line)]:
                return self.expression_map[self.compute_visual_label(line)]
            else:
                raise("error")
                 
    def __getitem__(self, index):
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.gt[index]

    def __len__(self):
        return len(self.imgs)

    def to_categorical(self, y):
        return np.eye(self.num_classes, dtype='int')[y]

#print(vd.__getitem__(0))
