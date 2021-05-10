import os
import cv2
import json
import torch
import csv
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pdb
import time
from PIL import Image
import glob
import sys
#  sys.path.append('./torchvggish/')
#  from torchvggish import vggish_input 
import scipy.io.wavfile as wav
from scipy import signal
import random
import soundfile as sf



class GetAudioVideoDataset(Dataset):

    def __init__(self, args, mode='train', transforms=None):
 
        data = []
        with open(args.test) as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                data.append(item[0] + '.mp4')
        self.audio_path = args.data_path + 'audio/'
        self.video_path = args.data_path + 'frames/'

        self.imgSize = args.image_size 

        self.mode = mode
        self.transforms = transforms
        # initialize video transform
        self._init_atransform()
        self._init_transform()
        #  Retrieve list of audio and video files
        self.video_files = []
   
        for item in data[:]:
            self.video_files.append(item )
        print(len(self.video_files))
        self.count = 0

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize, Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])            

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])
#  
    def _load_frames(self, paths):
        frames = []
        for path in paths:
            frames.append(self._load_frame(path))
        frames = self.vid_transform(frames)
        return frames

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def __len__(self):
        # Consider all positive and negative examples
        return len(self.video_files)  # self.length

    def __getitem__(self, idx):
        file = self.video_files[idx]

        # Image
        frame = self.img_transform(self._load_frame(self.video_path + file[:-3] + 'jpg'))
        frame_ori = np.array(self._load_frame(self.video_path  + file[:-3] + 'jpg'))
        # Audio
        samples, samplerate = sf.read(self.audio_path + file[:-3]+'wav')

        # repeat if audio is too short
        if samples.shape[0] < samplerate * 20:
            n = int(samplerate * 20 / samples.shape[0]) + 1
            samples = np.tile(samples, n)
        resamples = samples[:samplerate*20]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples,samplerate, nperseg=512,noverlap=274)
        spectrogram = np.log(spectrogram+ 1e-7)
        spectrogram = self.aid_transform(spectrogram)
 

        return frame,spectrogram,resamples,file,torch.tensor(frame_ori)



