import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np
import json
import argparse
import csv
from model import AVENet
from datasets import GetAudioVideoDataset
from utils import AverageMeter,  accuracy 
import cv2
from sklearn.metrics import auc
import xml.etree.ElementTree as ET
import pdb
from PIL import Image
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='',type=str,help='Root directory path of data')
    parser.add_argument('--image_size',default=224,type=int,help='Height and width of inputs')
    parser.add_argument('--gt_path',default='',type=str)
    parser.add_argument('--summaries_dir',default='',type=str,help='Model path')
    parser.add_argument('--test',default='',type=str,help='test csv files')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument('--epsilon', default=0.65, type=float, help='pos')
    parser.add_argument('--epsilon2', default=0.4, type=float, help='neg')
    parser.add_argument('--tri_map',action='store_true')
    parser.set_defaults(tri_map=True)
    parser.add_argument('--Neg',action='store_true')
    parser.set_defaults(Neg=True)

    return parser.parse_args() 

class Evaluator():

    def __init__(self):
        super(Evaluator, self).__init__()
        self.ciou = []

    def cal_CIOU(self, infer, gtmap, thres=0.01):
        #  pdb.set_trace()
        #  gtmap = np.reshape(gtmap,[256, 256])
        #  infer = np.reshape(infer,[256, 256])
        infer_map = np.zeros((224, 224))
        #  infer_map = np.zeros((20,20))
        infer_map[infer>=thres] = 1
        ciou = np.sum(infer_map*gtmap) / (np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))
        self.ciou.append(ciou)
        return ciou, np.sum(infer_map*gtmap),(np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))


    def cal_AUC(self):
        results = []
        for i in range(21):
            result = np.sum(np.array(self.ciou)>=0.05*i)
            result = result / len(self.ciou)
            results.append(result)
        x = [0.05*i for i in range(21)]
        auc = sklearn.metrics.auc(x, results)
        print(results)
        return auc

    def final(self):
        ciou = np.mean(np.array(self.ciou)>=0.5)
        return ciou

    def clear(self):
        self.ciou = []


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    return value

def main():
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    model= AVENet(args) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model = model.cuda()
    
    print('load pretrained model.')
    checkpoint = torch.load(args.summaries_dir)

    model_dict = model.state_dict()
    pretrained_dict = checkpoint['model_state_dict']
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.to(device)
    testdataset = GetAudioVideoDataset(args,  mode='test')
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False,num_workers = 16)

    softmax = nn.Softmax(dim=1)
    print("Loaded dataloader.")

    model.eval()
    accuracies = AverageMeter()
    accuracies5 = AverageMeter()
    iou = []
    for step, (image, spec, audio,name,im) in enumerate(testdataloader):
        print('%d / %d' % (step,len(testdataloader) - 1))
        spec = Variable(spec).cuda()
        image = Variable(image).cuda()
        heatmap,_,Pos,Neg = model(image.float(),spec.float(),args)
        im_arr = im.data.cpu().numpy()
        heatmap_arr =  heatmap.data.cpu().numpy()
        Pos = Pos.data.cpu().numpy()
        Neg = Neg.data.cpu().numpy()
        audio = audio.numpy()

        for i in range(spec.shape[0]):
            heatmap_now = cv2.resize(heatmap_arr[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            heatmap_now = normalize_img(-heatmap_now)
            gt = ET.parse(args.gt_path + '%s.xml' % name[i][:-4]).getroot()
            gt_map = np.zeros([224,224])
            bboxs = []
            for child in gt: 
                for childs in child:
                    bbox = []
                    if childs.tag == 'bbox':
                        for index,ch in enumerate(childs):
                            if index == 0:
                                continue
                            bbox.append(int(224 * int(ch.text)/256))
                    bboxs.append(bbox)

            for item_ in bboxs:
                temp = np.zeros([224,224])
                (xmin,ymin,xmax,ymax) = item_[0],item_[1],item_[2],item_[3]
                temp[item_[1]:item_[3],item_[0]:item_[2]] = 1
                gt_map += temp
            gt_map /= 2
            gt_map[gt_map>1] = 1
            pred =  heatmap_now
            pred = 1 - pred
            threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
            pred[pred>threshold]  = 1
            pred[pred<1] = 0
            evaluator = Evaluator()
            ciou,inter,union = evaluator.cal_CIOU(pred,gt_map,0.5)
            iou.append(ciou)
    results = []
    for i in range(21):
        result = np.sum(np.array(iou) >= 0.05 * i)
        result = result / len(iou)
        results.append(result)
    x = [0.05 * i for i in range(21)]
    auc_ = auc(x, results)
    print('cIoU' , np.sum(np.array(iou) >= 0.5)/len(iou))
    print('auc',auc_)


if __name__ == "__main__":
    main()

