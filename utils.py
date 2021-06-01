import csv
import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
from scipy import stats
from sklearn import metrics
import numpy as np
import xml.etree.ElementTree as ET


class Evaluator():

    def __init__(self):
        super(Evaluator, self).__init__()
        self.ciou = []

    def cal_CIOU(self, infer, gtmap, thres=0.01):
        infer_map = np.zeros((224, 224))
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

def testset_gt(args,name):

    if args.testset == 'flickr':
        gt = ET.parse(args.gt_path + '%s.xml' % name[:-4]).getroot()
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
        
    elif args.testset == 'vggss':
        gt = args.gt_all[name[:-4]]
        gt_map = np.zeros([224,224])
        for item_ in gt:
            item_ =  list(map(lambda x: int(224* max(x,0)), item_) )
            temp = np.zeros([224,224])
            (xmin,ymin,xmax,ymax) = item_[0],item_[1],item_[2],item_[3]
            temp[ymin:ymax,xmin:xmax] = 1
            gt_map += temp
        gt_map[gt_map>0] = 1
    return gt_map

