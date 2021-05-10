import torch
from torch import nn
import torch.nn.functional as F
from models import audio_convnet
from models import image_convnet
from models import base_models
import pdb
import random

def normalize_img(value, vmax=None, vmin=None):
    #  pdb.set_trace()
    value1 = value.view(value.size(0), -1)
    value1 -= value1.min(1, keepdim=True)[0]
    value1 /= value1.max(1, keepdim=True)[0]
    return value1.view(value.size(0), value.size(1), value.size(2), value.size(3))

class AVENet(nn.Module):

    def __init__(self, args):
        super(AVENet, self).__init__()

        # -----------------------------------------------
        self.imgnet = base_models.resnet18(modal='vision', pretrained=True)
        self.audnet = base_models.resnet18(modal='audio')
        self.m = nn.Sigmoid()
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))

        self.epsilon = args.epsilon
        self.epsilon2 = args.epsilon2
        self.tau = 0.03
        self.trimap = args.tri_map
        self.Neg = args.Neg
        self.random_threshold = args.random_threshold

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, image, audio,args,mode='val'):
        # Image
        B = image.shape[0]
        self.mask = ( 1 -100 * torch.eye(B,B)).cuda()
        img = self.imgnet(image)
        img =  nn.functional.normalize(img, dim=1)

        # Audio
        aud = self.audnet(audio)
        aud = self.avgpool(aud).view(B,-1)
        aud = nn.functional.normalize(aud, dim=1)
        # Join them
        A = torch.einsum('ncqa,nchw->nqa', [img, aud.unsqueeze(2).unsqueeze(3)]).unsqueeze(1)
        A0 = torch.einsum('ncqa,ckhw->nkqa', [img, aud.T.unsqueeze(2).unsqueeze(3)])

        # trimap
        Pos = self.m((A - self.epsilon)/self.tau) 
        if self.trimap:    
            Pos2 = self.m((A - self.epsilon2)/self.tau) 
            Neg = 1 - Pos2
        else:
            Neg = 1 - Pos

        Pos_all =  self.m((A0 - self.epsilon)/self.tau) 

        # positive
        sim1 = (Pos * A).view(*A.shape[:2],-1).sum(-1) / (Pos.view(*Pos.shape[:2],-1).sum(-1))
        #negative
        sim = ((Pos_all * A0).view(*A0.shape[:2],-1).sum(-1) / Pos_all.view(*Pos_all.shape[:2],-1).sum(-1) )* self.mask
        sim2 = (Neg * A).view(*A.shape[:2],-1).sum(-1) / Neg.view(*Neg.shape[:2],-1).sum(-1)

        if self.Neg:
            logits = torch.cat((sim1,sim,sim2),1)/0.07
        else:
            logits = torch.cat((sim1,sim),1)/0.07
        
        return A,logits,Pos,Neg
