#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LabelSmooth_CE_weight(nn.Module):
    """revised from MultiLabelLongTailed/losses/loss_DB_smooth.py cross_entropy"""
    def __init__(self, cls_num_list, labelsmoothing_alpha=0.05, E1=20, E2=50, E=100):
        super(LabelSmooth_CE_weight, self).__init__()

        self.cls_num_list = cls_num_list
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)

        # weight of each class for imbalance dataset
        weight = torch.cuda.FloatTensor(1.0 / cls_num_list)
        self.weight = (weight / weight.sum()) * len(cls_num_list)
        self.labelsmoothing_alpha = labelsmoothing_alpha
        #hyper-parameters of stages
        self.E1 = E1
        self.E2 = E2
        self.E = E

    def asymmetric_cross_entropy(self, x, target, weight=None, reduction='mean'):
        num_classes = x.size(1)
        smooth_target = torch.zeros_like(x).scatter_(1, target.unsqueeze(1),1)
        if self.labelsmoothing_alpha>0:
            for i in range(num_classes):
                smooth_target[:,i] = smooth_target[:,i] * (1-(self.labelsmoothing_alpha+0.01)) + self.labelsmoothing_alpha
        log_prob = F.log_softmax(x, dim=1)
        loss = -torch.sum(log_prob*smooth_target, dim=1)
        
        if weight is not None:
            weight = weight[target]
            loss = loss * weight
        assert reduction in (None, 'none', 'mean', 'sum')
        if reduction=='mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss

    def forward(self, x, target, e, f1_score=[1,1]):
        '''
        :param x: input, also pred
        :param target: label
        :param e: current epoch
        :param f1_score: f1 score on validation set
        :return: loss
        '''
        if e <= self.E1:
            return self.asymmetric_cross_entropy(x, target)
        elif e > self.E1 and e <= self.E2:
            now_power = (e-self.E1) / (self.E2-self.E1)
            per_cls_weights = [torch.pow(num, now_power) for num in self.weight]
            per_cls_weights = torch.cuda.FloatTensor(per_cls_weights)
            return self.asymmetric_cross_entropy(x, target, weight=per_cls_weights)
        else:
            f1_score = torch.cuda.FloatTensor(f1_score)
            weight = torch.cuda.FloatTensor(1.0 / f1_score)
            self.weight = (weight / weight.sum()) * len(self.cls_num_list)
            now_power = (e - self.E2) / (self.E - self.E2)
            per_cls_weights = [torch.pow(num, now_power) for num in self.weight]
            per_cls_weights = torch.cuda.FloatTensor(per_cls_weights)
            return self.asymmetric_cross_entropy(x, target, weight=per_cls_weights)

def function_test():
    torch.manual_seed(42)
    batch_size = 8
    num_cls = 2
    x = torch.randn(batch_size,num_cls)
    label = torch.randint(0, 2, (batch_size,))
    smoothing=0.05
    x = x.to("cuda:0")
    label = label.to("cuda:0")
    criterion = LabelSmooth_CE_weight(cls_num_list=[3,5],labelsmoothing_alpha=smoothing, E1=1, E2=2, E=3)
    
    loss_0 = criterion(x, label, e=0)
    print('loss_0:==>', loss_0)

    loss_1 = criterion(x, label, e=1)
    print('loss_1:==>', loss_1)

    loss_2 = criterion(x, label, e=2)
    print('loss_2:==>', loss_2)

    loss_3 = criterion(x, label, e=3)
    print('loss_3:==>', loss_3)


if __name__ == '__main__':
    function_test()
