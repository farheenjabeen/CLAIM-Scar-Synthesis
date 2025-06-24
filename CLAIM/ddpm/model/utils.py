#!/usr/bin/env python3
# encoding: utf-8
import os
import random
import torch
import torch.backends.cudnn as cudnn
import warnings
import numpy as np
import math
import logging
import time
import nibabel as nib
import matplotlib.pyplot as plt



class LR_Scheduler(object):
    def __init__(self, base_lr, num_epochs, warmup=100, mode='poly'):
        self.mode = mode
        self.lr = base_lr
        self.num_epochs = num_epochs
        self.warmup = warmup

    def __call__(self, optimizer, epoch):
        if self.mode == 'poly':
            now_lr = round(self.lr * np.power(1 - np.float32(epoch)/np.float32(self.num_epochs), 0.9), 8)
        elif self.mode == 'warmup':
            if epoch < self.warmup*2:
                now_lr = round(0.5 * self.lr * (1.0 + math.cos(((np.float32(epoch)/np.float32(self.warmup))*math.pi))),8)
            else:
                now_lr = round(self.lr * np.power(1 - (np.float32(epoch) - np.float32(self.warmup*2))/(np.float32(self.num_epochs)-np.float32(self.warmup*2)), 0.9), 8)
        elif self.mode == 'cousinewarmup':
            if self.warmup == 0:
                if epoch < 100:
                    now_lr = round(self.lr * (math.sin(((np.float32(epoch))/(np.float32(100.0 * 2.0)))*math.pi)),8)
                else:
                    now_lr = round(0.5 * self.lr * (1.0 + math.cos(((np.float32(epoch) - np.float32(100.0))/(np.float32(self.num_epochs)-np.float32(100.0)))*math.pi)), 8)
            else:
                if epoch < self.warmup*2:
                    now_lr = round(0.5 * self.lr * (1.0 + math.cos(((np.float32(epoch)/np.float32(self.warmup))*math.pi))),8)
                else:
                    now_lr = round(0.5 * self.lr * (1.0 + math.cos(((np.float32(epoch) - np.float32(self.warmup*2))/(np.float32(self.num_epochs)-np.float32(self.warmup*2)))*math.pi)), 8)
        elif self.mode == 'warmuppoly':
            if epoch < 50:
                now_lr = round(self.lr * (((np.float32(epoch+1.0))/(np.float32(50.0)))),8)
            else:
                now_lr = round(self.lr * np.power(1 - (np.float32(epoch) - np.float32(50.0))/(np.float32(self.num_epochs+1)-np.float32(50.0)), 0.9), 8)
        self._adjust_learning_rate(optimizer, now_lr)
        return now_lr

    def _adjust_learning_rate(self, optimizer, lr):
        optimizer.param_groups[0]['lr'] = lr
