
"""
    @author Roger

    This class implements a 2D FCN for the task of segmentation in CT data

"""

from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from utils import *
from loss_functions import *
from scipy.misc import imsave
import collections
import datetime

import Queue, threading

class seg_GAN(object):
    def __init__(self, sess, batch_size=10, height=512,width=512, wd=0.0005, checkpoint_dir=None, path_patients_h5=None, learning_rate=2e-8,lr_step=30000,
                 lam_dice=1, lam_fcn=1, lam_adv=1,adversarial=False):


        self.sess = sess
        self.adversarial=adversarial
        self.lam_dice=lam_dice
        self.lam_fcn=lam_fcn
        self.lam_adv=lam_adv
        self.lr_step=lr_step
        self.wd=wd
        self.learning_rate=learning_rate
        self.batch_size=batch_size       
        self.height=height
        self.width=width
        self.checkpoint_dir = checkpoint_dir
        self.data_queue = Queue.Queue(100) # a queue with two space for 20 "chunks"
        self.path_patients_h5=path_patients_h5
        #self.data_generator = Generator_2D_slices_h5(path_patients_h5,self.batch_size)
        self.build_model()

    def build_model(self):
