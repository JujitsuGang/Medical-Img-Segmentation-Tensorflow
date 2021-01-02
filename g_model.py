
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

        self.classweights=tf.transpose(tf.constant([[1.0,1.0,1.0,1.0,1.0]],dtype=tf.float32,name='classweights'))
        self.num_classes=5

        self.inputCT=tf.placeholder(tf.float32, shape=[None, self.height, self.width, 1])#5 chans input
        #print 'inputCT shape ', self.inputCT.get_shape()
        self.CT_GT=tf.placeholder(tf.int32, shape=[None, self.height, self.width])
        batch_size_tf = tf.shape(self.inputCT)[0]  #variable batchsize so we can test here
        self.train_phase = tf.placeholder(tf.bool, name='phase_train')
        self.G, self.layer = self.generator(self.inputCT,batch_size_tf)
        print 'G shape ',self.G.get_shape
        self.prediction=tf.argmax(self.G,3)#preds by the generator
        t_vars = tf.trainable_variables()

        if self.adversarial:
            self.probs_G=tf.nn.softmax(self.G)
            self.GT_1hot=tf.one_hot(self.CT_GT,self.num_classes,1.0,0.0,axis=3,dtype=tf.float32)
            print 'GT_1hot shape ',self.GT_1hot.get_shape()
            print 'prediction shape ',self.prediction.get_shape
            self.D, self.D_logits = self.discriminator(self.GT_1hot)#real CT GT data (1hot so they have same n channels)as input      
            self.D_, self.D_logits_ = self.discriminator(self.probs_G, reuse=True)#fake generated CT probmaps as input
            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
            self.d_loss=self.d_loss_real+self.d_loss_fake
            self.g_loss, self.diceterm, self.fcnterm, self.bceterm=self.combined_loss_G(batch_size_tf)

            self.d_vars = [var for var in t_vars if 'd_' in var.name]

            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
                          .minimize(self.d_loss, var_list=self.d_vars)

        else:
            self.g_loss, self.diceterm, self.fcnterm=self.combined_loss_G(batch_size_tf)



        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        

        self.g_vars = [var for var in t_vars if 'g_' in var.name]
                     
        print 'learning rate ',self.learning_rate
        self.learning_rate_tensor = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                             self.lr_step, 0.1, staircase=True)
        #self.g_optim = tf.train.GradientDescentOptimizer(self.learning_rate_tensor).minimize(self.g_loss, global_step=self.global_step)
        self.g_optim = tf.train.MomentumOptimizer(self.learning_rate_tensor, 0.9).minimize(self.g_loss, global_step=self.global_step)
        
        self.merged = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter("./summaries", self.sess.graph)




        self.saver = tf.train.Saver(max_to_keep=50000)


    def generator(self,input_op,batch_size_tf):
               
        ######## FCN for the 32x32x32 to 24x24x24 ###################################
        conv1_1 = conv_op(input_op, name="g_conv1_1", kh=7, kw=7, n_out=32, dh=1, dw=1,wd=self.wd)#512x512
        conv1_2 = conv_op(conv1_1, name="g_conv1_2", kh=7, kw=7, n_out=32, dh=1, dw=1,wd=self.wd)
        pool1 = mpool_op(conv1_2,   name="g_pool1",   kh=2, kw=2, dw=2, dh=2)#256x256
        conv2_1 = conv_op(pool1, name="g_conv2_1", kh=7, kw=7, n_out=64, dh=1, dw=1,wd=self.wd)#256x256
        conv2_2 = conv_op(conv2_1, name="g_conv2_2", kh=7, kw=7, n_out=64, dh=1, dw=1,wd=self.wd)
        pool2 = mpool_op(conv2_2,   name="g_pool2",   kh=2, kw=2, dw=2, dh=2)#128x128
        conv3_1 = conv_op(pool2, name="g_conv3_1", kh=7, kw=7, n_out=96, dh=1, dw=1,wd=self.wd)
        conv3_2 = conv_op(conv3_1, name="g_conv3_2", kh=7, kw=7, n_out=96, dh=1, dw=1,wd=self.wd)
        pool3 = mpool_op(conv3_2,   name="g_pool2",   kh=2, kw=2, dw=2, dh=2)#64x64
        conv4_1 = conv_op(pool3, name="g_conv4_1", kh=7, kw=7, n_out=128, dh=1, dw=1,wd=self.wd)
        conv4_2 = conv_op(conv4_1, name="g_conv4_2", kh=7, kw=7, n_out=128, dh=1, dw=1,wd=self.wd)
        deconv1 = deconv_op(conv4_2,    name="g_deconv1", kh=4, kw=4, n_out=64, wd=self.wd, batchsize=batch_size_tf)#128x128
        concat1=concatenate_op(deconv1,conv3_2,name="g_concat1")
        deconv2 = deconv_op(concat1,    name="g_deconv2", kh=4, kw=4, n_out=64, wd=self.wd, batchsize=batch_size_tf)#256x256
        concat2=concatenate_op(deconv2,conv2_2,name="g_concat2")
        deconv3 = deconv_op(concat2,    name="g_deconv3", kh=4, kw=4, n_out=32, wd=self.wd, batchsize=batch_size_tf)#512x512
        concat3=concatenate_op(deconv3,conv1_2,name="g_concat3")
        upscore = conv_op(concat3, name="g_upscore", kh=7, kw=7, n_out=self.num_classes, dh=1, dw=1,wd=self.wd,activation=False)#512x512
        return upscore,upscore


    def discriminator(self, inputCT, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        print 'ct shape ',inputCT.get_shape()
        h0=conv_op_bn(inputCT, name="d_conv_dis_1_a", kh=5, kw=5, n_out=32, dh=1, dw=1, wd=self.wd, padding='VALID',train_phase=self.train_phase)
        print 'h0 shape ',h0.get_shape()
        m0=mpool_op(h0, 'pool0', kh=2, kw=2, dh=2, dw=2)
        print 'm0 shape ',m0.get_shape()
        h1 = conv_op_bn(m0, name="d_conv2_dis_a", kh=5, kw=5, n_out=64, dh=1, dw=1, wd=self.wd, padding='VALID',train_phase=self.train_phase)
        print 'h1 shape ',h1.get_shape()
        m1=mpool_op(h1, 'pool1', kh=2, kw=2, dh=2, dw=2)
        print 'mi shape ',m1.get_shape()
        h2 = conv_op_bn(m1, name="d_conv3_dis_a", kh=5, kw=5, n_out=128, dh=1, dw=1, wd=self.wd, padding='VALID',train_phase=self.train_phase)#28
        h3 = conv_op_bn(h2, name="d_conv4_dis_a", kh=5, kw=5, n_out=64, dh=1, dw=1, wd=self.wd, padding='VALID',train_phase=self.train_phase)
        fc1=fullyconnected_op(h3, name="d_fc1", n_out=64, wd=self.wd, activation=True)
        fc2=fullyconnected_op(fc1, name="d_fc2", n_out=32, wd=self.wd, activation=True)
        fc3=fullyconnected_op(fc2, name="d_fc3", n_out=1, wd=self.wd, activation=False)
        return tf.nn.sigmoid(fc3), fc3




    def train(self, config):
        path_test=config.dir_patients#'/home/trullro/CT_cleaned/'
        _, patients, _ = os.walk(path_test).next()#every folder is a patient
        patients.sort()
        patientstmp=patients[-4]
        print 'global_step ', self.global_step.name
        print 'lr_step ',self.lr_step
        print 'trainable vars '