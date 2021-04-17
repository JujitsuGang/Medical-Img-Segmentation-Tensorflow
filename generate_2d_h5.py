'''
Created on Jul 1, 2016

@author: roger
'''
import h5py, os
import numpy as np
import SimpleITK as sitk
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#import caffe
from multiprocessing import Pool
import argparse

def worker(idx,namepatient,path_patients,dirname):
    print namepatient
    ctitk=sitk.ReadImage(os.path.join(path_patients,namepatient,namepatient+'.nii.gz')) 
    
       
    ctnp=sitk.GetArrayFromImage(ctitk)
    ctnp[np.where(ctnp>3000)]=3000#we calp the images so they are in range -1000 to 3000  HU
    muct=np.mean(ctnp)
    stdct=np.std(ctnp)
    
    ctnp=(1/stdct)*(ctnp-muct)#normalize each patient
    segitk=sitk.ReadImage(os.path.join(path_patients,namepatient,'GT.nii.gz'))
    segnp=sitk.GetArrayFromImage(segitk)
    
    bodyitk=sitk.ReadImage(os.path.join(path_patients,namepatient,'CONTOUR.nii.gz'))
    bodynp=sitk.GetArrayFromImage(bodyitk)
    
    idxbg=np.where(bodynp==0)
    ctnp[idxbg]=np.min(ctnp)#just put the min val in the parts that are not body
    segnp[idxbg]=5#ignore this value in the protoxt
    
    
    list_idx=[]
    for i in xrange(ctnp.shape[0]):#select slices with organs
        if len(np.unique(segnp[i,:,:]))>2:
            list_idx.append(i)
    ctnp=ctnp[list_idx,:,:]
    segnp=segnp[list_idx,:,:]
    
    #shuffle the data
    idx_rnd=np.random.choice(ctnp.shape[0], ctnp.shape[0], replace=False)
    ctnp=ctnp[idx_rnd,:,:]
    segnp=segnp[idx_rnd,:,:]
    ctnp = np.expand_dims(ctnp, axis=1) 
    train_filename = os.path.join(dirname, 'train{}.h5'.format(idx))
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    with h5py.File(train_filename, 'w') as f:
        f.create_dataset('data', data=ctnp, **comp_kwargs)
        f.create_dataset('label', data=segnp, **comp_kwargs)        
    print "patient  {0} finishedm type:  {1}!".format(namepatient, ctnp.dtype)

def create_training(path_patients,dirsaveto):
    _, patients, _ = os.walk(path_patients).next()#ever