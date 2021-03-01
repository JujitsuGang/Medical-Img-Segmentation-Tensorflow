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
    bodynp=sitk.Ge