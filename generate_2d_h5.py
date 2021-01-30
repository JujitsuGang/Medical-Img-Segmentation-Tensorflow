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
    ctitk=sitk.ReadImage(os.path.join(path_patients,namepatient,nam