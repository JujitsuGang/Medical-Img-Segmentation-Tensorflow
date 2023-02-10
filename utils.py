
import tensorflow as tf
import numpy as np
import os
import SimpleITK as sitk
import h5py
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import glob

from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.interpolation import shift

ignore_label=0



def process_eso(vol_out):
    print 'iterpolating eso...'
    voleso=vol_out==1
    voleso=voleso.astype(np.uint8)
    idxesoini=np.where(voleso>0)
    
    id_eso=np.where(vol_out==1)
    seg_eso=np.zeros_like(vol_out)
    seg_eso[id_eso]=1
    listorgan=np.where(seg_eso>0)
    zmin=np.min(listorgan[0])
    zmax=np.max(listorgan[0])
    ini_found=False