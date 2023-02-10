
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
    for idx in xrange(zmin,zmax):
        eso_slice=seg_eso[idx]
        centroid=center_of_mass(eso_slice)
        if not ini_found:#if we have not found the first slice empty
            if np.isnan(centroid).any():#look for the first emppty slice
                #print 'is NAN ',idx
                ini=idx-1
                pini=list(center_of_mass(seg_eso[idx-1]))
                pini.append(idx-1)
                ini_found=True
        else:#if we have already found the first empty slice, look for the final one
            idvox=np.where(eso_slice==1)
            nvoxels=len(idvox[0])
            if not np.isnan(centroid).any() and nvoxels>5:#the slice with data and enough voxels

                #print 'final nan ',idx
                fin=idx
                pfin=list(center_of_mass(seg_eso[fin]))
                pfin.append(idx)
                #print 'pini ',pini
                #print 'pfin ',pfin
                for z in xrange(ini,fin):#we will fill the empty slices here
                    newcenter=interpolateline(pini,pfin,z)
                    #print 'new center ',newcenter
                    #print 'prev center ',center_of_mass(seg_eso[z-1])
                    translation=np.int16(np.array(newcenter)-np.array(center_of_mass(seg_eso[z-1])))
                    #print 'trans ',translation
                    #tx = tf.SimilarityTransform(translation=(0,0))#tuple(translation)
                    if z==ini:
                        slicetmp = shift(seg_eso[z-5],translation)#tf.warp(seg_eso[z-1], tx)
                    else:
                        slicetmp = shift(seg_eso[z-1],translation)#tf.warp(seg_eso[z-1], tx)
                    #print 'unique slice befor trans ',np.unique(seg_eso[z-1])
                    #print 'unique slice tmp ',np.unique(slicetmp)
                    seg_eso[z]=slicetmp
                ini_found=False
    idxeso=np.where(seg_eso>0)
    volfinal=np.copy(vol_out)
    volfinal[idxesoini]=0
    volfinal[idxeso]=1
    return volfinal
    
    
def interpolateline(p0,p1,z):
    #p1 and p2 are 3d points x,y,z and z is the slice for which we want to compute x and y
    
    x=(float(z-p0[2])/(p1[2]-p0[2]))*(p1[0]-p0[0])+p0[0]
    y=(float(z-p0[2])/(p1[2]-p0[2]))*(p1[1]-p0[1])+p0[1]
    print 'x ',x
    print 'y ',y
    return x,y


def postprocess(vol_out):
    print 'postprocessing now...'
    r=int(vol_out.shape[1]/2.0)
    c=int(vol_out.shape[2]/2.0)
    sizecropup=150
    sizecropdown=100
    sizecrop=200

    mask=np.zeros_like(vol_out)
    mask[20:,r-sizecropup:r+sizecropdown,c-sizecrop/2:c+sizecrop/2]=1
    mask[-25:,r-sizecropup:r+sizecropdown,c-sizecrop/2:c+sizecrop/2]=0
    vol_out*=mask
    print vol_out.shape
    print np.unique(vol_out)

    volheart=vol_out==2
    volheart=volheart.astype(np.uint8)
    idxheartini=np.where(volheart>0)

    volaorta=vol_out==4
    volaorta=volaorta.astype(np.uint8)
    idxaortaini=np.where(volaorta>0)

    voltrach=vol_out==3
    voltrach=voltrach.astype(np.uint8)
    idxtrachini=np.where(voltrach>0)

    voleso=vol_out==1
    voleso=voleso.astype(np.uint8)
    idxesoini=np.where(voleso>0)

    cc = sitk.ConnectedComponentImageFilter()

    vol_out1 = cc.Execute(sitk.GetImageFromArray(volheart))
    voltmp=sitk.RelabelComponent(vol_out1)
    volheartfiltered=sitk.GetArrayFromImage(voltmp)
    volheartfiltered=volheartfiltered==1

    vol_out2 = cc.Execute(sitk.GetImageFromArray(volaorta))
    voltmp=sitk.RelabelComponent(vol_out2)
    volaortafiltered=sitk.GetArrayFromImage(voltmp)
    volaortafiltered=volaortafiltered==1

    vol_out3 = cc.Execute(sitk.GetImageFromArray(voltrach))
    voltmp=sitk.RelabelComponent(vol_out3)
    voltrachfiltered=sitk.GetArrayFromImage(voltmp)
    voltrachfiltered=voltrachfiltered==1

    vol_out4 = sitk.GetImageFromArray(voleso)
    voltmp=sitk.BinaryMedian(vol_out4)
    volesofiltered=sitk.GetArrayFromImage(voltmp)
    #vol_out4 = cc.Execute(sitk.GetImageFromArray(voleso))
    #voltmp=sitk.RelabelComponent(vol_out4)
    #volesofiltered=sitk.GetArrayFromImage(voltmp)
    #volesofiltered=volesofiltered==1

    maskheart=np.logical_and(volheartfiltered,volheart)
    maskaorta=np.logical_and(volaortafiltered,volaorta)
    masktrachea=np.logical_and(voltrachfiltered,voltrach)
    maskeso=volesofiltered>0#np.logical_and(volesofiltered,volesofiltered)
    #maskeso=np.logical_and(volesofiltered,voleso)

    for ind in xrange(volheartfiltered.shape[0]):
        maskheart[ind]=binary_fill_holes(maskheart[ind]).astype(int)
        maskaorta[ind]=binary_fill_holes(maskaorta[ind]).astype(int)
        masktrachea[ind]=binary_fill_holes(masktrachea[ind]).astype(int)
        maskeso[ind]=binary_fill_holes(maskeso[ind]).astype(int)
    idxheart=np.where(maskheart>0)
    idxaorta=np.where(maskaorta>0)
    idxtrachea=np.where(masktrachea>0)
    idxeso=np.where(maskeso>0)
    vol_out[idxheartini]=0
    vol_out[idxheart]=2
    vol_out[idxaortaini]=0
    vol_out[idxaorta]=4
    vol_out[idxtrachini]=0
    vol_out[idxtrachea]=3
    vol_out[idxesoini]=0
    vol_out[idxeso]=1

    volfinal=np.copy(vol_out)
    return volfinal





def psnr(ct_generated,ct_GT):
    print ct_generated.shape
    print ct_GT.shape

    mse=np.sqrt(np.mean((ct_generated-ct_GT)**2))
    print 'mse ',mse
    max_I=np.max([np.max(ct_generated),np.max(ct_GT)])
    print 'max_I ',max_I
    return 20.0*np.log10(max_I/mse)

def dice(im1, im2,organid):