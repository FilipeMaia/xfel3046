#!/usr/bin/env python

import numpy as np
import h5py

import hdf5plugin
import matplotlib.pyplot as plt

from PIL import Image
import os
from skimage.transform import warp_polar



def to_polar(im, rmax, cenx, ceny):
    '''
    unwraps 2d cartesian (x,y) image to 2d polar (theta, r) image
    im: 2D image to unwrap
    rmax: radius of circle (pixels)
    cenx, ceny: pixels of center
    '''
    x = warp_polar( im, center=(cenx,ceny), radius=rmax)
    return np.rot90(x, k=3)





def polar_angular_correlation(  polar, polar2=None):
    '''
    calculates correlation from convolution of polar images
    polar: polar image to correlate
    polar2: polar image to convolve. leave none for auto correlation of polar

    returns: out (q1=q2, psi correlation plane)

    '''

    # fpolar = np.fft.fft( polar, axis=1 )
    # if polar2 is not None:
        # fpolar2 = np.fft.fft( polar2, axis=1)
        # out = np.fft.ifft( fpolar2.conjugate() * fpolar, axis=1 )
    # else:
        # out = np.fft.ifft( fpolar.conjugate() * fpolar, axis=1 )


    if polar2 is None:
        polar2 = polar[:]

    fpolar = np.fft.fft( polar, axis=1 )
    fpolar2 = np.fft.fft( polar2, axis=1)

    out = np.fft.ifft( fpolar.conjugate() * fpolar2, axis=1 )
    return np.real(out)




def polar_angular_intershell_correlation( polar, polar2=None):
    '''
    calculates correlation from convolution of polar images
    polar: polar image to correlate
    polar2: polar image to convolve. leave none for auto correlation of polar

    returns: out (q1, q2, psi correlation volume)

    '''

#     fpolar = np.fft.fft( polar, axis=1 )

    # if polar2 != None:
        # fpolar2 = np.fft.fft( polar2, axis=1)
    # else:
        # fpolar2 = fpolar

    # out = np.zeros( (polar.shape[0],polar.shape[0],polar.shape[1]) )
    # for i in np.arange(polar.shape[0]):
        # for j in np.arange(polar.shape[0]):
            # out[i,j,:] = fpolar[i,:]*fpolar2[j,:].conjugate()
    # out = np.fft.ifft( out, axis=2 )

    if polar2 is None:
        polar2 = polar[:]

    fpolar = np.fft.fft( polar, axis=1 )
    fpolar2 = np.fft.fft( polar2, axis=1)

    out = np.zeros( (polar.shape[0],polar.shape[0],polar.shape[1]) )
    for i in np.arange(polar.shape[0]):
        for j in np.arange(polar.shape[0]):
            out[i,j,:] = fpolar[i,:]*fpolar2[j,:].conjugate()
    out = np.fft.ifft( out, axis=2 )

    return out

def mask_correction(  corr, maskcorr ):
    imask = np.where( maskcorr != 0 )
    corr[imask] *= 1.0/maskcorr[imask]
    return corr






