#!/usr/bin/env python

import numpy as np
from skimage.transform import warp_polar



def angle_between_pol(t1, t2):
    psi = np.abs((t1 - t2 + np.pi) % (np.pi*2) - np.pi)
    return psi


def angle_between_rect(q1, q2):
    dot = np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))
    if dot > 1:
        dot = 1.0
    elif dot < -1:
        dot = -1.0
    psi = np.arccos(dot)
    return psi


def angle_between_sph(theta1, theta2, phi1, phi2):
    w1 = np.array([np.cos(phi1) * np.sin(theta1),
                   np.sin(phi1) * np.sin(theta1),
                   np.cos(theta1)])

    w2 = np.array([np.cos(phi2) * np.sin(theta2),
                   np.sin(phi2) * np.sin(theta2),
                   np.cos(theta2)])
    return angle_between_rect(w1, w2)


def index_x(x_val, x_min, x_max, nx, wrap=False):
    dx = (x_max - x_min) / nx
    x_val = round(x_val, 14)

    if not wrap:
        x_out = (x_val - x_min) / dx
        if x_val == x_max:
            x_out = nx - 1
    else:
        if x_val <= x_min + dx / 2 or x_val >= x_max - dx / 2:
            x_out = 0

        else:
            x_out = index_x(x_val, x_min + dx / 2, x_max - dx / 2, nx - 1) + 1

    return int(x_out)


def circle_center(a, b, c):

    D = 2*(a[0]*(b[1] - c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]) )

    ux = (a[0]**2+a[1]**2)*(b[1]-c[1]) + (b[0]**2+b[1]**2)*(c[1]-a[1]) +\
            (c[0]**2+c[1]**2)*(a[1]-b[1])

    uy = (a[0]**2+a[1]**2)*(c[0]-b[0]) + (b[0]**2+b[1]**2)*(a[0]-c[0]) +\
            (c[0]**2+c[1]**2)*(b[0]-a[0])

    return ( ux/D, uy/D )


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

    if polar2 is None:
        polar2 = polar[:]

    fpolar = np.fft.fft( polar, axis=1 )
    fpolar2 = np.fft.fft( polar2, axis=1)

    out = np.fft.ifft( fpolar.conjugate() * fpolar2, axis=1 )
    return np.real(out)




def polar_angular_intershell_correlation( polar, polar2=None):

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
    smallimask = np.where( np.abs(maskcorr) > 1e-10 )
    corr[imask] *= 1.0/maskcorr[imask]

    corr[smallimask] =0
    return corr




