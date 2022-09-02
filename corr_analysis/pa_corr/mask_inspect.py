from corr_utils import polar_angular_correlation, to_polar, circle_center,mask_correction
from plot_utils import plot_2d, plot_1d





import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

import h5py

from extra_geom import AGIPD_1MGeometry
from extra_data import open_run, stack_detector_data
import extra_data



geomfilename = '../p3046_manual_refined_geoass_run10.geom'
geom = AGIPD_1MGeometry.from_crystfel_geom(geomfilename)



maskh5fname = '../agipd_mask.h5'

mask_file = h5py.File(maskh5fname, 'r')

print(mask_file['combined'])


mask, center = geom.position_modules(mask_file['combined'])

plot_2d(mask)







plt.show()
