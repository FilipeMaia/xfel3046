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






run = open_run(proposal=3046, run=24)

sel = run.select('*/DET/*', 'image.data', require_all=True)

train_id, train_data = run.select('*/DET/*', 'image.data', require_all=True).train_from_index(60)
stacked = stack_detector_data(train_data, 'image.data')

stacked_pulse = stacked[20][0]
res, _ = geom.position_modules(stacked_pulse)



fig, axes = plt.subplots(1,1)
plot_2d(res, fig=fig, axes=axes, vminmax=(4e3, 5e3))
plt.show()


# ###make mask
# stacked_pulse = stacked[0][0]
# res, _ = geom.position_modules(stacked_pulse)
# mask = np.zeros(res.shape)
# mask[np.where(res!=0)] = 1
# mask_unwrap = to_polar(mask, 500, center[1], center[0])
# # # mask_unwrap = mask_unwrap[:,:]
# mask_corr = polar_angular_correlation(mask_unwrap)


# fig, axes = plt.subplots(1,3)
# plot_2d(mask, fig=fig, axes=axes[0])
# plot_2d(mask_unwrap, fig=fig, axes=axes[1])
# plot_2d(mask_corr, fig=fig, axes=axes[2])



# # # plot_2d(res, fig=fig, axes=axes[0], vminmax=(5e3, 6e3))
# # # plot_2d(mask, fig=fig, axes=axes[1], vminmax=(0, 1))





# corr = np.zeros(mask_corr.shape)


# count=0


# # for train_id, data in sel.trains(require_all=True):
# for train_id, data in [sel.train_from_index(60)]:
    # print(count)
    # if count>=1:
        # break

    # stacked = stack_detector_data(train_data, 'image.data')

    # for i, pulse in enumerate(stacked):
        # # print(i)

        # stacked_pulse = pulse[1]
        # res, center = geom.position_modules(stacked_pulse)

        # im_unwrap = to_polar(res, 500, center[1], center[0])

        # # im_unwrap = im_unwrap[]

        # im_corr = polar_angular_correlation(im_unwrap)


        # im_corr_mask_corrected = mask_correction(im_corr, mask_corr)
        # corr += im_corr_mask_corrected

    # count+=1








# fig, axes = plt.subplots(1,2)
# plot_2d(corr, fig=fig, axes=axes[0], title='corr',)
# plot_2d(corr, fig=fig, axes=axes[1], title='corr', vminmax=(0.4, 0.5))






# # plot_2d(corr, subtmean=True, title='corr mask corrected')
# # # plot_2d(im_corr_mask_corrected, blur=4, subtmean=False, title='')


# # fig, axes = plt.subplots(1,1)
# # plot_1d(corr[100:120, :].sum(axis=0), norm=True, fig=fig, axes=axes, label='corr (mask corrected)')
# # plot_1d(mask_corr[100:120, :].sum(axis=0), norm=True, fig=fig, axes=axes, label='mask', color='blue')
# # plt.legend()





# plt.show()
