from corr_utils import polar_angular_correlation, to_polar, circle_center,mask_correction
from plot_utils import plot_2d, plot_1d





import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

import h5py

from extra_geom import AGIPD_1MGeometry
from extra_data import open_run, stack_detector_data



geomfilename = '../../xfel3046/geometry/agipd_2995_v04.geom'
geom = AGIPD_1MGeometry.from_crystfel_geom(geomfilename)


# geom = AGIPD_1MGeometry.from_quad_positions(quad_pos=[
    # (-525, 625),
    # (-550, -10),
    # (520, -160),
    # (542.5, 475),
    # ])

center = circle_center(
        (276.134, 287.016),
        (410.735, 1034.42),
        (955.738, 562.901),
        )

# # geom.inspect()



run = open_run(proposal=3046, run=10)

sel = run.select('*/DET/*', 'image.data', require_all=True)

train_id, train_data = run.select('*/DET/*', 'image.data', require_all=True).train_from_index(60)
stacked = stack_detector_data(train_data, 'image.data')



###make mask
stacked_pulse = stacked[0][0]
res, _ = geom.position_modules(stacked_pulse)
mask = np.zeros(res.shape)
mask[np.where(res!=0)] = 1
mask_unwrap = to_polar(mask, 400, center[1], center[0])
# # mask_unwrap = mask_unwrap[:,:]
mask_corr = polar_angular_correlation(mask_unwrap)


fig, axes = plt.subplots(1,3)
plot_2d(mask, fig=fig, axes=axes[0])
plot_2d(mask_unwrap, fig=fig, axes=axes[1])
plot_2d(mask_corr, fig=fig, axes=axes[2])



# # plot_2d(res, fig=fig, axes=axes[0], vminmax=(5e3, 6e3))
# # plot_2d(mask, fig=fig, axes=axes[1], vminmax=(0, 1))





corr = np.zeros(mask_corr.shape)


count=0


# for train_id, data in sel.trains(require_all=True):
for train_id, data in [sel.train_from_index(40)]:
    print(count)
    if count>=1:
        break

    stacked = stack_detector_data(train_data, 'image.data')

    for i, pulse in enumerate(stacked):
        # print(i)

        stacked_pulse = pulse[1]
        res, _ = geom.position_modules(stacked_pulse)

        im_unwrap = to_polar(res, 400, center[1], center[0])

        # im_unwrap = im_unwrap[]

        im_corr = polar_angular_correlation(im_unwrap)

        corr += im_corr


        # im_corr_mask_corrected = mask_correction(im_corr, mask_corr)
        # corr += im_corr_mask_corrected


    count+=1

fig, axes = plt.subplots(1,3)
plot_2d(res, fig=fig, axes=axes[0], title='res')
plot_2d(im_unwrap, fig=fig, axes=axes[1], title='res_unwrap')
plot_2d(corr, fig=fig, axes=axes[2], title='corr')


mask_corrected_corr = mask_correction(corr, mask_corr)


fig, axes = plt.subplots(1,1)

plot_2d(mask_corrected_corr)








# fig, axes = plt.subplots(1,2)
# plot_2d(corr, fig=fig, axes=axes[0], title='corr',)
# plot_2d(corr, fig=fig, axes=axes[1], title='corr', vminmax=(0.4, 0.5))






# # plot_2d(corr, subtmean=True, title='corr mask corrected')
# # # plot_2d(im_corr_mask_corrected, blur=4, subtmean=False, title='')


# # fig, axes = plt.subplots(1,1)
# # plot_1d(corr[100:120, :].sum(axis=0), norm=True, fig=fig, axes=axes, label='corr (mask corrected)')
# # plot_1d(mask_corr[100:120, :].sum(axis=0), norm=True, fig=fig, axes=axes, label='mask', color='blue')
# # plt.legend()





plt.show()
