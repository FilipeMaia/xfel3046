import plotting.image
import plotting.line
import analysis.agipd
import analysis.event
from backend import add_record
import numpy as np
import h5py as h5
import extra_geom
from pyFAI import azimuthalIntegrator
state = {}
state['Facility'] = 'EuXFEL'
#state['EuXFEL/DataSource'] = 'tcp://10.253.0.74:55777'
state['EuXFEL/DataSource'] = 'tcp://max-exfl-display01.desy.de:9876'


npt = 256

light = np.zeros(npt)
dark = np.zeros(npt)
diff = np.zeros(npt)

n_light = 0
n_dark = 0

mask_file = '../mask/agipd_mask.h5'
with h5.File(mask_file,'r') as f:
    mask = np.asarray(f['combined'])

geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom('/home/awollter/p003046/usr/Shared/awollter/xfel3046/geometry/p3046_manual_refined_geoass_run10.geom')
#geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom('geometry/agipd_august_2022_v3.geom')

#geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom('geometry/agipd_2995_v04.geom')
assemask, centremask = geom.position_modules_fast(mask)
pixel_size = 200e-6
detector_distance = 0.217
wavelength = 0.15498e-9 #8 keV
ai = azimuthalIntegrator.AzimuthalIntegrator(dist = detector_distance,
                                   poni1 = centremask[0] * pixel_size,
                                   poni2 = centremask[1] * pixel_size,
                                   pixel1 = pixel_size,
                                   pixel2 = pixel_size,
                                   rot1 = 0, rot2=0, rot3=0,
                                wavelength = wavelength)


'''
find a way to implement a mask for an ROI that can be used to calculate difference scattering
roi_mask
using module 3, second asic from edge (p2a1 in geom file) 
y = 0:127, x = 64:127
roi_mask_x = [0,127]
roi_mask_y = [64,127
'''

 
def onEvent(evt):
    global dark
    global light
    global diff
    global n_light
    global n_dark
    '''
    Print Processing rate
    '''
    
    analysis.event.printProcessingRate()

    #print(analysis.event.printNativeKeys(evt))
    #print(evt['SPB_RR_SYS/ADC/UTC1-2.channel_0.output.schema.data.rawData'])
    #print(evt['SPB_DET_AGIPD1M-1/DET/9CH0:xtdf']['image.data'])
    #print(evt['SPB_DET_AGIP1M-1/DET/STACKED:xtdf']['image.data'])
    #print(evt['SPB_RR_SYS/ADC/UTC1-2:channel_0.output'].keys())

    laser_voltage = evt['SPB_RR_SYS/ADC/UTC1-2:channel_0.output']['data.rawDataVolt']
    laser_voltage.data = np.asarray(laser_voltage.data)


    #print(laser_voltage.data.shape)
    #    laser_rec = add_record(evt['analysis'], "analysis", "LASER", laser_voltage)
    plotting.line.plotTrace(laser_voltage, group = 'analysis')
    
    #det = evt['photonPixelDetectors']['AGIPD01'].data
    # det = evt['SPB_DET_AGIP1M-1/DET/STACKED:xtdf']['
    #print(evt['SPB_DET_AGIP1M-1/DET/STACKED:xtdf'].keys())

    trainId = evt['SPB_DET_AGIPD1M-1/DET/10CH0:xtdf']['header.trainId'].data

    #print(laser_voltage.data.max() )

    laserOn = int(laser_voltage.data.max() > 0)

    laserOn_rec = add_record(evt['analysis'], "analysis", "laser on", laserOn)
    plotting.line.plotHistory(evt['analysis']['laser on'],label='laser on')

    #print('LASER ON' if laserOn else 'LASER OFF')
    #print(det[0,0,:,:])
    #module = add_record(evt["analysis"], "analysis", "single", det[10,2,:,:])
 
    #print(module.data.sum())

    '''
    Assemble modules and plot
    '''
    
    #print(centre)
    # applying the mask on the assebled detector image
    # assem[:, mask] = np.nan    
    # plotting.image.plotImage(assem_rec, history=10, log = True)

    modules = np.array(evt['photonPixelDetectors']['AGIPD Stacked'].data[1:])

    # normalisation of the 2D scattering patterns needed
    # norm_2d = np.average(modules[:, 2, roi_mask_x[0]:roi_mask_x[1], roi_mask_y[0]:roi_mask_x[1]])
    
    assem, centre = geom.position_modules_fast(np.nanmean(modules, axis = 0))

    #assem[:200,:200] = 10
    #print(np.isnan(assem))
    #print(centre)
    #assem[np.isnan(assem)] = -1

    assem[assemask] = -1
    assem_rec = add_record(evt['analysis'], 'analysis','assem_rec', assem[::-1,:])
    plotting.image.plotImage(assem_rec, history=10)


    i_sum = np.sum(assem[:])

    i_sum_rec = add_record(evt['analysis'], 'analysis','i_sum_rec', i_sum)
    
    plotting.line.plotHistory(evt['analysis']['i_sum_rec'],label='sum_I')

    
    Q, i = ai.integrate1d(assem,
                          npt,
                          #method = "BBox",
                          #mask = (assemask | (assem = 0)),
                          radial_range = (0.1, 2.4),
                          #correctSolidAngle = True,
                          #polarization_factor = 1,
                          unit = "q_A^-1")
        
    Q_rec = add_record(evt['analysis'], 'analysis', 'Q', Q)
    i_rec = add_record(evt['analysis'], 'analysis', 'i', i)
    
    plotting.line.plotTrace(i_rec, Q_rec)   
    
    #normalize the scattering curves to intensity from q>1.4 to q<1.7
    norm = np.average(i[np.logical_and(Q>1.4, Q<1.7)])
    i_norm = i/norm    

    print("Laser On, ", laserOn)

    if laserOn:
        light += i 
        n_light += 1
    else:
        dark += i
        n_dark += 1
        
    if (n_light > 0 and n_dark > 0):
        diff = light/n_light-dark/n_dark
        diff_rec = add_record(evt['analysis'], 'analysis', 'diff', diff)
        plotting.line.plotTrace(diff_rec, Q_rec)
    else:
        return
        


    #print(assem.shape)
    #det_arr[module_numbers] = mods[:,ind]                                                                           
    #assem = geom.position_modules_fast(det_arr)[0][::-1,::-1]                                                       
    #assem[np.isnan(assem)] = -1
    #brightest_hit = add_record(evt['analysis'], 'analysis', 'Random hit', assem)
    #random_image = add_record(evt["analysis"], "analysis", "Random Image", module.data[10,0])                       
    #print(assem.shape)
    
