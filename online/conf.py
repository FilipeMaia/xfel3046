import plotting.image
import plotting.histogram
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
state['EuXFEL/DataSource'] = 'tcp://10.253.0.74:55777'


npt = 256

light = np.zeros(npt)
dark = np.zeros(npt)
diff = np.zeros(npt)

n_light = 0
n_dark = 0

light_image = None
dark_image = None
latest_light = None
latest_dark = None
light_image_norm = None
dark_image_norm = None
diff_image_norm = None

mask_file = '../mask/newer_mask.h5'
with h5.File(mask_file,'r') as f:
    mask = np.asarray(f['combined_mask'])

geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom('/home/awollter/p003046/usr/Shared/awollter/xfel3046/geometry/agipd_september_2022_v03.geom')
#geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom('/home/awollter/p003046/usr/Shared/awollter/xfel3046/geometry/p3046_manual_refined_geoass_run10.geom')

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
find a wa0y to impl.....................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................ement a mask for an ROI that can be used to calculate difference scattering
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
    global light_image
    global dark_image
    global latest_light
    global latest_dark
    global light_image_norm
    global dark_image_norm
    global diff_image_norm

    '''
    Print Processing rate
    '''
    
    analysis.event.printProcessingRate()
    #print(analysis.event.printNativeKeys(evt))
    #print(evt['SPB_RR_SYS/ADC/UTC1-2.channel_0.output.schema'])
    #print(evt['SPB_DET_AGIPD1M-1/DET/9CH0:xtdf']['image.data'])
    #print(evt['SPB_DET_AGIP1M-1/DET/STACKED:xtdf']['image.data'])
    #print(evt['SPB_RR_SYS/ADC/UTC1-2:channel_0.output'].keys())
    #print(evt['SPB_XTD9_XGM/XGM/DOOCS'].keys()) # For the gas monitor
    #xgm = evt['SPB_XTD9_XGM/XGM/DOOCS']['output.data.intensitySa1TD.value']
    #print(xgm.data)
    '''
    rawDataVolt used to define when the laser is on
    '''
    laser_voltage = evt['SPB_RR_SYS/ADC/UTC1-2:channel_0.output']['data.rawDataVolt']
    laser_voltage.data = np.asarray(laser_voltage.data)

    #print(laser_voltage.data.shape)
    #    laser_rec = add_record(evt['analysis'], "analysis", "LASER", laser_voltage)

    #___________: plot laser Voltage
    plotting.line.plotTrace(laser_voltage, group = 'laser')
    
    #det = evt['photonPixelDetectors']['AGIPD01'].data
    # det = evt['SPB_DET_AGIP1M-1/DET/STACKED:xtdf']['
    #print(evt['SPB_DET_AGIP1M-1/DET/STACKED:xtdf'].keys())

    trainId = evt['SPB_DET_AGIPD1M-1/DET/10CH0:xtdf']['header.trainId'].data

    #print(laser_voltage.data.max() )
 
    #___________: define laser on, if laser_voltage is not streamed use temporary solution
    laser_on = float(laser_voltage.data.max() > 0)

    
    '''
    Commented part fo
    '''
    #!!!! TEMPORARY RANDOM LASER ON !!!!
    # laser_on = trainId%2
    # laser_on = (np.random.random() > 0.5)
    laserOn = laser_on


    # Does not work at the moment 
    laser_on_rec = add_record(evt['analysis'], "analysis", "laser on", laser_on)
    plotting.line.plotHistory(laser_on_rec, group='laser')

    #print('LASER ON' if laserOn else 'LASER OFF')
    #print(det[0,0,:,:])
    #module = add_record(evt["analysis"], "analysis", "single", det[10,2,:,:])
 
    #print(module.data.sum())

    '''
    Assemble modules
    '''
    
    #print(centre)
    # applying the mask on the assebled detector image
    # assem[:, mask] = np.nan    
    # plotting.image.plotImage(assem_rec, history=10, log = True)
    
    modules = np.array(evt['photonPixelDetectors']['AGIPD Stacked'].data[1:174])
    i_per_pulse =  np.nansum(modules[:,mask==1], axis=1)
    i_mask = np.abs(i_per_pulse-np.mean(i_per_pulse))/np.abs(i_per_pulse) > 0.2
    #print(i_mask.shape)
    # normalisation of the 2D scattering patterns needed
    # norm_2d = np.average(modules[:, 2, roi_mask_x[0]:roi_mask_x[1], roi_mask_y[0]:roi_mask_x[1]])
    modules_mask = modules[~i_mask,:]
    print(modules_mask.shape)
    assem, centre = geom.position_modules_fast(np.nanmean(modules_mask, axis = 0))

    assem[assemask != 1] = np.nan
    assem_norm = assem/np.nansum(assem[50:150,50:150])
        
    #print(i_per_pulse.shape)
 
    i_per_pulse_rec = add_record(evt['analysis'], 'analysis','i_per_pulse', i_per_pulse)   
    plotting.line.plotTrace(i_per_pulse_rec)

    i_sum = np.nansum(assem[:])
    i_sum_rec = add_record(evt['analysis'], 'analysis','i_sum_rec', i_sum)   
    plotting.line.plotHistory(evt['analysis']['i_sum_rec'],label='sum_I')


    assemask_rec = add_record(evt['analysis'], 'analysis','mask', assemask)   
    plotting.image.plotImage(assemask_rec, history=10, vmin=0., vmax=4.)
    '''
    Do Azimuthal integration
    '''

    Q, i = ai.integrate1d(assem,
                          npt,
                          #method = "BBox",
                          #mask = (assemask | (assem = 0)),
                          radial_range = (0.1, 2.4),
                          #correctSolidAngle = True,
                          #polarization_factor = 1,
                          unit = "q_A^-1")

    assem[np.isnan(assem)] = -1
    assem_rec = add_record(evt['analysis'], 'analysis','assem_rec', assem[::-1,::-1])
    
    #___________: plot assembled detector
    plotting.image.plotImage(assem_rec, history=10, vmin=0., vmax=4.)
        
    Q_rec = add_record(evt['analysis'], 'analysis', 'Q', Q)
    i_rec = add_record(evt['analysis'], 'analysis', 'i', i)
    
    #___________: plot azimuthal inegration
    plotting.line.plotTrace(i_rec, Q_rec)   
    
    #normalize the scattering curves to intensity from q>1.4 to q<1.7
    norm = np.nansum(i[np.logical_and(Q>.4, Q<1.1)])  #/len([np.logical_and(Q>0.2, Q<2.2])) )
    i_norm = i/norm  

    #print(norm)  

    i_norm_rec = add_record(evt['analysis'], 'analysis', 'i norm', i_norm)
    plotting.line.plotTrace(i_norm_rec, Q_rec)   

    '''
    Look at low Q
    '''

    low_q_sum = (np.nansum(i_norm[np.logical_and(Q>.1, Q<.15)]))
    low_q_sum_rec = add_record(evt['analysis'], 'analysis', 'low q sum', low_q_sum)
    print(low_q_sum)

    #___________: plot azimuthal inegration at low q
    plotting.histogram.plotHistogram(low_q_sum_rec, bins = 50, hmin = -3, hmax = 3)   
    plotting.line.plotHistory(low_q_sum_rec, group = 'low q')
    
    print("Laser On, ", laserOn)
    print("n_light, ", n_light)
    print("n_dark, ", n_dark)

    '''
    Light On and Off images
    '''

    if laserOn:
        light += i_norm
        n_light += 1
        latest_light = i_norm

        if light_image is None:
            light_image = assem
            light_image_norm = assem_norm

        else:
            light_image += assem
            light_image_norm += assem_norm

        light_rec = add_record(evt['analysis'], 'analysis','light_image', light_image/n_light)
    
        plotting.image.plotImage(light_rec, history=10, vmin=0., vmax=300.)

    else:
        latest_dark = i_norm

        if dark_image is None:
            dark_image = assem
            dark_image_norm = assem_norm

        else:
            dark_image += assem
            dark_image_norm += assem_norm

        dark += i_norm
        n_dark += 1
        dark_rec = add_record(evt['analysis'], 'analysis','dark_image', dark_image/n_dark)

        plotting.image.plotImage(dark_rec, history=10, vmin=0., vmax=300.)
    
    if (n_light > 0 and n_dark > 0):
        print("here")
        diff = light/n_light - dark/n_dark
        diff_image_norm = light_image_norm/n_light - dark_image_norm/n_dark

        print(diff.shape)
        diff_rec = add_record(evt['analysis'], 'analysis', 'diff', diff)

        plotting.line.plotTrace(diff_rec, Q_rec)

        diff_image_rec = add_record(evt['analysis'], 'analysis','diff_image', light_image/n_light-dark_image/n_dark)

        plotting.image.plotImage(diff_image_rec, history=10, vmin=-100., vmax=100.)

    else:
        diff_rec = add_record(evt['analysis'], 'analysis', 'diff', np.zeros((256,)))
       
        plotting.line.plotTrace(diff_rec, Q_rec)
        
    if ((latest_light is not None) and (latest_dark is not None)):
        latest_diff = latest_light-latest_dark
        latest_diff_rec = add_record(evt['analysis'], 'analysis', 'latest_diff', latest_diff)
        
        plotting.line.plotTrace(latest_diff_rec, Q_rec)

    if diff_image_norm is not None:
        Q_diff_norm , i_diff_norm = ai.integrate1d(diff_image_norm,
                                                   npt,
                                                   #method = "BBox",
                                                   #mask = (assemask | (assem = 0)),
                                                   radial_range = (0.1, 2.4),
                                                   #correctSolidAngle = True,
                                                   #polarization_factor = 1,
                                                   unit = "q_A^-1")
    
        i_diff_norm_rec = add_record(evt['analysis'], 'analysis', 'i diff norm', i_diff_norm)
    
        plotting.line.plotTrace(i_diff_norm_rec, Q_rec)
        
        
    #print(assem.shape)
    #det_arr[module_numbers] = mods[:,ind]                                                                           
    #assem = geom.position_modules_fast(det_arr)[0][::-1,::-1]                                                       
    #assem[np.isnan(assem)] = -1
    #brightest_hit = add_record(evt['analysis'], 'analysis', 'Random hit', assem)
    #random_image = add_record(evt["analysis"], "analysis", "Random Image", module.data[10,0])                       
    #print(assem.shape)
    
