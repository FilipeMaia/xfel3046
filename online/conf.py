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
state['EuXFEL/DataSource'] = 'tcp://10.253.0.74:55777'

mask_file = '/home/awollter/p003046/usr/Shared/awollter/xfel3046/mask/agipd_mask.h5'
with h5.File(mask_file,'r') as f:
    mask = np.asarray(f['combined'])

geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom('/home/awollter/p003046/usr/Shared/awollter/xfel3046/geometry/p3046_manual_refined_geoass_run10.geom')
#geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom('geometry/agipd_august_2022_v3.geom')

#geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom('geometry/agipd_2995_v04.geom')
assemask, centremask = geom.position_modules_fast(mask)
pixel_size = 200e-6
detector_distance = 0.217
wavelength = 0.15498e-9 #8 keV
npt = 256
ai = azimuthalIntegrator.AzimuthalIntegrator(dist = detector_distance,
                                   poni1 = centremask[0] * pixel_size,
                                   poni2 = centremask[1] * pixel_size,
                                   pixel1 = pixel_size,
                                   pixel2 = pixel_size,
                                   rot1 = 0, rot2=0, rot3=0,
                                   wavelength = wavelength)
    
def onEvent(evt):
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
    #plotting.line.plotTrace(laser_voltage, group = 'analysis')
    
    #det = evt['photonPixelDetectors']['AGIPD01'].data
    # det = evt['SPB_DET_AGIP1M-1/DET/STACKED:xtdf']['
    #print(evt['SPB_DET_AGIP1M-1/DET/STACKED:xtdf'].keys())
    trainId = evt['SPB_DET_AGIPD1M-1/DET/10CH0:xtdf']['header.trainId'].data
    #print(laser_voltage.data.max() )
    laserOn = laser_voltage.data.max() > 0
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

    assem, centre = geom.position_modules_fast(np.nanmean(modules, axis = 0))
    #assem[:200,:200] = 10
    print(assem.mean())
    #print(np.isnan(assem))
    print(assem.shape)
    #print(centre)
    #assem[np.isnan(assem)] = -1
    assem[assemask] = -1
    assem_rec = add_record(evt['analysis'], 'analysis','assem_rec', assem[::-1,:])
    plotting.image.plotImage(assem_rec, history=10)
    
    
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
    
    
    #print(assem.shape)
    #det_arr[module_numbers] = mods[:,ind]                                                                           
    #assem = geom.position_modules_fast(det_arr)[0][::-1,::-1]                                                       
    #assem[np.isnan(assem)] = -1
    #brightest_hit = add_record(evt['analysis'], 'analysis', 'Random hit', assem)
    #random_image = add_record(evt["analysis"], "analysis", "Random Image", module.data[10,0])                       
    #print(assem.shape)
    
