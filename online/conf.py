import plotting.image
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
detector_distance = 0.334
wavelength = 0.15498e-9 #8 keV


def onEvent(evt):
    '''
    Print Processing rate
    '''
   
    analysis.event.printProcessingRate()

    #print(analysis.event.printNativeKeys(evt))
    #print(evt['SPB_RR_SYS/ADC/UTC1-2.channel_0.output.schema.data.rawData'])
    #print(evt['SPB_DET_AGIPD1M-1/DET/9CH0:xtdf']['image.data'])
    #print(evt['SPB_DET_AGIP1M-1/DET/STACKED:xtdf']['image.data'])

    #det = evt['photonPixelDetectors']['AGIPD01'].data
    #det = evt['SPB_DET_AGIP1M-1/DET/STACKED:xtdf']['image.data']
    #print(evt['SPB_DET_AGIP1M-1/DET/STACKED:xtdf'].keys())
    #print(det[0,0,:,:])
    #module = add_record(evt["analysis"], "analysis", "single", det[10,2,:,:])
    
    #print(module.data.sum())

    '''
    Assemble modules and plot
    '''
    assem, centre = geom.position_modules_fast(evt['photonPixelDetectors']['AGIPD Stacked'].data)
    #print(centre)
    # applying the mask on the assebled detector image
    # assem[:, mask] = np.nan
    ai = azimuthalIntegrator.AzimuthalIntegrator(dist = detector_distance,
                                   poni1 = centre[0] * pixel_size,
                                   poni2 = centre[1] * pixel_size,
                                   pixel1 = pixel_size,
                                   pixel2 = pixel_size,
                                   rot1 = 0, rot2=0, rot3=0,
                                   wavelength = wavelength)
    npt = 128

    assem[:,assemask] = np.nan
    Q, i = ai.integrate1d(assem[10],
                           npt,
                           method = "BBox",
                           #mask = (assemask | (assem[10] = 0)),
                           radial_range = (0.05, 2),
                           correctSolidAngle = True,
                           polarization_factor = 1,
                           unit = "q_A^-1")
    
    print(Q.shape)
    print(i.shape)
    
    assem[np.isnan(assem)] = 0 
    assem_rec = add_record(evt["analysis"], "analysis", "Assem Image", assem[10,::-1,::-1])                    
    plotting.image.plotImage(assem_rec, history=10, log = True)
    
    '''
    
    '''
    
    #print(assem.shape)
    #det_arr[module_numbers] = mods[:,ind]                                                                           
    #assem = geom.position_modules_fast(det_arr)[0][::-1,::-1]                                                       
    #assem[np.isnan(assem)] = -1
    #brightest_hit = add_record(evt['analysis'], 'analysis', 'Random hit', assem)
    #random_image = add_record(evt["analysis"], "analysis", "Random Image", module.data[10,0])                       
    #print(assem.shape)
    
