import plotting.image
import analysis.agipd
import analysis.event
from backend import add_record
import numpy as np
import h5py as h5
import extra_geom

state = {}
state['Facility'] = 'EuXFEL'
state['EuXFEL/DataSource'] = 'tcp://10.253.0.74:55777'

def onEvent(evt):
    #print(analysis.event.printNativeKeys(evt))
    #print(evt['photonPixelDetectors'].keys())
    #print(evt['SPB_DET_AGIPD1M-1/DET/9CH0:xtdf']['image.data'])
    #print(evt['SPB_DET_AGIP1M-1/DET/STACKED:xtdf']['image.data'])

    #det = evt['photonPixelDetectors']['AGIPD01'].data
    det = evt['SPB_DET_AGIP1M-1/DET/STACKED:xtdf']['image.data']
    print(evt['SPB_DET_AGIP1M-1/DET/STACKED:xtdf'].keys())
    #print(det[0,0,:,:])
    module = add_record(evt["analysis"], "analysis", "single", det[10,2,:,:])
    module.data[np.isnan(module.data)] = 0
    print(module.data.sum())
    #geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom('/home/alfredo/p003046/usr/Shared/alfredo/xfel3046/geometry/agipd_2995_v04.geom')
    #assem, centre = geom.position_modules_fast(np.empty(geom.expected_data_shape))

    #det_arr[module_numbers] = mods[:,ind]                                                                           
    #assem = geom.position_modules_fast(det_arr)[0][::-1,::-1]                                                       
    #assem[np.isnan(assem)] = -1
    #brightest_hit = add_record(evt['analysis'], 'analysis', 'Random hit', assem)
    #random_image = add_record(evt["analysis"], "analysis", "Random Image", module.data[10,0])                       
    #print(assem.shape)
    plotting.image.plotImage(module, history=10)
    
