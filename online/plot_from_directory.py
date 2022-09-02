import plotting.image
import analysis.agipd
import analysis.event
from backend import add_record
import numpy as np
import h5py as h5
import extra_geom

state = {}
state['Facility'] = 'EuXFEL'
state['EventIsTrain'] = True 
state['EuXFEL/SelModule'] = 0 
state['EuXFEL/DataSource'] = 'tcp://max-exfl-display003.desy.de:55555'


def onEvent(evt):
    
    '''
    Print Processing rate
    '''
    analysis.event.printProcessingRate()
    
    # Collecting the detector info
    det = evt['photonPixelDetectors']['AGIPD01'].data
    module = add_record(evt["analysis"], "analysis", "single", det[0,:,:])
    geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom('/home/awollter/p003046/usr/Shared/awollter/xfel3046/geometry/agipd_september_2022_v03.geom')
    
    # Assemble the modules into a single image
    assem, centre = geom.position_modules_fast(np.empty(geom.expected_data_shape))

    #det_arr[module_numbers] = mods[:,ind]                                                                           
    #assem = geom.position_modules_fast(det_arr)[0][::-1,::-1]                                                       
    #assem[np.isnan(assem)] = -1
    #brightest_hit = add_record(evt['analysis'], 'analysis', 'Random hit', assem)
    #random_image = add_record(evt["analysis"], "analysis", "Random Image", module.data[10,0])                       
    #print(assem.shape)
    #plotting.image.plotImage(assem, history=10)
