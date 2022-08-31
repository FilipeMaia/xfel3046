import plotting.image
import analysis.agipd
import analysis.event
import pyFAI 
from backend import add_record

state = {}
state['Facility'] = 'EuXFEL'
state['EventIsTrain'] = True 
state['EuXFEL/SelModule'] = 0 
state['EuXFEL/DataSource'] = 'tcp://10.253.0.74:55777'
#state['EuXFEL/DataSource'] = 'tcp://max-exfl-display001.desy.de:1337'

def onEvent(evt):

    #analysis.event.printKeys(evt)
    analysis.event.printNativeKeys(evt)
    analysis.event.printProcessingRate()
    
    data = evt._evt['SPB_DET_AGIP1M-1/DET/STACKED:xtdf']['image.data']

    
    print(data.shape)
    record = add_record(evt['analysis'],'analysis', 'AGIPD', data[10,10,:,:])
    plotting.image.plotImage(record) 
