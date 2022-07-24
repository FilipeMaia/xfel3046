## Rescaling Q

import numpy as np

def rescaleQ(Q_old,Energy,percentage):


    h           = 4.136e-15         #   Planck constant in eVs 
    c           = 3e18            #   Speed of light in Å/s 
    
    l   = h*c/Energy       # wavelength in Å. 
    
    
    if max(Q_old*l)>np.sqrt(8)*np.pi:
        print('max value of Q may not exceed '+str(np.sqrt(8)*np.pi/l)+', for Q values above that the scale values are faulty')
    
    
   
        
    var1        = (l*Q_old/np.pi)**2# helping variables to simplify expression
    var2        = (100/(100+percentage))**2*(1./(1-var1/8)**2-1);
    Q_new       = np.real((4*np.pi/l)*np.sqrt(1/2*(1-(1/(np.sqrt(1+var2))))));
  
    
    # factor      = Q_new/Q_old    #  Q_new = factor.*Q_old.
    return Q_new

