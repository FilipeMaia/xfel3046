




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from scipy.ndimage import gaussian_filter



def plot_1d(line, **new_kwargs):

    kwargs ={   
                'fig':None,
                'axes':None,
                'log':False,
                'xlabel':'',
                'ylabel':'',
                'title':'',
                'norm':False,
                'subtmean':False,
                'blur':False,
                'color':'red',
                'save':'',
                'label':''
            }

    kwargs.update(new_kwargs)

    line = np.copy(line)

    # Make figure if none given
    if kwargs['fig'] is None:
        kwargs['fig'] = plt.figure()
        kwargs['axes'] = plt.gca()

    if kwargs['log']:
        line = np.log10(np.abs(line)+1)

    if kwargs['norm']:
        line -= line.min()
        line *= 1/line.max()

    if kwargs['subtmean']:
        mean = line.mean(axis=1)
        line -= mean

    if kwargs['blur']:
        line = gaussian_filter(line,  sigma=kwargs['blur'])

    kwargs['axes'].plot(line, color=kwargs['color'])

    kwargs['axes'].set_title(kwargs['title'])
    kwargs['axes'].set_xlabel(kwargs['xlabel'])
    kwargs['axes'].set_ylabel(kwargs['ylabel'])

    if kwargs['save'] is not None:
        plt.savefig(kwargs['save'])











def plot_2d(im, **new_kwargs):
    kwargs ={   'extent':None,
                'fig':None,
                'axes':None,
                'log':False,
                'cmap':'viridis',
                'cb':False,
                'vminmax':(None, None),
                'xlabel':'',
                'ylabel':'',
                'title':'',
                'origin':'lower',
                'ticks':True,
                'norm':False,
                'subtmean':False,
                'blur':False,
                'save':''
            }

    kwargs.update(new_kwargs)

    im = np.copy(im)


    # Make figure if none given
    if kwargs['fig'] is None:
        kwargs['fig'] = plt.figure()
        kwargs['axes'] = plt.gca()

    #Preprocessing: log intensity scale
    if kwargs['log']:
        im = np.log10(np.abs(im)+1)

    if kwargs['norm']:
        im -= im.min()
        im *= 1/im.max()

    if kwargs['subtmean']:
        means = im.mean(axis=1)
        xx, yy = np.meshgrid(np.ones(im.shape[1]), means)
        im -= yy

    if kwargs['blur']:
        im = gaussian_filter(im,  sigma=kwargs['blur'])





    kwargs['axes'].imshow(im, origin=kwargs['origin'], extent=kwargs['extent'], aspect='auto', cmap=kwargs['cmap'])

    # Postprocessing: colorscale limits
    vmin, vmax = kwargs['vminmax']
    if vmin is None:
        vmin = im.min()
    if vmax is None:
        vmax = im.max()


    if vmax==vmin:
        kwargs['cb']=False

    # Add colorbar
    if kwargs['cb']:
        norm = Normalize(vmin, vmax, clip=False)
        sm = ScalarMappable(cmap=kwargs['cmap'], norm=norm)
        sm._A = []
        kwargs['fig'].colorbar(sm, ax=kwargs['axes'])


    # Set colorlimits
    for axes_im in kwargs['axes'].get_images():
        axes_im.set_clim(vmin,vmax)

    kwargs['axes'].set_title(kwargs['title'])
    kwargs['axes'].set_xlabel(kwargs['xlabel'])
    kwargs['axes'].set_ylabel(kwargs['ylabel'])

    if kwargs['save'] is not None:
        plt.savefig(kwargs['save'])







