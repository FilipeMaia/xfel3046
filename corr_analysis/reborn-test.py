

import reborn
from reborn import detector, data

from reborn.external import crystfel
from reborn.viewers.mplviews import view_pad_data
# geomfile = data.cspad_geom_file
# pads = crystfel.geometry_file_to_pad_geometry_list(geomfile)


# dat = pads.solid_angles()



geom = reborn.external.crystfel.load_crystfel_geometry('./p3046_manual_refined_geoass_run10.geom')


print(geom)
