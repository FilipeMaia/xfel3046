

import reborn
from reborn import detector, data

from reborn.external import crystfel
from reborn.viewers.mplviews import view_pad_data
from reborn.external.euxfel import EuXFELFrameGetter
# geomfile = data.cspad_geom_file
# pads = crystfel.geometry_file_to_pad_geometry_list(geomfile)


# dat = pads.solid_angles()


prop_num = 3046

# geom = reborn.external.crystfel.load_crystfel_geometry('./p3046_manual_refined_geoass_run10.geom')
# geom = reborn.external.crystfel.load_crystfel_geometry('/gpfs/exfel/exp/SPB/202202/p003046/scratch/rkirian/xfel3046/geometry/p3046_manual_refined_geoass_run10.geom')
# geom = reborn.external.crystfel.load_crystfel_geometry('/gpfs/exfel/exp/SPB/202202/p003046/scratch/rkirian/xfel3046/geometry/p3046_manual_refined_geoass_run10.geom')
# geom = reborn.external.crystfel.load_crystfel_geometry('./august.geom')

geom = crystfel.geometry_file_to_pad_geometry_list('/gpfs/exfel/exp/SPB/202202/p003046/scratch/rkirian/xfel3046/geometry/p3046_manual_refined_geoass_run10.geom')
fg = EuXFELFrameGetter(experiment_id=prop_num, run_id = 48, geom = geom)

fg.view()



print(geom)
