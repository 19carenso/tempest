import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Seaborn enhances the aesthetics of matplotlib plots
import pandas as pd
import xarray as xr

from tempest import casestudy
from tempest import grid
from tempest import storm_tracker
from tempest import joint_distrib
from tempest import handler
from tempest.plots.hist import simple_hist
settings_path = 'settings/tropics_20d.yaml'


## Load handler and pass settings
hdlr = handler.Handler(settings_path)

## Initiate variables
cs = casestudy.CaseStudy(hdlr, overwrite = False ,verbose = False)

## Make grid and load data on grid 
gr = grid.Grid(cs, fast = True, overwrite= False, verbose_steps = False, verbose = False)

## Get ocean mask 
lm = gr.get_landmask()
ocean = lm.Landmask == 0
ocean = ocean.values[:,:,np.newaxis]

## Load storms 
st = storm_tracker.StormTracker(gr, label_var_id = "MCS_label", overwrite = False)

##
jd = joint_distrib.JointDistribution(gr, st, var_id_1 = "mean_unweighted_Prec", var_id_2 = "cond_alpha_50_Prec", nd=5, overwrite = True, dist_mask = ocean)
jd.make_mask()

import os 
import subprocess
import pickle

type = 'cg'

if type == 'km':
    speed_mask = np.outer(jd.dist1.ranks >= 0, jd.dist2.ranks >= 99)
    mask_yxt = jd.get_mask_yxt_from_mask_jdist(np.logical_and(jd.mask_coloc_c_90, speed_mask))

elif type == 'cg':
    speed_mask = np.outer(jd.dist1.ranks >= 99, jd.dist2.ranks >= 0)
    mask_yxt = jd.get_mask_yxt_from_mask_jdist(np.logical_and(jd.mask_coloc_ac_90, speed_mask))

thresholds = jd.prec.Treshold_cond_alpha_50_Prec
var_id = "QI"

path_data_in = gr.settings["DIR_DATA_3D_IN"]
prec_with_ditvi = "Prec_with_QI"
ditvi = cs._update_ditvi(prec_with_ditvi, ["Prec", var_id])[prec_with_ditvi]

indices = np.where(np.logical_and(mask_yxt, ocean))
# indices = np.where(mask_yxt)

regrouped_indices = {}
QI_vert_profiles = []
pressure_coordinates = []
grid_locations = []
print('computing ', type)

for Y, X, D in zip(*indices):
    if D not in regrouped_indices:
        regrouped_indices[D] = []
    regrouped_indices[D].append((Y,X))
# Create temp files
temp = os.path.join(hdlr.settings["DIR_TEMPDATA"], cs.name)
temp_var = os.path.join(temp, var_id)
if not os.path.exists(temp_var):
    os.makedirs(temp_var)
temp_file = os.path.join(temp_var, f"temp_extract.nc")
output_file = os.path.join(temp_var, f"avg_vert_profile.nc")
temp_mask = os.path.join(temp_var, "temp_mask.nc")

for D in np.sort(list(regrouped_indices.keys())):
    global_tuples = regrouped_indices[D]
    day = jd.prec.days.values[D]
    for i_t in ditvi[day]:
        print("computing", day, i_t)
        root = hdlr.get_rootname_from_i_t(i_t)
        prec = hdlr.load_prec(gr, i_t)
        cloud = hdlr.load_seg(gr, i_t)
        filename_var = root+f"_{var_id}.nc"
        filepath_var = os.path.join(path_data_in, filename_var)

        for i_tuple, tuple in enumerate(global_tuples):
            print(100*i_tuple/len(global_tuples), '%')
        # print(Y, X, D, weight[Y,X,D].values)
            Y, X = tuple
            lat_start, lat_stop = gr.slices_i_lat[Y].start, gr.slices_i_lat[Y].stop
            lon_start, lon_stop = gr.slices_j_lon[Y][X].start, gr.slices_j_lon[Y][X].stop

            local_prec = prec[gr.slices_i_lat[Y], gr.slices_j_lon[Y][X]]
            local_cloud = ~np.isnan(cloud[0, gr.slices_i_lat[Y], gr.slices_j_lon[Y][X]])

            local_prec_mask = local_prec > thresholds[Y, X, D]
            local_where_conv = np.logical_and(local_prec_mask.values, local_cloud.values)

            if np.any(local_where_conv):
                local_lat, local_lon = local_prec.lat, local_prec.lon

                # conv_dataarray = xr.DataArray(local_where_conv, coords=[('lat', local_lat.data), ('lon', local_lon.data)])
                # conv_dataarray.to_netcdf(temp_mask)

                ncks_command = f"ncks -O -d lon,{lon_start},{lon_stop-1} -d lat,{lat_start},{lat_stop-1} -d time,0 {filepath_var} {temp_file}"
                subprocess.run(ncks_command, check = True, shell=True)

                # ncwa_command = f"ncwa -O -a lat,lon -w {temp_mask} -y avg {temp_file} {output_file}"
                # subprocess.run(ncwa_command, check = True, shell=True)

                broadcast_mask = np.broadcast_to(local_where_conv, (1, 74, np.shape(local_where_conv)[0], np.shape(local_where_conv)[1]))
                var = xr.open_dataset(temp_file)

                if np.shape(var.values) != np.shape(broadcast_mask): # this should not be but it is in cg case.. nothing yet for km
                    print('Y : ', Y, 'X : ', X, 'D : ', D)
                    print(lon_start, lon_stop-1)
                    print(lat_start, lat_stop-1)

                    continue

                profiles = var.QI.where(broadcast_mask)
                qi_profile = np.mean(profiles, axis = (0, 2, 3))
                pressure_coord = var.p
                QI_vert_profiles.append(qi_profile)
                pressure_coordinates.append(pressure_coord) 
                grid_locations.append((Y, X, D))

dir_out = gr.settings["DIR_DATA_OUT"]
path_dir_out = os.path.join(os.path.join(os.getcwd(), dir_out))

file_1_out = os.path.join(path_dir_out, f"mostly_{type}_QI_vert_profiles.pkl")
file_2_out = os.path.join(path_dir_out, f"mostly_{type}_pressure_coordinates.pkl")
file_3_out = os.path.join(path_dir_out, f"mostly_{type}_grid_locations.pkl")


with open(file_1_out, 'wb') as f: 
    pickle.dump(QI_vert_profiles, f)

with open(file_2_out, 'wb') as f: 
    pickle.dump(pressure_coordinates, f)

with open(file_3_out, 'wb') as f: 
    pickle.dump(grid_locations, f)