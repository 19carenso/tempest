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
    mask_yxt = jd.get_mask_yxt_from_mask_jdist(jd.mask_coloc_c_90)
elif type == 'cg':
    mask_yxt = jd.get_mask_yxt_from_mask_jdist(jd.mask_coloc_ac_90)

thresholds = jd.prec.Treshold_cond_alpha_50_Prec
var_id = "QI"

path_data_in = gr.settings["DIR_DATA_3D_IN"]
prec_with_ditvi = "Prec_with_QI"
ditvi = cs._update_ditvi(prec_with_ditvi, ["Prec", var_id])[prec_with_ditvi]

indices = np.where(np.logical_and(mask_yxt, ocean))

regrouped_indices = {}
QI_vert_profiles = []
corresponding_data = []

for Y, X, D in zip(*indices):
    if D not in regrouped_indices:
        regrouped_indices[D] = []
    regrouped_indices[D].append((Y,X))

for D in np.sort(list(regrouped_indices.keys())):
    global_tuples = regrouped_indices[D]
    day = jd.prec.days.values[D]
    for i_t in ditvi[day]:
        root = hdlr.get_rootname_from_i_t(i_t)
        prec = hdlr.load_prec(gr, i_t)
        cloud = hdlr.load_seg(gr, i_t)
        for tuple in global_tuples:
        # print(Y, X, D, weight[Y,X,D].values)
            Y, X = tuple
            lat_start, lat_stop = gr.slices_i_lat[Y].start, gr.slices_i_lat[Y].stop
            lon_start, lon_stop = gr.slices_j_lon[Y][X].start, gr.slices_j_lon[Y][X].stop

            local_prec = prec[gr.slices_i_lat[Y], gr.slices_j_lon[Y][X]]
            local_cloud = ~np.isnan(cloud[gr.slices_i_lat[Y], gr.slices_j_lon[Y][X]])
            local_where_conv = local_prec > thresholds[Y, X, D]

            if np.any(local_where_conv):
                for lat, lon in zip(*np.where(local_where_conv)):
                    true_lat = gr.lat[lat_start+lat].item()
                    true_lon = gr.lon[lon_start+lon].item()
                    print(prec.sel(lat = true_lat, lon = true_lon).item())

                    filename_var = root+f"_{var_id}.nc"
                    filepath_var = os.path.join(path_data_in, filename_var)
                    temp = os.path.join(hdlr.settings["DIR_TEMPDATA"], cs.name)
                    temp_var = os.path.join(temp, var_id)
                    if not os.path.exists(temp_var):
                        os.makedirs(temp_var)
                    temp_file = os.path.join(temp_var, f"vertical_profile.nc")

                    ncks_command = f"ncks -O -d lon,{true_lon} -d lat,{true_lat} -d time,{0} {filepath_var} {temp_file}"
                    subprocess.run(ncks_command, shell=True)
                    # old # var = xr.open_dataset(filepath_var).sel(lon=grid.casestudy.lon_slice,lat=grid.casestudy.lat_slice).isel(time=0, z=z) #, chunks = chunks)
                    var = xr.open_dataset(temp_file)
                    QI_vert_profiles.append(var)
                    corresponding_data.append((Y, X,  D))

dir_out = gr.settings["DIR_DATA_OUT"]
path_dir_out = os.path.join(os.path.join(os.getcwd(), dir_out))

file_1_out = os.path.join(path_dir_out, f"QI_vert_profiles_mostly_{type}.pkl")
file_2_out = os.path.join(path_dir_out, f"mostly_{type}_corresponding_data.pkl")

with open(file_1_out, 'wb') as f: 
    pickle.dump(QI_vert_profiles, f)

with open(file_2_out, 'wb') as f: 
    pickle.dump(corresponding_data, f)