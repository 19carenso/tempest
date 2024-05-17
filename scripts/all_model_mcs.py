import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns  # Seaborn enhances the aesthetics of matplotlib plots

import pandas as pd
import xarray as xr
import seaborn as sns
import warnings
import logging
# sns.set()
import os 


from tempest import casestudy
from tempest import grid
from tempest import storm_tracker
from tempest import joint_distrib
from tempest import handler
from tempest.plots.hist import simple_hist


workdir=os.getcwd()
print(workdir)
sys.path.append(workdir)

# settings_path = 'settings/sam_4km_30min_30d.yaml'
settings_paths = [
                    "settings/arpege_summer_30d.yaml",
                    "settings/fv3_summer_30d.yaml",
                    "settings/ifs_summer_30d.yaml",
                    "settings/mpas_summer_30d.yaml",
                    "settings/nicam_summer_30d.yaml",
                    "settings/obs_summer_30d.yaml",
                    "settings/sam_summer_30d.yaml",
                    "settings/um_summer_30d.yaml",
                    "settings/sam_4km_30min_30d.yaml"
      ]


hdlrs = [handler.Handler(settings_path) for settings_path in settings_paths]
css = [casestudy.CaseStudy(hdlr, overwrite = False ,verbose = False) for hdlr in hdlrs]
grs = [grid.Grid(cs, fast = True, overwrite= False, verbose_steps = False, verbose = False) for cs in css]
jds = [joint_distrib.JointDistribution(gr, None, var_id_1 = "mean_unweighted_Prec", var_id_2 = "cond_alpha_85_Prec", 
        nbpd = 20,  nd=5, overwrite = True, dist_mask = False) for gr in grs]

lm = grs[0].get_landmask()
ocean = lm.Landmask == 0
ocean = ocean.values[:,:,np.newaxis]


for gr in grs:
    print("Model running is ", gr.settings["MODEL"])
    try: 
        print("building storms from MCS label")
        st = storm_tracker.StormTracker(gr, label_var_id="MCS_label", overwrite_storms=True, overwrite=False, verbose=True)
    except Exception as e:
        print("An error occurred:", e)

    try: 
        print("building storms from MCS Feng")
        st = storm_tracker.StormTracker(gr, label_var_id="MCS_Feng", overwrite_storms=True, overwrite=False, verbose=True)
    except Exception as e:
        print("An error occurred:", e)