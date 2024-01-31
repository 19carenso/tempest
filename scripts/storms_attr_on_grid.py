import os
import glob
import sys
import yaml

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from tempest import casestudy
from tempest import grid
from tempest import storm_tracker
from tempest import joint_distrib
from tempest import handler
from tempest.plots.hist import simple_hist
settings_path = 'settings/tropics_20d.yaml'

workdir=os.getcwd()
print(workdir)
sys.path.append(workdir)

# Instantiate CaseStudy by passing the settings. 
# Should also create appropriate directories
hdlr = handler.Handler(settings_path) 
cs = casestudy.CaseStudy(hdlr, overwrite = False ,verbose = True)
gr = grid.Grid(cs, fast = True, overwrite = False, verbose_steps = True, verbose = True) 
st = storm_tracker.StormTracker(gr, overwrite_storms = False) #overwrite_storms = True takes 200G of memory, overwrite = True is super long so job-only
jd = joint_distrib.JointDistribution(gr, storm_tracker=st, var_id_1 = "mean_unweighted_Prec", var_id_2 = "mean_Prec_cond_50", nd=5, overwrite = False)

for var_name in st.ds_storms.data_vars: ## here we simply do all, but later we could select the ones not already in ds MCS_label
    if st.ds_storms[var_name].dims == ('label',) and var_name != "vavg":
        jd.add_mcs_var_from_labels(var_name)
        print(var_name, " added to MCS_label ds ! ")
