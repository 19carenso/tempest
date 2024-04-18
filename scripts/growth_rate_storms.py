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
settings_path = 'settings/sam_summer_30d.yaml'

hdlr = handler.Handler(settings_path)
cs = casestudy.CaseStudy(hdlr, overwrite = False ,verbose = False)
gr = grid.Grid(cs, fast = True, overwrite= False, verbose_steps = False, verbose = False)
st = storm_tracker.StormTracker(gr, label_var_id = "MCS_Feng", overwrite_storms = True, overwrite = True) #overwrite = True is super long
# jd = joint_distrib.JointDistribution(gr, st)

output_file_path = st.file_storms  
ds_storms = xr.open_dataset(output_file_path)


## Here maybe print data on storms before celaning growth rates
print(np.sum(~np.isnan(ds_storms.growth_rate)))
print(np.sum(ds_storms.r_squared < 0.8))
print(np.sum(ds_storms.r_squared < 0.85))
print(np.sum(ds_storms.r_squared < 0.9))
print(np.sum(ds_storms.r_squared < 0.95))

ds_storms['norm_growth_rate'] = ds_storms['growth_rate'] / ds_storms['s_max']
ds_storms['growth_rate'] = ds_storms['growth_rate'].where((ds_storms['norm_growth_rate'] >= 0) & (ds_storms['norm_growth_rate'] <= 1) & (ds_storms['r_squared'] >= 0.8) , np.nan)
ds_storms['s_max'] = ds_storms['s_max'].where((ds_storms['norm_growth_rate'] >= 0) & (ds_storms['norm_growth_rate'] <= 1) & (ds_storms['r_squared'] >= 0.8) , np.nan)
ds_storms['norm_growth_rate'] = ds_storms['norm_growth_rate'].where((ds_storms['norm_growth_rate'] >= 0) & (ds_storms['norm_growth_rate'] <= 1), np.nan)

os.remove(output_file_path)
ds_storms.to_netcdf(output_file_path)

print(st.ds_storms.isel(label = 0).growth_rate)

st =  storm_tracker.StormTracker(gr, overwrite_storms = False, overwrite = False) 

print(st.storms[0].growth_rate)