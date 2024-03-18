import os
import glob
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tempest import casestudy
from tempest import grid
from tempest import storm_tracker
from tempest import joint_distrib
from tempest import handler
settings_path = 'settings/tropics_20d.yaml'

hdlr = handler.Handler(settings_path)
cs = casestudy.CaseStudy(hdlr, overwrite = False ,verbose = False)
gr = grid.Grid(cs, fast = True, overwrite= False, verbose_steps = False, verbose = False)
st = storm_tracker.StormTracker(gr, label_var_id = "MCS_label", overwrite_storms = True, overwrite = True) #overwrite = True is super long
jd = joint_distrib.JointDistribution(gr, st)

print(st.ds_storms.isel(label = 0).growth_rate)

st =  storm_tracker.StormTracker(gr, overwrite_storms = False, overwrite = False) 

print(st.storms[0].growth_rate)
