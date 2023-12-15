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
settings_path = 'settings/intro.yaml'

hdlr = handler.Handler(settings_path)
cs = casestudy.CaseStudy(hdlr, overwrite = False ,verbose = False)
gr = grid.Grid(cs, fast = True, overwrite= False, verbose_steps = False, verbose = False)
st = storm_tracker.StormTracker(gr, overwrite = True) #overwrite = True is super long
jd = joint_distrib.JointDistribution(gr, st)


print(st.storms[0].growth_rate)

st.save_storms()

st =  storm_tracker.StormTracker(gr, overwrite = False) #overwrite = True is super long

print(st.storms[0].growth_rate)
