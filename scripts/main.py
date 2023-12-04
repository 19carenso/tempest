import os
import glob
import sys
import yaml

from tempest import casestudy
from tempest import grid
from tempest import handler
from tempest import joint_distrib

settings_path = 'settings/tropics.yaml'

workdir=os.getcwd()
print(workdir)
sys.path.append(workdir)

# Instantiate CaseStudy by passing the settings. 
# Should also create appropriate directories
hdlr = handler.Handler(settings_path)
cs = casestudy.CaseStudy(hdlr, overwrite = False ,verbose = True)
gr = grid.Grid(cs, fast = True, overwrite= False, verbose_steps = True, verbose = True)

if __name__ == '__main__':

    gr.compute_funcs_for_var_id("QV_sat_2d")

    # jd = joint_distrib.JointDistribution(gr, nd= 5, overwrite = False, storm_tracking = True)

    # gr.compute_funcs_for_var_id("OM500") ## by default calls var_id = 'Prec'
    # gr.compute_funcs_for_var_id("OM700") ## by default calls var_id = 'Prec'
    # gr.compute_funcs_for_var_id("OM850") ## by default calls var_id = 'Prec'

    # gr.compute_funcs_for_var_id("RH500") ## by default calls var_id = 'Prec'
    # gr.compute_funcs_for_var_id("RH700") ## by default calls var_id = 'Prec'
    # gr.compute_funcs_for_var_id("RH850") ## by default calls var_id = 'Prec'
    
    # jd.get_mcs_bin_fraction() 