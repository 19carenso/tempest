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
hdlr = handler.Handler(settings_path) ## CAREFUL ITS SET TO FALSE
cs = casestudy.CaseStudy(hdlr, overwrite = True ,verbose = True)
gr = grid.Grid(cs, fast = True, overwrite = True, verbose_steps = True, verbose = True)

if __name__ == '__main__':

    # gr.regrid_funcs_and_save_for_day("16-08-11", "Prec") carefull it corrupts the file 

    # gr.compute_funcs_for_var_id("MCS_label_Tb_Feng", overwrite_var_id=True)
    # gr.compute_funcs_for_var_id("MCS_label", overwrite_var_id=True)


    # gr.compute_funcs_for_var_id("Prec", overwrite_var_id=True)
    gr.compute_funcs_for_var_id("MCS_label_Tb_Feng", overwrite_var_id=True)

    # jd = joint_distrib.JointDistribution(gr, nd= 5, overwrite = False, storm_tracking = True)

    # gr.compute_funcs_for_var_id("OM500") ## by default calls var_id = 'Prec'
    # gr.compute_funcs_for_var_id("OM700") ## by default calls var_id = 'Prec'
    # gr.compute_funcs_for_var_id("OM850") ## by default calls var_id = 'Prec'

    # gr.compute_funcs_for_var_id("RH500") ## by default calls var_id = 'Prec'
    # gr.compute_funcs_for_var_id("RH700") ## by default calls var_id = 'Prec'
    # gr.compute_funcs_for_var_id("RH850") ## by default calls var_id = 'Prec'
    
    # jd.get_mcs_bin_fraction()   