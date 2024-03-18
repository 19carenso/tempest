import os
import glob
import sys
import yaml

from tempest import casestudy
from tempest import grid
from tempest import handler
from tempest import joint_distrib

settings_path = 'settings/tropics_20d.yaml'

workdir=os.getcwd()
print(workdir)
sys.path.append(workdir)

# Instantiate CaseStudy by passing the settings. 
# Should also create appropriate directories
hdlr = handler.Handler(settings_path)
cs = casestudy.CaseStudy(hdlr, overwrite = False ,verbose = True)
gr = grid.Grid(cs, fast = True, overwrite= True, verbose_steps = False, verbose = False)
# jd.prec['mean_Prec_cond_50'] = jd.prec["cond_alpha_50_Prec"] / jd.prec["Sigma_cond_alpha_50_Prec"]
# jd.prec.to_netcdf("/scratchx/mcarenso/tempest/DYAMOND_SAM_post_20_days_Tropics/prec2.nc")

if __name__ == '__main__':
    # gr.regrid_funcs_and_save_for_day("16-08-11", "Prec") # carefull it corrupts the file 
    # gr.compute_funcs_for_var_id("MCS_label", overwrite_var_id=True)
    gr.compute_funcs_for_var_id("Prec", overwrite_var_id=True)
  
    # gr.compute_funcs_for_var_id("MCS_label", overwrite_var_id=True)
    # gr.compute_funcs_for_var_id("Conv_MCS_label", overwrite_var_id=True)

    # jd = joint_distrib.JointDistribution(gr, nd= 5, overwrite = False, storm_tracking = True)
    
    # jd.get_mcs_bin_fraction()   
