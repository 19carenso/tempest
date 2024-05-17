import os
import glob
import sys
import yaml

from tempest import casestudy
from tempest import grid
from tempest import handler
from tempest import joint_distrib
from tempest import storm_tracker

# Read settings path from command line arguments
if len(sys.argv) < 2:
    print("Usage: python main.py <settings_path>")
    sys.exit(1)

settings_path = sys.argv[1]

workdir=os.getcwd()
print(workdir)
sys.path.append(workdir)

# Instantiate CaseStudy by passing the settings. 
# Should also create appropriate directories
hdlr = handler.Handler(settings_path)
cs = casestudy.CaseStudy(hdlr, overwrite = True ,verbose = True)
gr = grid.Grid(cs, fast = True, overwrite= True, verbose_steps = False, verbose = False)
# jd.prec['mean_Prec_cond_50'] = jd.prec["cond_alpha_50_Prec"] / jd.prec["Sigma_cond_alpha_50_Prec"]
# jd.prec.to_netcdf("/scratchx/mcarenso/tempest/DYAMOND_SAM_post_20_days_Tropics/prec2.nc")

if __name__ == '__main__':
    # 1st batch 
    # gr.compute_funcs_for_var_id("Prec", overwrite_var_id=True)
    # gr.compute_funcs_for_var_id("MCS_label", overwrite_var_id=True)
    # gr.compute_funcs_for_var_id("MCS_Feng", overwrite_var_id=True)
    gr.compute_funcs_for_var_id("sst", overwrite_var_id=True)

    # 2nd batch
    # gr.compute_funcs_for_var_id("vDCS", overwrite_var_id=True)
    # gr.compute_funcs_for_var_id("MCS_cond_Prec_15")
    # gr.compute_funcs_for_var_id("vDCS_cond_Prec_15")
    # gr.compute_funcs_for_var_id("clouds_cond_Prec_15")

    #


## post 1st batch 
# st = storm_tracker.StormTracker(gr, label_var_id = "MCS_label", overwrite_storms = True, overwrite = False)
# st = storm_tracker.StormTracker(gr, label_var_id = "MCS_Feng", overwrite_storms = True, overwrite = False)

## post 2nd batch
# st = storm_tracker.StormTracker(gr, label_var_id = "vDCS", overwrite_storms = True, overwrite = False)