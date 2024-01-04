import os
import glob
import sys
import yaml
import gc as gc
import numpy as np
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
cs = casestudy.CaseStudy(hdlr, overwrite = True ,verbose = True)
gr = grid.Grid(cs, fast = True, overwrite = True, verbose_steps = True, verbose = True)

if __name__ == '__main__':
    daily_mean_prec = []
    for date in list(cs.days_i_t_per_var_id["Precac"].keys()):
        print("\n", date)
        for i_t in list(cs.days_i_t_per_var_id["Precac"][date]):
            print(i_t, end ='.')
            data = hdlr.load_var(gr, "Precac", i_t)
            daily_mean_prec.append(np.mean(data.values))
            del data
            gc.collect()
    dir_out= "/home/mcarenso/code/tempest/output"
    file = 'mean_precac_tropics.npy'
    # Save the array as a NumPy file
    np.save(os.path.join(dir_out, file), np.array(daily_mean_prec))
    print(f"Mean precipitation saved to {os.path.join(dir_out, file)}")
    # gr.compute_funcs_for_var_id("MCS_label", overwrite_var_id=True)


    # gr.compute_funcs_for_var_id("MCS_label", overwrite_var_id=True)
    # gr.compute_funcs_for_var_id("Prec_t_minus_1", overwrite_var_id=True)

    # jd = joint_distrib.JointDistribution(gr, nd= 5, overwrite = False, storm_tracking = True)

    # gr.compute_funcs_for_var_id("OM500") ## by default calls var_id = 'Prec'
    # gr.compute_funcs_for_var_id("OM700") ## by default calls var_id = 'Prec'
    # gr.compute_funcs_for_var_id("OM850") ## by default calls var_id = 'Prec'

    # gr.compute_funcs_for_var_id("RH500") ## by default calls var_id = 'Prec'
    # gr.compute_funcs_for_var_id("RH700") ## by default calls var_id = 'Prec'
    # gr.compute_funcs_for_var_id("RH850") ## by default calls var_id = 'Prec'
    
    # jd.get_mcs_bin_fraction() 