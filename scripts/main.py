import os
import glob
import sys
import yaml

from tempest import casestudy
from tempest import grid
from tempest import handler
from tempest import joint_distrib
from tempest import storm_tracker

settings_path = "settings/screamv1_summer_30d.yaml"

workdir=os.getcwd()
print(workdir)
sys.path.append(workdir)

# Instantiate CaseStudy by passing the settings. 
# Should also create appropriate directories
hdlr = handler.Handler(settings_path)
overwrite = True
cs = casestudy.CaseStudy(hdlr, overwrite = overwrite ,verbose = True)
gr = grid.Grid(cs, fast = True, overwrite= overwrite, verbose_steps = False, verbose = False)
# jd.prec['mean_Prec_cond_50'] = jd.prec["cond_alpha_50_Prec"] / jd.prec["Sigma_cond_alpha_50_Prec"]
# jd.prec.to_netcdf("/scratchx/mcarenso/tempest/DYAMOND_SAM_post_20_days_Tropics/prec2.nc")

if __name__ == '__main__':
    # # 1st batch 
    gr.compute_funcs_for_var_id("Prec", overwrite_var_id=True)
    gr.compute_funcs_for_var_id("MCS_label", overwrite_var_id=True)
    gr.compute_funcs_for_var_id("MCS_Feng", overwrite_var_id=True)
    
    print("Storm tracking clouds")
    st = storm_tracker.StormTracker(gr, label_var_id = "MCS_label", overwrite_storms = True, overwrite = False)
    print("Storm tracking mcs")
    st = storm_tracker.StormTracker(gr, label_var_id = "MCS_Feng", overwrite_storms = True, overwrite = False)

    # # 2nd batch
    gr.compute_funcs_for_var_id("vDCS", overwrite_var_id=True)
    gr.compute_funcs_for_var_id("MCS_cond_Prec_15", overwrite_var_id = True)
    gr.compute_funcs_for_var_id("vDCS_cond_Prec_15")
    gr.compute_funcs_for_var_id("clouds_cond_Prec_15")
    st = storm_tracker.StormTracker(gr, label_var_id = "vDCS", overwrite_storms = True, overwrite = False)

    cloud_types = ["clouds_cond_prec_15", "vdcs_cond_prec_15", "mcs_cond_prec_15"]
    for cloud_type in cloud_types:
        gr.build_cloud_intersect(cloud_type, coverage_threshold=0.5)


    # gr.compute_funcs_for_var_id("Prec_lowRes", overwrite_var_id=True)

    # gr.compute_funcs_for_var_id("sst", overwrite_var_id=True)

    # gr.compute_funcs_for_var_id("T2mm")

    # gr.compute_funcs_for_var_id("MCS_label", overwrite_var_id=True)

    # gr.compute_funcs_for_var_id("MCS_Feng", overwrite_var_id=True)
    
    # gr.compute_funcs_for_var_id("Conv_MCS_label", overwrite_var_id=True)

    # gr.compute_funcs_for_var_id("vDCS")
    # st = storm_tracker.StormTracker(gr, label_var_id = "vDCS", overwrite_storms = True, overwrite = False)

    # gr.compute_funcs_for_var_id("MCS_cond_Prec_15")
    # gr.compute_funcs_for_var_id("vDCS_cond_Prec_15")
    # gr.compute_funcs_for_var_id("clouds_cond_Prec_15")
    # gr.compute_funcs_for_var_id("sliding_clouds_cond_Prec_15")
    # gr.compute_funcs_for_var_id("sliding_mcs_cond_Prec_15")

    # gr.compute_funcs_for_var_id("vDCS_Strat_Prec",  overwrite_var_id=True)
    # gr.compute_funcs_for_var_id("vDCS_Conv_Prec", overwrite_var_id=True)
    # jd = joint_distrib.JointDistribution(gr, nd= 5, overwrite = False, storm_tracking = True)
    # jd.get_mcs_bin_fraction()   
