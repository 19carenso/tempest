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
cs = casestudy.CaseStudy(hdlr, overwrite = True ,verbose = False)
gr = grid.Grid(hdlr, fast = True, overwrite= True, verbose_steps = False, verbose = False)

if __name__ == '__main__':
    # print("Loaded Configuration:")
    # for key, value in settings.items():
    #     print(f"{key}: {value}")
    # print(gr.days_i_t_per_var_id["MCS_label"])
    gr.compute_funcs_for_var_id() ## by default calls var_id = 'Prec'
    # jd = joint_distrib.JointDistribution(gr)
    # jd.get_mcs_bin_fraction()