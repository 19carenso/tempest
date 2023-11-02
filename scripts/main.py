import os
import glob
import sys
import yaml

from tempest import casestudy
from tempest import grid
from tempest import handler

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
    gr.compute_funcs_for_var_id("Prec") ## by default calls var_id = 'Prec'