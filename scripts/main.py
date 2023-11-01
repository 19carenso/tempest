import os
import glob
import sys
import yaml

from tempest import casestudy
from tempest import grid

settings_path = 'settings/tropics.yaml'

workdir=os.getcwd()
print(workdir)
sys.path.append(workdir)

with open(settings_path, 'r') as file:
    settings = yaml.safe_load(file)

# Instantiate CaseStudy by passing the settings. 
# Should also create appropriate directories
cs = casestudy.CaseStudy(settings, overwrite = True ,verbose = False)
gr = grid.Grid(settings, fast = True, overwrite= True, verbose_steps = True, verbose = False)

if __name__ == '__main__':
    # print("Loaded Configuration:")
    # for key, value in settings.items():
    #     print(f"{key}: {value}")
    gr.compute_funcs_for_var_id("Prec") ## by default calls var_id = 'Prec'