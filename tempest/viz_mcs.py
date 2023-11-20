import os
import glob
import sys
import yaml

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tempest import casestudy
from tempest import grid
from tempest import joint_distrib
from tempest import handler


if __name__ == '__main__':
    settings_path = 'settings/tropics.yaml'
    # Instantiate CaseStudy by passing the settings. 
    # Should also create appropriate directories
    hdlr = handler.Handler(settings_path)
    cs = casestudy.CaseStudy(hdlr, verbose = True)
    gr = grid.Grid(hdlr, verbose = False, overwrite = False)
    # jd = joint_distrib.JointDistribution(gr)
