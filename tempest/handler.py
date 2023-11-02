import pandas as pd
import os 
import sys
import re
import yaml 

import gc
import xarray as xr


class Handler():
    def __init__(self, settings_path):
        with open(settings_path, 'r') as file:
            self.settings = yaml.safe_load(file)
