import pandas as pd
import os 
import sys
import re
import yaml 

import gc
import xarray as xr

from importlib import import_module


settings_path = 'settings/settings.yaml'

with open(settings_path, 'r') as file:
    settings = yaml.safe_load(file)

## This function is specific to your REL_TABLE in settings
def load_rel_table(file_path):
    """
    Load a .csv file and return its contents as a pandas DataFrame.

    :param file_path: The path to the .csv file to be loaded.
    :type file_path: str

    :return: A pandas DataFrame containing the data from the .csv file.
    :rtype: pandas.DataFrame
    """
    # print(pd.__version__)
    # print(sys.executable)

    df = pd.read_csv(file_path)
    df.sort_values(by='UTC', ignore_index=True,inplace=True)
    return df

## This function is specific to your TIME_RANGE and files in DIR_DATA_IN

def extract_digit_after_sign(input_string):
    # Define a regular expression pattern to match the sign followed by a digit
    pattern = r'[+-]\d'

    # Search for the pattern in the input string
    match = re.search(pattern, input_string)

    if match:
        # Extract the digit after the sign
        digit = match.group(0)[1]
        return int(digit)
    else:
        return None
    
def get_timestamp_from_filename(filename):
    """
    input: the filename of the classical DYAMOND .nc data file "DYAMOND_9216x4608x74_7.5s_4km_4608_0000345840.U10m.2D.nc"
    output: 345840
    """
    timestamp_pattern = r'_(\d{10})\.\w+\.2D\.nc'
    match = re.search(timestamp_pattern, filename)
    if match : 
        timestamp = match.group(1)
        return timestamp
    else : return None

def get_rootname_from_i_t(i_t):
    string_timestamp = str(i_t * 240).zfill(10)
    result = f"DYAMOND_9216x4608x74_7.5s_4km_4608_"+string_timestamp
    return result


def load_seg(i_t,df):
    ## i_t is incorrectly passed there 
    full_path = '/'+df.iloc[i_t]['img_seg_path']
    
    if settings['DIR_TOOCANSEG_DYAMOND'] is None:
        path_TOOCAN = full_path
    else:

        filename = os.path.basename(full_path)
        date = os.path.basename(os.path.dirname(full_path))
        path_TOOCAN = os.path.join(settings['DIR_TOOCANSEG_DYAMOND'],date,filename)
        
    # Load TOOCAN data
    img_TOOCAN = xr.open_dataarray(path_TOOCAN)
    
    return img_TOOCAN

def load_var(grid, var_id, i_t): # the fact that we have to pass grid as an object justifies that 
                                 # Handler class should be made and inherit from CaseStudy or even Grid.
    if var_id in grid.new_variables_names:
        try:
            load_func = globals()[grid.new_var_functions[var_id]] # globals is the trick to get the func in same file with a string
                                                                  # but getattr within Handler would do. 
            
            da_new_var = load_func(grid, var_id, i_t)
            return da_new_var
        except AttributeError:
            print(f"Function {function} not found in 'utils' module.")
    else : 
        root = get_rootname_from_i_t(i_t)
        filename_var = root+f".{var_id}.2D.nc"
        filepath_var = os.path.join(grid.data_in, filename_var)
        da_var = xr.open_dataarray(filepath_var).load().sel(lon=grid.lon_slice,lat=grid.lat_slice)[0]
        return da_var
    
def load_prec(grid, var_id, i_t):
    """
    First handmade function (of I hope a long serie)
    These functions will typically be the kind of ones we'll add more and more 
    and invite people to add more and more. They must be quite independent of
    any other parts of TEMPEST naming conventions as they're by nature add-ons. 
    This is why I use very explicit variable names. 
    Oh and they must clean their loadings as they'll be called a lot...
    """
    dependency = grid.new_var_dependencies[var_id][0] # Returns 'Precac' only, could be called more simply
    previous_precac = load_var(grid, 'Precac', i_t -1)
    current_precac = load_var(grid, 'Precac', i_t)

    prec= previous_precac - current_precac
    del previous_precac
    del current_precac
    gc.collect()
    return prec
