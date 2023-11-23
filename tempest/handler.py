import pandas as pd
import os 
import sys
import re
import yaml 

import gc
import xarray as xr


class Handler():
    def __init__(self, settings_path):
        self.settings_path = settings_path
        with open(self.settings_path, 'r') as file:
            self.settings = yaml.safe_load(file)

        self.rel_table = self.load_rel_table(self.settings['REL_TABLE'])

    def load_seg(self, i_t):
        
        #path_dyamond = self.get_rootname_from_i_t(i_t)
        path_toocan = self.rel_table.loc[self.rel_table['Unnamed: 0.1'] == i_t-1, 'img_seg_path']
        if len(path_toocan)==1 : path_toocan = '/' + path_toocan.values[0]
        else : print('Rel_table has a problem')
        img_toocan = xr.open_dataarray(path_toocan, engine='netcdf4')

        ## MJC : I don't understand why you need to do this.
        # if self.settings['DIR_TOOCANSEG_DYAMOND'] is None: path_TOOCAN = full_path
        # else:
        #     filename = os.path.basename(full_path)
        #     date = os.path.basename(os.path.dirname(full_path))
        #     path_TOOCAN = os.path.join(self.settings['DIR_TOOCANSEG_DYAMOND'],date,filename)
            
        return img_toocan
    
    def i_t_from_utc(self, utc):
        #path_dyamond = self.get_rootname_from_i_t(i_t)
        i_t = self.rel_table.loc[self.rel_table["UTC"] == utc, 'Unnamed: 0'].values
        if len(i_t)!=1 : print("issue with utc, i_t correspondance in relation table")
        else :
            i_t = int(i_t[0]+1)
        return i_t

    def get_timestamp_from_filename(self, filename):
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

    def get_rootname_from_i_t(self, i_t):
        """
        input: the i_t of the classical DYAMOND .nc data file, eg. 1441 (*240 = 345840)
        output: data rootname eg. "DYAMOND_9216x4608x74_7.5s_4km_4608_0000345840"
        """

        string_timestamp = str(int(int(i_t) * 240)).zfill(10)
        result = f"DYAMOND_9216x4608x74_7.5s_4km_4608_"+string_timestamp
        return result

    def load_rel_table(self, file_path):
        """
        Load a .csv file and return its contents as a pandas DataFrame.
        Rel_table contains the path to the output of the storm tracking file per file.

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

    ## This method is specific to your TIME_RANGE and files in DIR_DATA_IN
    def extract_digit_after_sign(self, input_string):
        """
        Extract the digit after the sign in a string.
        If the string does not contain a sign followed by a digit, return None.
        """

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

    def load_var(self, grid, var_id, i_t): 
        """
        Load a variable from a file.
        If the variable is a new one, call the appropriate function.
        """
        if var_id in grid.new_variables_names:
            if hasattr(self, grid.new_var_functions[var_id]):
                load_func = getattr(self, grid.new_var_functions[var_id])
            else : print(f"Handler has no method {grid.new_var_functions[var_id]}")

            da_new_var = load_func(grid, i_t)
            return da_new_var
            
        else : 
            root = self.get_rootname_from_i_t(i_t)
            filename_var = root+f".{var_id}.2D.nc"
            filepath_var = os.path.join(grid.data_in, filename_var)
            da_var = xr.open_dataarray(filepath_var).load().sel(lon=grid.lon_slice,lat=grid.lat_slice)[0]
            return da_var
        
    def load_prec(self, grid, i_t):
        """
        First handmade function (of I hope a long serie)
        These functions will typically be the kind of ones we'll add more and more 
        and invite people to add more and more. They must be quite independent of
        any other parts of TEMPEST naming conventions as they're by nature add-ons. 
        This is why I use very explicit variable names. 
        Oh and they must del their loadings as they'll be called a lot...
        """

        previous_precac = self.load_var(grid, 'Precac', i_t -1)
        current_precac = self.load_var(grid, 'Precac', i_t)

        prec= current_precac - previous_precac
        del previous_precac
        del current_precac
        gc.collect()
        return prec
    
