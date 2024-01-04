import pandas as pd
import os 
import sys
import re
import yaml 
import numpy as np
import datetime as dt
import gc
import xarray as xr
import warnings 

from .thermo import saturation_specific_humidity

class Handler():
    def __init__(self, settings_path):
        self.settings_path = settings_path
        with open(self.settings_path, 'r') as file:
            self.settings = yaml.safe_load(file)

        self.rel_table = self.load_rel_table(self.settings['REL_TABLE'])

    def load_seg(self, grid, i_t):
        path_toocan = self.get_filename_classic(i_t) ## There is the differences
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=xr.SerializationWarning)
            img_toocan = xr.open_dataset(path_toocan, engine='netcdf4').cloud_mask
        return img_toocan

    def load_seg_tb_feng(self, grid, i_t):
        path_toocan = self.get_filename_tb_feng(i_t) ## There is the differences
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=xr.SerializationWarning)
            img_toocan = xr.open_dataset(path_toocan, engine='netcdf4').cloud_mask
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

    def get_filename_classic(self, i_t):
        root = self.settings['DIR_STORM_TRACK']

        dict_date_ref = self.settings["DATE_REF"]
        datetime_ref = dt.datetime(dict_date_ref['year'], dict_date_ref['month'], dict_date_ref['day'])
        timestamp_ref = datetime_ref.timestamp()

        i_t_in_seconds = i_t * 30 * 60
        timezone_weird_lag_to_watch = 2*60*60 #2hours
        timestamp = timestamp_ref + i_t_in_seconds + timezone_weird_lag_to_watch
        date = dt.datetime.utcfromtimestamp(timestamp)
        
        string_date = date.strftime("%Y_%m_%d")
        hours = int(date.strftime("%H"))
        minutes = int(date.strftime("%M"))
        n_half_hours = int(2*hours + minutes/30 + 1)
        dir_path = os.path.join(root, string_date)
        string_date_no_underscore = string_date.replace('_', '')
        file_root= "mcs_mask_TOOCAN_SAM_"+string_date_no_underscore+'-'+str(n_half_hours).zfill(3)+'.nc'
        filename = os.path.join(dir_path, file_root)
        return filename

    def get_filename_tb_feng(self, i_t):
        root = self.settings['DIR_STORM_TRACK_TB_FENG']

        dict_date_ref = self.settings["DATE_REF"]
        datetime_ref = dt.datetime(dict_date_ref['year'], dict_date_ref['month'], dict_date_ref['day'])
        timestamp_ref = datetime_ref.timestamp()

        i_t_in_seconds = i_t * 30 * 60
        timezone_weird_lag_to_watch = 2*60*60 #2hours
        timestamp = timestamp_ref + i_t_in_seconds + timezone_weird_lag_to_watch
        date = dt.datetime.utcfromtimestamp(timestamp)
        
        string_date = date.strftime("%Y_%m_%d")
        hours = int(date.strftime("%H"))
        minutes = int(date.strftime("%M"))
        n_half_hours = int(2*hours + minutes/30 + 1)
        dir_path = os.path.join(root, string_date)
        string_date_no_underscore = string_date.replace('_', '')
        file_root= "mcs_mask_TOOCAN_SAM_"+string_date_no_underscore+'-'+str(n_half_hours).zfill(3)+'.nc'
        filename = os.path.join(dir_path, file_root)
        return filename

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
        Load a variable at specified i_t.
        If the variable is a new one, calls the appropriate function, that will recursively call load_var
        If the variable is 3D one, it will return a dataset instead.
            Must be handled in your designed funcs that depends on 3D vars
        """
        new_var_names = grid.casestudy.new_variables_names
        var_2d = grid.casestudy.var_names_2d
        var_3d = grid.casestudy.var_names_3d
        new_var_funcs = grid.casestudy.new_var_functions
        
        if var_id in new_var_names:
            if hasattr(self, new_var_funcs[var_id]):
                load_func = getattr(self, new_var_funcs[var_id])
            else : print(f"Handler has no method {new_var_funcs[var_id]}")
            da_new_var = load_func(grid, i_t)
            return da_new_var
            
        else : 
            root = self.get_rootname_from_i_t(i_t)
            if var_id in var_2d : 
                path_data_in = grid.settings["DIR_DATA_2D_IN"]
                filename_var = root+f".{var_id}.2D.nc"
                filepath_var = os.path.join(path_data_in, filename_var)
                var = xr.open_dataarray(filepath_var).load().sel(lon=grid.casestudy.lon_slice,lat=grid.casestudy.lat_slice)[0]
            elif var_id in var_3d :
                path_data_in = grid.settings["DIR_DATA_3D_IN"]
                chunks = {'z': 74} # Always 74 vertical level in these data
                filename_var = root+f"_{var_id}.nc"
                filepath_var = os.path.join(path_data_in, filename_var)
                var = xr.open_dataset(filepath_var, chunks = chunks).sel(lon=grid.casestudy.lon_slice,lat=grid.casestudy.lat_slice).isel(time=0)
            return var
        
    def load_prec(self, grid, i_t):
        """
        First handmade function (of I hope a long serie)
        These functions will typically be the kind of ones we'll add more and more 
        and invite people to add more and more. They must be quite independent of
        any other parts of TEMPEST naming conventions as they're by nature add-ons. 
        This is why I use very explicit variable names. 
        Oh and they must del their loadings as they'll be called a lot...
        """
        if i_t in self.settings["prec_i_t_bug_precac"]:
            previous_precac = self.load_var(grid, 'Precac', i_t-2)
        else : 
            previous_precac = self.load_var(grid, 'Precac', i_t-1)   

        current_precac = self.load_var(grid, 'Precac', i_t)

        prec= current_precac - previous_precac
        prec = xr.where(prec < 0, 0, prec)
        
        del previous_precac
        del current_precac
        gc.collect()
        return prec

    def load_prec_minus_1(self, grid, i_t):
        return self.load_prec(grid, i_t)

    def compute_qv_sat(self, grid, i_t):
        pp = self.load_var(grid, "PP", i_t)
        tabs = self.load_var(grid, "TABS", i_t)
        # retrieve surface temperature
        p_surf = 100*pp["p"].values[0]+pp["PP"].values[0, 0, :, :]
        t_surf = tabs["TABS"][0,0,:,:].values
        
        original_shape = p_surf.shape
        qv_sat = saturation_specific_humidity(t_surf.ravel(), p_surf.ravel()).reshape(original_shape)
        
        del pp
        del tabs
        del p_surf
        del t_surf
        gc.collect()
        return qv_sat

    def extract_w500(self, grid, i_t):
        w = self.load_var(grid, "W", i_t)
        ind_500 = int(np.abs(w.p.load() - 500).argmin())
        w_500 = w.isel(z=ind_500).W.compute()
        
        del w
        gc.collect()
        return w_500
    
    def compute_qv_sat_2d(self, grid, i_t):
        pres = 100*self.load_var(grid, "PSFC", i_t).values
        temp = self.load_var(grid, "T2mm", i_t).values
        original_shape = pres.shape
        qv_sat = saturation_specific_humidity(temp.ravel(), pres.ravel()).reshape(original_shape)
        del pres
        del temp
        gc.collect()
        return qv_sat
