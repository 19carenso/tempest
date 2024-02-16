import numpy as np
import os
import time
import sys
import glob
import pickle
import time
import math 
import xarray as xr 
import pandas as pd
import gc 

import warnings
from .load_toocan import load_toocan

import warnings
from scipy.optimize import OptimizeWarning

from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

class StormTracker():
    """
    To instantiate Toocan's object only once per jd.
    """
    def __init__(self, grid, label_var_id = "MCS_label", overwrite_storms = False, overwrite=False, verbose=False):
        
        self.grid = grid
        self.settings = grid.settings
        start_time = time.time()
        self.overwrite = overwrite 
        self.verbose = verbose
        self.label_var_id = label_var_id

        if self.label_var_id == 'MCS_label':
            self.dir_storm = self.settings['DIR_STORM_TRACKING']
        elif self.label_var_id == 'MCS_label_Tb_Feng':
            self.dir_storm = self.settings['DIR_STORM_TRACKING_TB_FENG']
                # Should check if regridded MCS labels are already stored in grid
            
        self.labels_regridded_yxtm = grid.get_var_id_ds(self.label_var_id)[self.label_var_id].values
        self.mask_labels_regridded_yxt = np.any(~np.isnan(self.labels_regridded_yxtm), axis=3)

        # get storm tracking data
        print("Loading storms...")
        self.ds_storms, self.file_storms = self.load_storms_tracking(overwrite_storms) #, self.label_storms, self.dict_i_storms_by_label
        print(f"Time elapsed for loading storms: {time.time() - start_time:.2f} seconds")
    
        ## test and incubate in a func
        if self.overwrite:
            grw_var_names= ["growth_rate", "r_squared", "growth_r_squared", "decay_r_squared", "t0", "t_max", "t_f", "s_max"]
            grw_vars = [ [] for _ in grw_var_names]
            with xr.open_dataset(self.file_storms) as ds_storms:
                for label in ds_storms.label:
                    storm = ds_storms.sel(label = label)
                    grw_output = self.get_storm_growth_rate(storm, r_treshold=0.85)
                    for out, grw_var in zip(grw_output, grw_vars):
                        grw_var.append(out)
                grw_arr_vars = [np.array(grw_var) for grw_var in grw_vars]
                grw_data_vars = [xr.DataArray(grw_arr_var, 
                                            dims = ['label'], 
                                            coords = {"label": ds_storms.label})            
                                                for grw_arr_var in grw_arr_vars]
                for grw_var_name, grw_data_var in zip(grw_var_names, grw_data_vars):
                    ds_storms[grw_var_name] = grw_data_var
                self.save_storms(ds_storms)
    
    def load_storms_tracking(self, overwrite):
        ## Make it a netcdf so that we don't struggle with loading it anymore
        dir_out = os.path.join(self.settings["DIR_DATA_OUT"], self.grid.casestudy.name)
        file_storms = os.path.join(dir_out, "storms_"+self.label_var_id+".nc")
        if os.path.exists(file_storms) and not overwrite:
            print("loading storms from netcdf")
            with open(file_storms, 'rb') as file:
                ds_storms = xr.open_dataset(file)
        else : 
            if overwrite : print("Loading storms again because overwrite_storms is True")
            paths = glob.glob(os.path.join(self.dir_storm, '*.gz'))
            storms = load_toocan(paths[0])+load_toocan(paths[1])
            # weird bug of latmin and lonmax being inverted ! 
            filtered_storms = []

            for storm in storms:
                # print(storm)
                # Swap latmin and lonmax
                save_latmin = storm.latmin
                storm.latmin = storm.lonmax
                storm.lonmax = save_latmin
                
                # Check the condition based on Utime_End
                if storm.Utime_End / 1800 >= 960:
                    # Add the storm to the new list if the condition is met
                    filtered_storms.append(storm)

            # Update the original list with the filtered storms
            storms = filtered_storms
            print("making ds storms ...")
            ## !!!!!!!!!!!!
            print(type(storms))
            print(storms[0])
            ds_storms = self.make_ds(storms)
            ds_storms.to_netcdf(file_storms)
            del filtered_storms
            gc.collect()
            print("ds storms saved ! ")

        # label_storms = [ds_storms['label'].isel(i) for i in range(len(ds_storms['label']))]
        # dict_i_storms_by_label = {}
        # for i, label in enumerate(ds_storms['label']):
        #     if label not in dict_i_storms_by_label.keys():
        #         dict_i_storms_by_label[label] = i

        return ds_storms, file_storms #, label_storms, dict_i_storms_by_label

    def save_storms(self, ds_storms):
        os.remove(self.file_storms)
        ds_storms.to_netcdf(self.file_storms)
            
    def _piecewise_linear(self, t:np.array,t_breaks:list,s_max:float):
        """
        Define piecewise linear surface growth over timer with constant value at beginning and end.

        Args:
            t (np.array): t coordinate
            t_breaks (list): t values of break points
            s_breaks (list): surface values of break points

        Returns:
            np.array: piecewize surface
            
        """
        
        N_breaks = len(t_breaks)
        s_breaks = [0, s_max, 0]
        cond_list = [t <= t_breaks[0]]+\
                    [np.logical_and(t > t_breaks[i-1],t <= t_breaks[i]) for i in range(1,N_breaks)]+\
                    [t > t_breaks[N_breaks-1]]
                    
        def make_piece(k):
            def f(t):
                return s_breaks[k-1]+(s_breaks[k]-s_breaks[k-1])/(t_breaks[k]-t_breaks[k-1])*(t-t_breaks[k-1])
            return f 
        
        # Returns 0 rather than s_breaks[0], s_breaks[N_breaks-1] to ensure apparition and disparition of cloud.
        func_list = [lambda t: s_breaks[0]]+\
                    [make_piece(k) for k in range(1,N_breaks)]+\
                    [lambda t: s_breaks[N_breaks-1]]
                    
        return np.piecewise(t,cond_list,func_list)

    def _piecewise_fit(self, t:np.array,s:np.array,t_breaks_0:list,s_max_0:float):    
        """
        Compute piecewise-linear fit of surf(t).

        Args:
            t (np.array): t coordinate
            s (np.array): surface over time
            t_breaks_0 (list): initial t values of break points
            s_max_0(list): maximal surf value

        Returns:
            t_breaks (list): fitted t values of break points
            s_max (list): fitted s_max value
            s_id (np.array): piecewize s fit

        """

        N_breaks = len(t_breaks_0)

        def piecewise_fun(t,*p):
            return self._piecewise_linear(t,p[0:N_breaks],p[-1])

        # we add bounds so that time breaks stay ordered
        t_lower_bounds = [0] + t_breaks_0[:-1] # 0 instead of -np.inf
        t_upper_bounds = t_breaks_0[1:] + [np.inf]
        
        s_lower_bounds = [0] # Positive or null surfaces. 
        s_upper_bounds = [+np.inf] #null surfaces at first and last breaks
        
        p , e = curve_fit(piecewise_fun, t, s, p0=t_breaks_0+[s_max_0], bounds = (t_lower_bounds + s_lower_bounds, t_upper_bounds + s_upper_bounds))

        s_id = self._piecewise_linear(t, p[0:N_breaks], p[-1])
        s_max = p[-1]
        t_breaks = list(p[0:N_breaks])
        
        return t_breaks,s_max,s_id

    def get_storm_growth_rate(self, storm, r_treshold = 0.85, verbose = False, plot = False):
        """
        Given a storm object, update it's growth_rate attribute 
        Returns an ax object to plot the fit
        """
        u_i, u_e = storm.Utime_Init.values, storm.Utime_End.values
        init= max(int(u_i/1800) - 960 , 0)
        end = min(int(u_e/1800) - 960 , 959)+1
        surf = storm.surfPix_172Wm2.values[init:end]*16
        
        if len(surf) <= 4 : 
            nan_output = [np.nan for _ in range(8)]
            # setattr(storm, 'growth_rate', growth_rate)
            if verbose : print("A very short-lived storm passed by here...")
            return nan_output
        else : 
            time = np.arange(0, len(surf))

            s_max = max(surf)
            time_breaks = [0, len(surf)//2, len(surf)]

            warnings.filterwarnings("error", category=OptimizeWarning)

            try: # parfois ça marche pô
                t_breaks, s_max, s_id = self._piecewise_fit(time, surf, time_breaks, s_max)
                t0, t_max, t_f = t_breaks
                r_squared = r2_score(surf, s_id)
                growth_r_squared = r2_score(surf[:math.ceil(t_breaks[1])], s_id[:math.ceil(t_breaks[1])])
                decay_r_squared = r2_score(surf[math.floor(t_breaks[1]):], s_id[math.floor(t_breaks[1]):])
                growth_rate = s_max / (t_breaks[1] - t_breaks[0])

            except OptimizeWarning as e:
                print("That's a complicated storm")
                return nan_output
            
            except Exception as e:
                # Handle the warning here, e.g., print a message or log it
                print("Caught Exception:", e)
                return nan_output
            
            if verbose : print(f"For storm with label {storm.label}, the growth rate computed by fitting a triangle is {growth_rate} with an r-score of {r_squared}")
        
            return [growth_rate, r_squared, growth_r_squared, decay_r_squared, t0, t_max, t_f, s_max]
        
    def make_ds(self, storms):
        """
        TODO : the whole clusters part seems to only return np.nan... so not working
        """

        attribute_names = [attr for attr in dir(storms[0]) if not callable(getattr(storms[0], attr)) and  not attr.startswith("clusters") and not attr.startswith("__")]
        print("Label wise attributes ", attribute_names)
        dict_storm = {attr: [getattr(obj, attr) for obj in storms if not attr.startswith("__")] for attr in attribute_names}

        ## get clusters per storm so per label
        clusters = [storm.clusters for storm in storms]

        #first one for attr names
        mcs_lfc = clusters[0]
        clusters_attribute_names = [attr for attr in dir(mcs_lfc)if not callable(getattr(mcs_lfc, attr)) and not attr.startswith("__")]
        print("Label's Lifecycle wise attributes ", clusters_attribute_names)

        ## Build ds for storms attributes
        data_vars = {}
        labels = dict_storm["label"]
        for key in dict_storm.keys():
            da = xr.DataArray(dict_storm[key], dims = ['label'], coords = {'label' : labels})
            data_vars[key] = da
            
        # ds_storms = xr.Dataset(data_vars)

        ## Build ds for clusters attributes (list)
        storm_clusters = {key : [] for key in clusters_attribute_names}
        for i_storm, mcs_lfc in enumerate(clusters):
            for attr in clusters_attribute_names:
                var = getattr(mcs_lfc, attr)
                if len(var)>0:
                    storm_clusters[attr].append(var)
                    
        Utime_min, Utime_max = 0,0
        for storm in storms : 
            Utime_min = min(Utime_min, storm.Utime_Init)
            Utime_max = max(Utime_max, storm.Utime_End)  
            
        Utime = np.arange(Utime_min, Utime_max + 1800, 1800)  
        time = np.arange(960, 1920, 1) #should be similar

        n_label = len(storm_clusters["Utime"])
        n_time = 960 # all utime values possible within timerange

        for key in storm_clusters.keys():
            # one cluster attributes is empty
            if len(storm_clusters[key])>0 : 
                data = np.full((n_label, n_time), np.nan)
                for i_label, list, index_Utime in zip(np.arange(len(storm_clusters[key])), storm_clusters[key], storm_clusters["Utime"]):
                    for i, utime in enumerate(index_Utime): # peut-être relaxé pour utilise Utime_Init/1800 : Utime_End/1800
                        idx_utime = int(utime/1800)-960 # Utime
                        if idx_utime >= 0 and idx_utime < 960: 
                            data[i_label, idx_utime] = list[i] 
                            # if idx_utime == 0  : print(i_label)
                da = xr.DataArray(data, dims = ['label', 'time'], coords = {'label' : dict_storm['label'], 'time' : time})
                    
                # handle same name 
                if key =="olrmin":
                    data_vars["lifecycle_olrmin"] = da
                else : 
                    data_vars[key] = da

        ds = xr.Dataset(data_vars)
        ds = ds.drop_duplicates('label', keep = False) #choix ici, attendre rep Laurent

        return ds



# if plot : 
#     # Return ax object if plotting is necessary
#     fig, ax = plt.subplots()
#     ax.scatter(time, surf, label='Surface')
#     time_plot = np.linspace(0, time.max(), 1000)
#     #ax.plot(time_plot, piecewise_linear(time_plot, t_breaks, s_breaks), 'r-', label='Idealized Surface')
#     ax.legend()
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Surface Values')
#     ax.set_title('Fitting a Triangle Function to Surface Values over Time')
#     plt.show()