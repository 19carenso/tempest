import numpy as np
import os
import time
import sys
import glob
import pickle
import time
import math 

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
    
    def __init__(self, grid, label_var_id = "MCS_label", overwrite_storms = True, overwrite=False, verbose=False):
        
        self.grid = grid
        self.settings = grid.settings
        start_time = time.time()
        self.overwrite = overwrite 
        self.verbose = verbose
        self.label_var_id = label_var_id
        if self.label_var_id == 'MCS_label':
            self.dir_storm = self.settings['DIR_STORM_TRACKING']
        elif self.label_var_id == 'MCS_label_Tb_Feng'    :
            self.dir_storm = self.settings['DIR_STORM_TRACKING_TB_FENG']
                # Should check if regridded MCS labels are already stored in grid
        self.labels_regridded_yxtm = grid.get_var_id_ds(self.label_var_id)[self.label_var_id].values
        self.mask_labels_regridded_yxt = np.any(~np.isnan(self.labels_regridded_yxtm), axis=3)

        # get storm tracking data
        print("Loading storms...")
        self.storms, self.label_storms, self.dict_i_storms_by_label = self.load_storms_tracking(overwrite_storms)
        print(f"Time elapsed for loading storms: {time.time() - start_time:.2f} seconds")
    
        if self.overwrite :
            for storm in self.storms:
                self.set_storm_growth_rate(storm, r_treshold=0.85)
            
            self.save_storms()
    
    def load_storms_tracking(self, overwrite):
        ## Make it a netcdf so that we don't struggle with loading it anymore
        dir_out = os.path.join(self.settings["DIR_DATA_OUT"], self.grid.casestudy.name)
        file_storms = os.path.join(dir_out, "storms"+self.label_var_id+".pkl")
        if os.path.exists(file_storms) and not overwrite:
            print("loading storms from pkl")
            with open(file_storms, 'rb') as file:
                storms = pickle.load(file)
        else : 
            if overwrite : print("Loading storms again because overwrite_storms is True")
            paths = glob.glob(os.path.join(self.dir_storm, '*.gz'))
            storms = load_toocan(paths[0])+load_toocan(paths[1])
            # weird bug of latmin and lonmax being inverted ! 
            for storm in storms :
                save_latmin = storm.latmin
                setattr(storm, "latmin", storm.lonmax)
                setattr(storm, "lonmax", save_latmin)

            with open(file_storms, 'wb') as file:
                pickle.dump(storms, file)
        label_storms = [storms[i].label for i in range(len(storms))]
        dict_i_storms_by_label = {}
        for i, storm in enumerate(storms):
                if storm.label not in dict_i_storms_by_label.keys():
                    dict_i_storms_by_label[storm.label] = i
        return storms, label_storms, dict_i_storms_by_label

    def save_storms(self):
        dir_out = os.path.join(self.settings["DIR_DATA_OUT"], self.grid.casestudy.name)
        file_storms = os.path.join(dir_out, "storms.pkl")
        paths = glob.glob(os.path.join(self.dir_storm, '*.gz'))
        storms = load_toocan(paths[0])+load_toocan(paths[1])
        with open(file_storms, 'wb') as file:
            pickle.dump(storms, file)
            
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
        t_lower_bounds = [-np.inf] + t_breaks_0[:-1]
        t_upper_bounds = t_breaks_0[1:] + [np.inf]
        
        s_lower_bounds = [0] # Positive or null surfaces. 
        s_upper_bounds = [+np.inf] #null surfaces at first and last breaks
        
        p , e = curve_fit(piecewise_fun, t, s, p0=t_breaks_0+[s_max_0], bounds = (t_lower_bounds + s_lower_bounds, t_upper_bounds + s_upper_bounds))

        s_id = self._piecewise_linear(t, p[0:N_breaks], p[-1])
        s_max = p[-1]
        t_breaks = list(p[0:N_breaks])
        
        return t_breaks,s_max,s_id

    def set_storm_growth_rate(self, storm, r_treshold = 0.85, verbose = False, plot = False):
        """
        Given a storm object, update it's growth_rate attribute 
        Returns an ax object to plot the fit
        """
        surf = np.array(storm.clusters.surfPix_172Wm2) * 16
        
        if len(surf) <= 4 : 
            growth_rate = np.nan
            setattr(storm, 'growth_rate', growth_rate)
            if verbose : print("A very short-lived storm passed by here...")
            return None
        else : 
            time = np.arange(0, len(surf))

            s_max = max(surf)
            time_breaks = [0, len(surf)//2, len(surf)]

            # warnings.filterwarnings("error", category=UndefinedMetricWarning)
            warnings.filterwarnings("error", category=OptimizeWarning)

            try:
                # Your existing code that raises the warning
                t_breaks, s_max, s_id = self._piecewise_fit(time, surf, time_breaks, s_max)
                t0, t_max, t_f = t_breaks
                r_squared = r2_score(surf, s_id)
                growth_r_squared = r2_score(surf[:math.ceil(t_breaks[1])], s_id[:math.ceil(t_breaks[1])])
                decay_r_squared = r2_score(surf[math.floor(t_breaks[1]):], s_id[math.floor(t_breaks[1]):])
                growth_rate = s_max / (t_breaks[1] - t_breaks[0])
                setattr(storm, 'growth_rate', growth_rate)
                setattr(storm, 'growth_rate_r2', r_squared)
                setattr(storm, 'growth_rate_r2_pos', growth_r_squared)
                setattr(storm, 'growth_rate_r2_neg', decay_r_squared)
                setattr(storm, 'growth_rate_t0', t0)
                setattr(storm, 'growth_rate_t_max', t_max)
                setattr(storm, 'growth_rate_t_f', t_f)
                setattr(storm, 'growth_rate_s_max', s_max)

            except OptimizeWarning as e:
                print("That's a complicated storm")
            
            except Exception as e:
                # Handle the warning here, e.g., print a message or log it
                print("Caught Exception:", e)
            
            if verbose : print(f"For storm with label {storm.label}, the growth rate computed by fitting a triangle is {growth_rate} with an r-score of {r_squared}")
        
            return r_squared, growth_r_squared, decay_r_squared, *t_breaks, s_max
        
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