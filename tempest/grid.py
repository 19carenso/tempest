import warnings
import time
import os 

import pickle
import copy
import gc


import numpy as np
import pandas as pd
import xarray as xr
import math

from datetime import datetime as dt

import matplotlib.pyplot as plt

from .casestudy import CaseStudy
from .utils import load_var # All these imports could actually be resumed by importing an instance of Handler

class Grid(CaseStudy): 
    # except verbose i actually don't want any
    def __init__(self, settings, fast = True, overwrite = True, verbose_steps = False, verbose=False):
        super().__init__(settings, verbose = verbose)
        
        ## Get the region borders
        self.n_lat = self.lat_slice.stop - self.lat_slice.start
        self.n_lon = self.lon_slice.stop - self.lon_slice.start
        
        ## Bool for running a quicker computation (can multiply by x2 or x8 depending on the function)
        self.fast = fast

        
        ## explicitly verbose
        self.verbose = verbose

        ## very talkative one, mainly for computation time measurements
        self.verbose_steps = verbose_steps

        self.make_output_ready(overwrite)

        # Funcs to compute on variable 
        # Actually this should be done in CaseStudy and passed there, so that it'd be eazy to control which func for any var_id
        self.func_names = ['max', 'mean']

    def make_output_ready(self, overwrite):
        self.path_out = os.getcwd() + self.settings["DIR_OUT"] + '/' + self.name 
        if not os.path.exists(self.path_out):
            self.overwrite = True
        else : self.overwrite = overwrite 

        if self.overwrite:
            self._prepare_grid()
            if not os.path.exists(self.path_out):
                os.makedirs(self.path_out)

            filename = self.path_out + '/grid_attributes.pkl'
            self.save_grid_attr(filename)
        else :
            filename = self.path_out + '/grid_attributes.pkl'
            self.load_grid_attr(filename)

    def save_grid_attr(self, filename):
        state = {attr: getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")}
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    def load_grid_attr(self, filename):
        print(f"Found grid attributes file , so loading {filename} instead of computing")
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        for attr, value in state.items():
            setattr(self, attr, value)

    def _prepare_grid(self):
        if  self.overwrite and self.fast : 
            # import one timestep to use its coordinates as a template
            for file in os.listdir(self.data_in):
                self.template_native_df = xr.open_dataset(os.path.join(self.data_in,file))
                break

            # compute
            
            if self.verbose_steps: print('-- Prepare data')

            if self.verbose_steps: print('compute coord centers...')
            self.lat_centers = self.template_native_df['lat'].sel(lat=self.lat_slice).values
            self.lon_centers = self.template_native_df['lon'].sel(lon=self.lon_slice).values
            
            if self.verbose_steps: print('compute pixel surface...')

            self.lat_length_on_center, self.lon_length_on_center = self._compute_length_centers_from_coord_borders()

            self.pixel_surface = self.lat_length_on_center * self.lon_length_on_center      

            self.global_area = self.pixel_surface.sum()/self.n_lat/self.n_lon #depending on the remeshed grid point surface you want computed here
            self.global_lat_area = self.global_area*self.n_lon 

            self.lat_global = [i for i in range(self.n_lat)]
            self.lon_global = [j for j in range(self.n_lon)]

            ## We start by computing the area of each latitude band
            if self.verbose_steps: print('compute lat band area')
            self.lat_area = np.sum(self.pixel_surface, axis=1)
            self.cumsum_lat_area = np.cumsum(self.lat_area)

            if self.verbose_steps: print('compute i and alpha lat')
            self.i_min, self.i_max, self.alpha_i_min, self.alpha_i_max = self._get_i_and_alpha_lat()

            if self.verbose_steps: print('compute area by lon')
            if not self.fast : 
                self.slices_i_lat = [slice(i_min+1, i_max) for i_min, i_max in zip(self.i_min[:,0], self.i_max[:,0])] 
            elif self.fast :
                self.slices_i_lat = [slice(i_min, i_max) for i_min, i_max in zip(self.i_min[:,0], self.i_max[:,0])]    
                
            self.area_by_lon_and_global_lat = self._compute_area_by_lon()
            self.cumsum_area_by_lon_and_global_lat = np.cumsum(self.area_by_lon_and_global_lat, axis = 1)

            if self.verbose_steps: print('compute j and alpha lon')
            self.j_min, self.j_max, self.alpha_j_min, self.alpha_j_max = self._get_j_and_alpha_lon()
            self.j_min, self.j_max =self.j_min.astype(int), self.j_max.astype(int)

            if self.verbose_steps: print('build slices j lon')
            self.slices_j_lon = self._build_slices_j_lon()

            if self.verbose_steps: print('compute grid surface')
            self.grid_surface = self.sum_data_from_center_to_global(self.pixel_surface)

            # remove template data
            # self.template_native_df = None  # USE GARBAGE COLLECTOR?
        
    def _compute_length_centers_from_coord_borders(self):
        lat_length = np.zeros(shape=(len(self.lat_centers), len(self.lon_centers)))
        lon_length = np.zeros(shape=(len(self.lat_centers), len(self.lon_centers)))
        
        self.lat_borders = self._get_coord_border_from_centers(self.lat_centers)
        self.lon_borders = self._get_coord_border_from_centers(self.lon_centers)
        
        for i_lat in range(len(self.lat_borders)-1):
            for j_lon in range(len(self.lon_borders)-1):
                lat1, lat2, lon1, lon2 = self.lat_borders[i_lat], self.lat_borders[i_lat+1], self.lon_borders[j_lon], self.lon_borders[j_lon+1]
                lat_length[i_lat, j_lon] = self.haversine(lat1, lon1, lat2, lon1)
                lon_length[i_lat, j_lon] = self.haversine(lat1, lon1, lat1, lon2)
        return lat_length, lon_length

    def _get_coord_border_from_centers(self, coord_centers):
        coord_borders = list()
        coord_borders.append(np.floor(coord_centers[0]))
        for i in range(len(coord_centers)-1):
            coord_borders.append((coord_centers[i]+coord_centers[i+1])/2)
        coord_borders.append(np.ceil(coord_centers[-1]))  
        return coord_borders
    
    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate the distance between two points on the Earth (specified in decimal degrees)
        using the Haversine formula.
        """
        R = 6371  # Earth's radius in kilometers

        # Convert decimal degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c

        return distance
    
    def _get_i_and_alpha_lat(self):
        i_min, i_max = np.zeros((self.n_lat, self.n_lon)), np.zeros((self.n_lat, self.n_lon))
        alpha_i_min, alpha_i_max = np.ones((self.n_lat, self.n_lon)), np.ones((self.n_lat, self.n_lon))

        for i_lat, cum_length in enumerate(self.cumsum_lat_area):
            border_left = cum_length - self.lat_area[i_lat] ## does lat_area really needs to be a list or is a constant ? 
            border_right = cum_length
            
            for i in range(self.n_lat):
                cum_global_length = (i+1)*self.global_lat_area
                
                if cum_global_length > border_left and ((cum_global_length < border_right) or (math.isclose(cum_global_length, border_right))): # here we must use math.isclose for the behavior of <=
                    bottom_contrib = (cum_global_length - border_left)/self.lat_area[i_lat]
                    top_contrib = (border_right - cum_global_length)/self.lat_area[i_lat]           
                    # if self.verbose : print('local', i_lat, cum_length,'global',  i, cum_global_length,'borders',  border_left, border_right, 'contribs', bottom_contrib, top_contrib)
                    
                    if not self.fast:
                        if i != self.n_lat-1:
                            i_min[i+1, :] = i_lat
                            alpha_i_min[i+1, :] = top_contrib if not (math.isclose(cum_global_length, border_right)) else 0
                        

                        i_max[i, :] = i_lat
                        alpha_i_max[i, :] = bottom_contrib if not (math.isclose(cum_global_length, border_right)) else 1
                    
                    if self.fast:
                        if bottom_contrib > top_contrib : ## hence this area deserves to be kept in the regridded new area
                            if i != self.n_lat-1:
                                i_min[i+1, :] = i_lat 
                            i_max[i,:] = i_lat
                            
                        elif top_contrib > bottom_contrib : ##there this area must be left to the nex regridded new area so we update our index to i_lat-1 , the previous area index
                            if i != self.n_lat-1:
                                i_min[i+1, :] = i_lat-1
                            i_max[i, :] = i_lat-1
                    
        return i_min.astype(int), i_max.astype(int), alpha_i_min, alpha_i_max
    
    def _compute_area_by_lon(self):
            area_by_lon = np.zeros((self.n_lat, self.lon_centers.shape[0]))
            for j_lon in range(self.lon_centers.shape[0]):
                for i, slice_i_lat in enumerate(self.slices_i_lat):
                    # if not self.fast : 
                    i_min = self.i_min[i, :]
                    i_min = self._check_all_values_same(i_min)
                    i_max = self.i_max[i, :]
                    i_max = self._check_all_values_same(i_max)
                    alpha_i_min = self.alpha_i_min[i, :]
                    alpha_i_min = self._check_all_values_same(alpha_i_min)
                    alpha_i_max = self.alpha_i_max[i, :]
                    alpha_i_max = self._check_all_values_same(alpha_i_max)
                
                    ## print i_min, i_max, alpha_i_min, alpha_i_max
                    #if self.verbose : print(i, i_min, i_max, alpha_i_min, alpha_i_max)
                    bottom_sum = self.pixel_surface[i_min,j_lon]*alpha_i_min
                    #if self.verbose : print(bottom_sum)
                    top_sum = self.pixel_surface[i_max,j_lon]*alpha_i_max
                    #if self.verbose : print(top_sum)
                    mid_sum = np.sum(self.pixel_surface[slice_i_lat, j_lon])
                    #if self.verbose : print(mid_sum)
                    area_by_lon[i, j_lon] = mid_sum+bottom_sum+top_sum
                    #print everything 
                    # elif self.fast :
                    #     area_by_lon[i, j_lon] = np.sum(self.pixel_surface[slice_i_lat, j_lon])
                    if False : print('i', i, 'j_lon', j_lon, 'i_min', i_min, 'i_max', i_max, 'slice_i_lat', slice_i_lat, 'alpha_i_min', alpha_i_min, 'alpha_i_max', alpha_i_max, 'bottom_sum', bottom_sum, 'top_sum', top_sum, 'mid_sum', mid_sum, 'area_by_lon', area_by_lon[i, j_lon])
            
            return area_by_lon
    
    def _get_j_and_alpha_lon(self):
        j_min, j_max = np.zeros((self.n_lat, self.n_lon)), np.zeros((self.n_lat, self.n_lon))
        alpha_j_min, alpha_j_max = np.ones((self.n_lat, self.n_lon)), np.ones((self.n_lat, self.n_lon))
        for i in range(self.n_lat):
            cumsum_area_by_lon = self.cumsum_area_by_lon_and_global_lat[i, :]
            for j_lon, cum_length in enumerate(cumsum_area_by_lon):
                border_left = cum_length - self.area_by_lon_and_global_lat[i, j_lon]
                border_right = cum_length
                
                for j in range(self.n_lon):
                    cum_global_length = (j+1)*self.global_area
                    
                    if cum_global_length > border_left  and ((cum_global_length) < border_right or (math.isclose(cum_global_length, border_right))):
                        left_contrib = (cum_global_length - border_left)/self.area_by_lon_and_global_lat[i, j_lon]
                        right_contrib = (border_right - cum_global_length)/self.area_by_lon_and_global_lat[i, j_lon]
                        
                        # if not self.fast:
                            # if self.verbose : print('local', j_lon, cum_length,'global',  j, cum_global_length,'borders',  border_left, border_right, 'contribs', left_contrib, right_contrib)
                        if j!= self.n_lon-1:
                            j_min[i, j+1] = j_lon
                            alpha_j_min[i, j+1] = right_contrib if not (math.isclose(cum_global_length, border_right)) else 0
                            
                        j_max[i, j] = j_lon
                        alpha_j_max[i, j] = left_contrib if not (math.isclose(cum_global_length, border_right)) else 1
                        
                        # elif self.fast: 
                        #     if left_contrib > right_contrib:
                        #         if j != self.n_lon-1:
                        #             j_min[i, j+1] = j_lon
                        #         j_max[i,j] = j_lon
                        #     elif right_contrib > left_contrib:
                        #         if j!= self.n_lon-1:
                        #             j_min[i, j+1] = j_lon
                        #         j_max[i,j] = j_lon
                                
        return j_min, j_max, alpha_j_min, alpha_j_max   
    
    def _build_slices_j_lon(self):
        slices_j_lon = np.empty((self.n_lat, self.n_lon), dtype=object)
        for i in range(self.n_lat):
            for j in range(self.n_lon):
                if not self.fast :  
                    slices_j_lon[i, j] = slice(int(self.j_min[i, j])+1, int(self.j_max[i, j])) 
                elif self.fast :
                    slices_j_lon[i, j] = slice(int(self.j_min[i, j])  , int(self.j_max[i, j])) 
        return slices_j_lon
    
    def sum_data_from_center_to_global(self, data_on_center):
        x = data_on_center
        X = np.zeros((self.n_lat, self.n_lon))
        for i, slice_i_lat in enumerate(self.slices_i_lat):
            for j, slice_j_lon in enumerate(self.slices_j_lon[i]):
                if self.verbose : pass #print(slice_i_lat, slice_j_lon)
                if self.fast : 
                    X[i,j] = np.sum(x[slice_i_lat, slice_j_lon])
                else : 
                    mid_sum = np.sum(x[slice_i_lat, slice_j_lon])
                    bottom_sum = np.sum( x[self.i_min[i,j], slice_j_lon]*self.alpha_i_min[i,j])
                    top_sum = np.sum( x[self.i_max[i,j], slice_j_lon]*self.alpha_i_max[i,j])
                    left_sum = np.sum( x[slice_i_lat, self.j_min[i,j]]*self.alpha_j_min[i,j])
                    right_sum = np.sum( x[slice_i_lat, self.j_max[i,j]]*self.alpha_j_max[i,j])
                    bottom_left_corner = x[self.i_min[i,j], self.j_min[i,j]]*self.alpha_j_min[i,j]*self.alpha_i_min[i,j]
                    bottom_right_corner = x[self.i_min[i,j], self.j_max[i,j]]*self.alpha_j_max[i,j]*self.alpha_i_min[i,j]
                    top_left_corner = x[self.i_max[i,j], self.j_min[i,j]]*self.alpha_j_min[i,j]*self.alpha_i_max[i,j]
                    top_right_corner = x[self.i_max[i,j], self.j_max[i,j]]*self.alpha_j_max[i,j]*self.alpha_i_max[i,j]
                    X[i, j] = mid_sum+bottom_sum+top_sum+left_sum+right_sum+bottom_left_corner+bottom_right_corner+top_left_corner+top_right_corner
        del x
        gc.collect()
        return X
    
    def _check_all_values_same(self, arr):
        first_value = arr[0]
        for value in arr:
            if value != first_value:
                raise ValueError("Array contains different values")

        return first_value
    
    def plot_grid(self, w = 10, h = 5):
        # Create a 1x2 grid of subplots
        plt.figure(figsize=(w, h))  # Adjust the figure size as needed
        plt.subplot(1, 2, 1)  # Create the first subplot

        ## Here checks if the global_pixel_surf in ds looks correct (and is accessible so that grid.ds is not corrupted)
        X = self.grid_surface
        plt.imshow(X, origin='lower')
        plt.colorbar()
        plt.title("Grid Pixel Surface")

        plt.subplot(1, 2, 2)  # Create the second subplot
        X = self.pixel_surface
        plt.imshow(X, origin='lower')
        plt.colorbar()
        plt.title("Pixel Surface")

        plt.tight_layout()  # Ensures that subplots do not overlap

        plt.show()  # Display the entire figure with both subplots

    def get_var_ds_file(self, var_id):
        filename = var_id.lower()+'.nc'
        file = os.path.join(self.data_out, filename)
        return(file)

    def get_var_id_ds(self, var_id):
        file = self.get_var_ds_file(var_id)
        if not os.path.exists(file):
            dims_global = ['lat_global', 'lon_global', 'days']
            days = list(self.days_i_t_per_var_id[var_id].keys())
            coords_global = {'lat_global': self.lat_global, 'lon_global': self.lon_global, 'days': days}
            da_global = xr.DataArray(None, dims = dims_global, coords = coords_global)
            ds = xr.Dataset({'global_pixel_surf': da_global})
            ds.to_netcdf(file)    
        else:
            # Open the existing NetCDF file using xarray
            ds = xr.open_dataset(file)
            # Check if the loaded dataset contains the required coordinates
            required_coordinates = ['lat_global', 'lon_global', 'days']
            missing_coordinates = [coord for coord in required_coordinates if coord not in ds.coords]
            if missing_coordinates:
                # Il est corrompu on dirait... ROBIN ALED
                print(f"The dataset is missing the following coordinates: {missing_coordinates}")

        return ds


## Kinda an issue that there is 3 funcs to basically do the same thing

    def compute_funcs_for_var_id(self, var_id='Prec'):
        """
        Save to netcdf all the funcs to apply to var_id, that were not already a key. 
        """
        var_ds = self.get_var_id_ds(var_id)         
        funcs = []
        keys = []
        keys_loaded = [key for key in list(var_ds.variables) if var_id in key] 
        
        # get the list of functions that have not been computed already and saved in var_ds 
        # maybe a function to handle that, and the accrosingfuncs.json would be appropriate     
        for func_name in self.func_names: # for now func_names is the same for all var_id, but it should be updated in a json to be in the same fashion that days_i_t_per_var_id
            key = '%s_%s'%(func_name,var_id)
            if key in keys_loaded : 
                print('%s already computed, skipping...'%key)
                # Here it should actually make another check that'd look at the days asked for computation in ditvi and compare to the one stored. 
                # making a temp ditvi with only missing days to compute for this var would then be great
                continue
            else :
                funcs.append(func_name) 
                keys.append(key)
        
        if len(funcs)>0 : 
            print(f"These keys : {keys} have to be loaded at .pkl level, or computed.")
            
            da_days_funcs = [[] for _ in funcs]
            days = list(self.days_i_t_per_var_id[var_id].keys())
            for day in days:
                da_funcs = self.regrid_funcs_and_save_by_day(day, var_id, funcs)
                for i, da_func in enumerate(da_funcs):
                    da_days_funcs[i].append(da_func)
                del da_funcs
                gc.collect()

            for da_day, key in zip(da_days_funcs, keys) : 
            ## concat the list of dataarrays along days dimensions
                da_var_regrid = xr.concat(da_day, dim = 'days')
                var_ds = var_ds.assign(**{key: da_var_regrid})
            
            file = self.get_var_ds_file(var_id)
            os.remove(file)
            var_ds.to_netcdf(file) ## this should update the stored .nc
            var_ds.close()
        
        else : print("nothing to compute then")


    def regrid_funcs_and_save_by_day(self, day, var_id='Prec', funcs=['max', 'mean']):
        """
        Compute multiple functions on new grid for a given day
        Save it as a datarray under a pickle file
        """
        outputs_da = [] ##Its a list of tuple (func, da_day_func_var_id)
        filename = f'{day}.pkl'  ## +26 meanPrecip and maxprecip computed for 40 days instead of 14
        filedirs = [os.path.join(os.getcwd() + self.settings["DIR_OUT"] + '/' + self.name, '%s_%s' % (str(func), var_id)) for func in funcs] #makes one directory per key. 
        
        ### !!!! ###
        # realizes here, that in this case some combinaisons of func+variables will be unwanted #
        # so they'll have to be sorted at key level ##
        filepaths = []
        for filedir in filedirs :
            os.makedirs(filedir, exist_ok=True)
            
            filepaths.append(os.path.join(filedir, filename))

        ## load it if it doesn't exist. Here should adapt funcs to load only those that are not saved.
        to_compute_bool = np.array([not os.path.isfile(filepath) for filepath in filepaths])
        print(to_compute_bool, funcs)
        funcs_to_load = list(np.array(funcs)[~to_compute_bool])
        funcs_to_compute = list(np.array(funcs)[to_compute_bool]) ## could be renamed funcs
        
        filepaths_to_load = list(np.array(filepaths)[~to_compute_bool])
        filepaths_to_save = list(np.array(filepaths)[to_compute_bool])

        # compute
        print(f"compute {funcs_to_compute} for {var_id} at {day}")
        var_regridded_per_funcs = self.regrid_funcs_for_day(day, var_id=var_id, funcs_to_compute=funcs_to_compute)

        # save as pickle file in the directory ${func}_Prec
        # here us funcs_to_compute and filepaths to save them. 
        for filepath, var_regridded, func in zip(filepaths_to_save, var_regridded_per_funcs, funcs_to_compute):
            with open(filepath, 'wb') as f:
                da_day = xr.DataArray(var_regridded[0], dims=['lat_global', 'lon_global', 'days'], 
                                  coords={'lat_global': self.lat_global, 'lon_global': self.lon_global, 'days': [day]})
                pickle.dump(da_day, f)
                outputs_da.append(da_day)

        print('%s already exists' % funcs_to_load)
        # load as pickle file in the directory ${func}_Prec
        # here use funcs_to_load, should be starting
        for filepath, func in zip(filepaths_to_load, funcs_to_load) : 
            with open(filepath, 'rb') as f:
                da_day = pickle.load(f)
                outputs_da.append(da_day)
        del var_regridded_per_funcs
        gc.collect()
        
        return outputs_da

    def regrid_funcs_for_day(self, day, var_id='Prec', funcs_to_compute=['max', 'mean']):
        """
        Compute multiple functions on new grid for a given day and return the results as a list.

        Args:
            day (str): The specific day for which regridding will be performed.
            var_id (str, optional): A variable identifier that determines what kind of data is being regridded.
            funcs_to_compute (list, optional): A list of functions to be applied to the data during regridding.

        Returns:
            list: A list of regridded data for each function in funcs_to_compute, in the same order.

        """

        def regrid_single_time_step(i_t, var_id, funcs_to_compute):
            """
            Regrid data for a single time step.

            Args:
                idx (int): Index representing the time step.
                var_id (str): A variable identifier.

            Returns:
                list: A list of regridded data for each function in funcs_to_compute.
            """
            # Get the data for the day
            var_current = load_var(self, var_id, i_t)

            # Compute each func(var) for the current time step
            results = []
            for func in funcs_to_compute:
                if var_current is not None:
                    var_regrid_idx = getattr(self, 'spatial_%s_data_from_center_to_global' % func)(var_current)
                else:
                    # Create an xarray filled with nans
                    var_regrid_idx = self.create_empty_array()
                    print('Regridding NaNs')

                results.append(var_regrid_idx)

            del var_current
            gc.collect()

            return results
        
        all_i_t_for_day_per_func = [[] for _ in funcs_to_compute]

        for i_t in self.days_i_t_per_var_id[var_id][day]:
            # Regrid time step for each func
            results = regrid_single_time_step(i_t, var_id, funcs_to_compute)
            for i_f, result in enumerate(results):
                all_i_t_for_day_per_func[i_f].append(result)

            del results
            gc.collect()

        # Stack and aggregate the data for each func
        # This stacking-aggregation is dependent of funcs to compute,
        # don't forget to modify this step if you're adding a new function 
        # or to include it in the method/function... at day level then.
        day_per_func = [[] for _ in funcs_to_compute]

        for i_f, func in enumerate(funcs_to_compute):
            stacked_array = np.stack(all_i_t_for_day_per_func[i_f], axis=0)
            aggregated_array = getattr(np, 'nan%s' % func)(stacked_array, axis=0)
            day_for_func = np.expand_dims(aggregated_array, axis = -1)
            day_per_func[i_f].append(day_for_func)

        return day_per_func    

### Functions add-ons for special regridding, built within class

    def spatial_mean_data_from_center_to_global(self, data_on_center):
        """
        Returns the mean of data_on_center, weighted by the relative value of the initial pixel divided by the 
        final grid_surface. 
        """
        weights = self.pixel_surface
        x = data_on_center*weights if type(data_on_center) == np.ndarray else data_on_center.values*weights
        X = self.sum_data_from_center_to_global(data_on_center = x)
        global_weights = self.grid_surface

        return X/global_weights 

    def spatial_max_data_from_center_to_global(self, data_on_center):
        x = data_on_center if type(data_on_center) == np.ndarray else data_on_center.values
        X = np.zeros((self.n_lat, self.n_lon))
        if not self.fast : alpha_max = self.__build_alpha_max__()
        for i, slice_i_lat in enumerate(self.slices_i_lat):
            for j, slice_j_lon in enumerate(self.slices_j_lon[i]):
                if self.verbose : 
                    print(slice_i_lat, slice_j_lon)
                if not self.fast :
                    m = np.nanmax(x[slice_i_lat, slice_j_lon].flatten())
                    b = np.nanmax((x[self.i_min[i,j], slice_j_lon]*alpha_max[self.i_min[i,j], slice_j_lon]).flatten())
                    t = np.nanmax((x[self.i_max[i,j], slice_j_lon]*self.alpha_max[self.i_max[i,j], slice_j_lon]).flatten())
                    l = np.nanmax((x[slice_i_lat, self.j_min[i,j]]*self.alpha_max[slice_i_lat, self.j_min[i,j]]).flatten())
                    r = np.nanmax((x[slice_i_lat, self.j_max[i,j]]*self.alpha_max[slice_i_lat, self.j_max[i,j]]).flatten())
                    blc = (x[self.i_min[i,j], self.j_min[i,j]]*self.alpha_max[self.i_min[i,j], self.j_min[i,j]]).flatten()
                    btc = (x[self.i_min[i,j], self.j_max[i,j]]*self.alpha_max[self.i_min[i,j], self.j_max[i,j]]).flatten()
                    tlc = (x[self.i_max[i,j], self.j_min[i,j]]*self.alpha_max[self.i_max[i,j], self.j_min[i,j]]).flatten()
                    trc = (x[self.i_max[i,j], self.j_max[i,j]]*self.alpha_max[self.i_max[i,j], self.j_max[i,j]]).flatten()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        X[i, j] = np.nanmax(np.array([m, b, t, l, r, blc, btc, tlc, trc],dtype=object))
                
                elif self.fast :
                    X[i,j] = np.nanmax(x[slice_i_lat, slice_j_lon].flatten())
        # print(X.shape, X)
        del x 
        gc.collect()
        return X    

