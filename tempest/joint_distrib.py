import numpy as np
import os
import time
import sys
import pickle
import time
import xarray as xr 

from skimage import measure # pylance: disable=import-error 
from scipy.optimize import curve_fit
import warnings
from math import ceil 

from .distribution import Distribution
# from .load_toocan import load_toocan

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from .plots.plot2d import set_frame_invlog, show_joint_histogram
from .plots.hist import simple_hist

class JointDistribution():
    """
    Creates a joint distribution for two precipitations variables based on Grid prec.nc
    """
    
    def __init__(self, grid, storm_tracker = None, nd=4, var_id = "Prec", var_id_1 = "mean_Prec", var_id_2 = "max_Prec", dist_mask = True, overwrite=False, verbose = False, regionalize = False, dist_bintype = "invlogQ"):
        """Constructor for class Distribution.
        Arguments:
        - name: name of reference variable
        - distribution1, distribution2: marginal distributions of two reference variables
        - overwrite: option to overwrite stored data in object
        """
        
        # Inheritance by hand because __super__() speaks too much 
        self.grid = grid
        self.verbose = verbose
        self.name = grid.casestudy.name
        self.settings = grid.settings
        self.var_id_1 = var_id_1
        self.var_id_2 = var_id_2
        self.regionalize = regionalize
        self.nd = nd
        self.dist_bintype = dist_bintype
        self.ditvi = grid.casestudy.days_i_t_per_var_id
        self.var_id = var_id
        self.prec = grid.get_var_id_ds(self.var_id) ## weird behavior, can't modify var_id input

        if type(dist_mask) is not bool:
            if dist_mask.shape == self.prec[var_id_1].shape:
                self.dist_mask = dist_mask & xr.where(self.prec.mean_Prec > 0.001, True, False) #self.prec.Treshold_cond_alpha_50_Prec > 2
        elif dist_mask == True : 
            dist_mask = xr.where(self.prec.mean_Prec > 0.01, True, False)
        elif dist_mask == False : 
            dist_mask = True

        
        self.shape = np.shape(self.prec[self.var_id_1].to_numpy())
        
        self.sample1 = self.prec[var_id_1].where(dist_mask).to_numpy().ravel()
        self.sample2 = self.prec[var_id_2].where(dist_mask).to_numpy().ravel()
        
        self.overwrite = overwrite

        cwd = os.getcwd()
        rel_dir_out = self.settings["DIR_OUT"]
        abs_dir_out = os.path.join(cwd, rel_dir_out)
        self.dir_out = os.path.join(abs_dir_out, self.name)
        if not os.path.exists(self.dir_out):
            print("First instance of this Joint Distribution, so setting overwrite to true")
            os.makedirs(self.dir_out)
            self.overwrite = True

        # Retrieves self.dist1 and self.dist2
        self.get_distribs()
        self.bins1 = self.dist1.bins
        self.bins2 = self.dist2.bins

        jd_name = self.dist1.name +'_joint_'+self.dist2.name
        self.jd_path = os.path.join(self.dir_out, jd_name)
        if not os.path.exists(self.jd_path):
            os.makedirs(self.jd_path)

        if overwrite : 
            print("Overwrite set to true, so computing basics and saving them")
            self.density = None
            self.bin_locations_stored = False

            self.compute_distribution(self.sample1, self.sample2)
            self.compute_normalized_density(self.sample1, self.sample2)
            
            self.digit_3d_1, self.digit_3d_2 = self.compute_conditional_locations()
            
            self.save_basics_to_npy()
        else : 
            print("Overwrite set to false so loading basics attributes from .npy")
            self.load_from_npy()
        
    
        ## Make a class out of this so that it only has to be loaded once (super long like 30sec to 2mins....)
        if storm_tracker is not None:
            # is this best practice ?
            self.ds_storm = storm_tracker.ds_storms
            self.file_storms = storm_tracker.file_storms
            # self.storms = storm_tracker.storms
            # self.label_storms = storm_tracker.label_storms
            # self.dict_i_storms_by_label = storm_tracker.dict_i_storms_by_label
            self.labels_regridded_yxtm = storm_tracker.labels_regridded_yxtm
            self.mask_labels_regridded_yxt = storm_tracker.mask_labels_regridded_yxt

            if self.verbose:
                print("Retrieve labels in jdist")
            start_time = time.time()
            self.labels_in_jdist = self.get_labels_in_jdist_bins()
            print(f"Time elapsed for propagating all labels: {time.time() - start_time:.2f} seconds")
            if self.regionalize : self.get_labels_per_region()

    def __repr__(self):
        """Creates a printable version of the Distribution object. Only prints the 
        attribute value when its string fits is small enough."""
        out = '< JointDistribution object:\n'
        # print keys
        for k, v in self.__dict__.items():
            out = out + f' . {k}: '
            if sys.getsizeof(str(v)) < 80:
                # show value
                out = out + f'{v}\n'
            else:
                # show type
                out = out + f'{type(v)}\n'
        out = out + ' >'
        return out

    def __str__(self):
        """Override string function to print attributes"""
        str_out = ''
        for k, v in self.__dict__.items():
            if '__' not in k:
                a_str = str(v)
                if 'method' not in a_str:
                    str_out = str_out + f"{k} : {a_str}\n"
        return str_out

    def get_distribs(self):
        name_dist1 = 'dist_' + self.var_id_1
        path_dist1 = os.path.join(self.dir_out, name_dist1)
        name_dist2 = 'dist_' + self.var_id_2
        path_dist2 = os.path.join(self.dir_out, name_dist2)

        if self.overwrite:
            self.dist1 = Distribution(name = name_dist1,  bintype = self.dist_bintype, nd = self.nd, fill_last_decade=True)
            self.dist1.compute_distribution(self.sample1)
            with open(path_dist1, 'wb') as file:
                pickle.dump(self.dist1, file)

            self.dist2 = Distribution(name = name_dist2,  bintype = self.dist_bintype, nd = self.nd, fill_last_decade=True)
            self.dist2.compute_distribution(self.sample2)
            with open(path_dist2, 'wb') as file:
                pickle.dump(self.dist2, file)
            print("Distribs have been recomputed because overwrite is set to True")

        elif os.path.exists(path_dist1) and os.path.exists(path_dist2):
            with open(path_dist1, 'rb') as file:
                self.dist1 = pickle.load(file)
            with open(path_dist2, 'rb') as file:
                self.dist2 = pickle.load(file)
            print("Distribs loaded")
        
        else : 
            print("Overwrite set to false, but not distribs found !")

    def save_basics_to_npy(self):
        """
        This function is called after the first instance of the object. It stores the very basics arrays required for functionning
        Then after computing any new array saving into this dir must be made in the same fashion
        """
        self.npy_attributes = {
            'bincount': self.bincount,
            'norm_density': self.norm_density,
            'digit_3d_1': self.digit_3d_1,
            'digit_3d_2': self.digit_3d_2
        }

        for attr, value in self.npy_attributes.items():
            attr_filename = attr + '.npy'
            attr_file_path = os.path.join(self.jd_path, attr_filename)
            np.save(attr_file_path, value)

    def load_from_npy(self):
        """
        This function loads everything that has been saved
        """
        files = os.listdir(self.jd_path)
        self.npy_attributes = {}
        for file in files:
            if file.endswith('.npy'):
                attr_name = os.path.splitext(file)[0]
                file_path = os.path.join(self.jd_path, file)
                attr_value = np.load(file_path)
                setattr(self, attr_name, attr_value)
                self.npy_attributes[attr_name] = attr_value

    def format_dimensions(self,sample):
        """Reshape the input data, test the validity and returns the dimensions
        and formatted data.
    
        Arguments:
        - sample: here we assume data is horizontal, formats it in shape (Ncolumns,)
        Controls if it matches the data used for the control distribution. If
        not, aborts.
        """
    
        # Get shape
        sshape = sample.shape
        # Initialize default output
        sample_out = sample
        # Get dimensions and adjust output shape
        if len(sshape) > 1: # reshape
            sample_out = np.reshape(sample,np.prod(sshape))
        Npoints, = sample_out.shape
        
        # Test if sample size is correct to access sample points
        if Npoints != self.size:
            print("Error: used different sample size")
    
        return sample_out

    def store_sample_points(self,sample1,sample2,sizemax=50,verbose=False,method='shuffle_mask'):
        """Find indices of bins in the sample data, to get a mapping or extremes 
        and fetch locations later
        """
    
        if self.bin_locations_stored and not self.overwrite:
            pass

        if verbose:
            print("Finding bin locations...")

        # print(sample.shape)
        sample1 = self.formatDimensions(sample1)
        sample2 = self.formatDimensions(sample2)
        # print(sample.shape)
        
        # Else initalize and find bin locations
        self.bin_locations = [[[] for _ in range(self.distribution2.nbins)] for _ in range(self.distribution1.nbins)]
        self.bin_sample_size = [[0 for _ in range(self.distribution2.nbins)] for _ in range(self.distribution1.nbins)]

        if method == 'shuffle_mask':

            if verbose: print('bin #: ',end='')
            # compute mask for each bin, randomize and store 'sizemax' first indices
            for i_bin in range(self.distribution1.nbins):
                
                for j_bin in range(self.distribution2.nbins):
                    
                    if verbose: print('%d,%d..'%(i_bin,j_bin),end='')
                    
                    # compute mask
                    mask1 = np.logical_and(sample1.flatten() >= self.distribution1.bins[i_bin],
                                sample1.flatten() < self.distribution1.bins[i_bin+1])
                    mask2 = np.logical_and(sample2.flatten() >= self.distribution2.bins[j_bin],
                                sample2.flatten() < self.distribution2.bins[j_bin+1])
                    mask = np.logical_and(mask1,mask2)
                    # get all indices
                    ind_mask = np.where(mask)[0]
                    # shuffle
                    np.random.seed(int(round(time.time() * 1000)) % 1000)
                    np.random.shuffle(ind_mask)
                    # select 'sizemax' first elements
                    self.bin_locations[i_bin][j_bin] = ind_mask[:sizemax]
                    # self.bin_sample_size[i_bin] = min(ind_mask.size,sizemax) # cap at sizemax
                    self.bin_sample_size[i_bin][j_bin] = ind_mask.size # count all points there
                    
                if verbose: print()

        # If reach this point, everything should have worked smoothly, so:
        self.bin_locations_stored = True

    def compute_distribution(self,sample1,sample2,vmin=None,vmax=None,minmode=None):

        """Compute ranks, bins, percentiles and corresponding probability densities.
        Arguments:
            - sample1,2: 1D numpy array of values
        Computes:
            - ranks, percentiles, bins and probability densities"""

        if not self.overwrite:
            pass

        # Compute probability density
        self.bincount, _, _ = np.histogram2d(x=sample1.flatten(),y=sample2.flatten(),bins=(self.bins1,self.bins2),density=False)        
 
    def compute_joint_density(self, sample1, sample2, method='default'):
        """Compute joint density. Method 'default' uses np.histogram2d, 'manual' uses np.ditigize and np.bincount."""
        
        if self.sample1.shape == sample2.shape:
            
            if method == 'default':

                # Compute bin count
                self.bincount, _, _ = np.histogram2d(x=self.sample1.flatten(),y=self.sample2.flatten(),bins=(self.bins1,self.bins2),density=False)

                # Compute probability density
                self.density, _, _ = np.histogram2d(x=self.sample1.flatten(),y=self.sample2.flatten(),bins=(self.bins1,self.bins2),density=True)

            elif method == 'manual':

                digit1 = np.digitize(self.sample1, self.bins1, right = True)
                digit2 = np.digitize(self.sample2, self.bins2, right = True)
                # if verbose : print(digit1, digit2)
                Ntot = self.sample1.size
            
                l1, l2 = len(self.bins1)-1, len(self.bins2)-1 # BF: adjusted nbins to match np.histogram2d

                dx_1 = np.diff(self.bins1)
                dx_2 = np.diff(self.bins2)

                # initialize
                self.bincount = np.zeros(shape = (l1, l2))
                self.density = np.zeros(shape = (l1, l2))
                
                # compute
                for i2 in range(1,l2): 
                    
                    idx = tuple(np.argwhere(digit2==i2).T)
                    
                    # bin count
                    self.bincount[:, i2-1] = np.bincount(digit1[idx], minlength=l1+1)[1:] # BF: adjusted nbins to match np.histogram2d (removed first element of bincount)
                    
                    # density = bincount / Ntot / bin width
                    self.density[:, i2-1] = self.bincount[:, i2-1]/Ntot/dx_1/dx_2[i2-1]
    
    def compute_normalized_density(self, verbose = False, method='default'):
        """Compute joint density normalized by the expected density of independent variables : N_ij * N_tot / N_i / N_j."""

        if self.bincount is None:
            
            self.compute_joint_density(self.sample1, self.sample2, method = method)
        
        l1, l2 = len(self.bins1)-1, len(self.bins2)-1 # BF: adjusted nbins to match np.histogram2d

        digit1 = np.digitize(self.sample1, self.bins1, right = True)
        digit2 = np.digitize(self.sample2, self.bins2, right = True)

        N1 = [np.sum(digit1==i1) for i1 in range(l1)]
        N2 = [np.sum(digit2==i2) for i2 in range(l2)]
        Ntot = self.sample1.size
        with np.errstate(divide='ignore'):
            Norm = Ntot / np.outer(N1, N2)
        Norm[np.isinf(Norm)] = 1

        self.norm_density = Norm * self.bincount
        
    def compute_conditional_locations(self, var_days=None):
        if var_days is not None : 
            var_shape = self.prec[self.var_id_1].sel(days = var_days).shape
        else : 
            var_days = list(self.prec.days.values)  
            var_shape = self.shape
            
        digit1 = np.digitize(self.prec[self.var_id_1].sel(days = var_days).values.flatten(), self.bins1, right = True)
        digit2 = np.digitize(self.prec[self.var_id_2].sel(days = var_days).values.flatten(), self.bins2, right = True)
        
        digit1_3d = np.reshape(digit1,var_shape)
        digit2_3d = np.reshape(digit2,var_shape)
        
        return digit1_3d, digit2_3d
        
    # def compute_conditional_data_over_density(self, data = None, mask = None):
    #     """
    #     TODO Could be adapted to use joint_digit BF formalism but seems to work well as is, be careful
    #     """         
    #     var_days = list(data.days.values)  
    #     reduced_prec = self.prec.sel(days = var_days)                     
    #     digit1 = np.digitize(reduced_prec[self.var_id1], self.bins1, right = True)
    #     digit2 = np.digitize(reduced_prec[self.var_id2], self.bins2, right = True)
        
    #     l1, l2 = len(self.bins1)-1, len(self.bins2)-1 # BF: adjusted nbins to match np.histogram2d

    #     if data is not None : 
    #         data_over_density = np.zeros(shape=(l1,l2))
    #         if mask is not None : 
    #             mask = data.where(mask)
    #     for i2 in range(l2): 
    #         if data is not None: ## TEST AND DEBUG THIS
    #             for i1 in range(l1):
    #                 data_idx = tuple(np.argwhere((digit1==i1) & (digit2==i2)).T)
    #                 if len(data_idx)>0 :
    #                     data_over_density[i1, i2] = np.nanmean(data.values.flatten()[data_idx])
    #                 else : data_over_density[i1, i2] = 0
    #     if data is not None:
    #         return data_over_density
        
    def compute_conditional_sum(self, sample1, sample2, data = None):
        """
        TODO
        """                        
        digit1 = np.digitize(sample1.flatten(), self.bins1, right = True)
        digit2 = np.digitize(sample2.flatten(), self.bins2, right = True)
        
        l1, l2 = len(self.bins1)-1, len(self.bins2)-1 # BF: adjusted nbins to match np.histogram2d

        if data is not None : data_over_density = np.zeros(shape=(l1,l2))

        for i2 in range(1,l2): 
            if data is not None: ## TEST AND DEBUG THIS
                for i1 in range(l1):
                    data_idx = tuple(np.argwhere((digit1==i1) & (digit2==i2)).T)
                    if len(data_idx)>0 :
                        data_over_density[i1, i2] = np.nansum(data.flatten()[data_idx])
                    else : data_over_density[i1, i2] = 0

        if data is not None:
            
            return data_over_density
        
    def joint_digit(self, d1, d2):
        jdig = 100*d1 + d2
        return jdig
    
    def make_mask(self):
        #-- create masks for a region on joint distribution
        dig_1D = np.digitize(self.dist1.ranks, self.dist2.ranks)
        dig_2D_i, dig_2D_j = np.meshgrid(dig_1D,dig_1D)
        self.dig_2D = self.joint_digit(*np.meshgrid(dig_1D,dig_1D))

        # mask a square over 90th percentile
        self.mask_jdist_90_90 = np.outer(self.dist1.ranks >= 90, self.dist2.ranks >= 90)
        self.mask_jdist_90_0 = np.outer(self.dist1.ranks >= 90, self.dist2.ranks >= 0)
        self.mask_jdist_0_90 = np.outer(self.dist1.ranks >= 0, self.dist2.ranks >= 90)
        self.mask_jdist_below_90 = np.outer(self.dist1.ranks < 90, self.dist2.ranks < 90)
        # mask above diagonal
        self.mask_lower_diag = dig_2D_i <= dig_2D_j
        self.mask_upper_diag = dig_2D_i > dig_2D_j
        # mask correlated and anticorrelated
        self.mask_corr = self.norm_density > 1
        self.mask_anticorr = np.logical_and(self.norm_density < 1,self.bincount>0)
        # mask left of branch 1
        self.mask_branch1 = np.logical_and(self.mask_upper_diag,self.mask_anticorr)
        self.mask_branch1_90 = np.logical_and(self.mask_branch1,self.mask_jdist_0_90)
        # mask below to branch 2
        self.mask_branch2 = np.logical_and(self.mask_lower_diag,self.mask_anticorr)
        self.mask_branch2_90 = np.logical_and(self.mask_branch2,self.mask_jdist_90_0)
        # mask_middle, upper and lower diagonal
        self.mask_coloc_c = np.logical_and(self.mask_upper_diag,self.mask_corr)
        self.mask_coloc_c_90 = np.logical_and(self.mask_coloc_c,self.mask_jdist_0_90)
        self.mask_coloc_ac = np.logical_and(self.mask_lower_diag,self.mask_corr)
        self.mask_coloc_ac_90 = np.logical_and(self.mask_coloc_ac,self.mask_jdist_90_0)

        # create mask
        self.mask_show = 1.*self.mask_branch1_90 + 2.*self.mask_coloc_c_90 + 3.*self.mask_coloc_ac_90 + 4.*self.mask_branch2_90
        self.mask_show[self.mask_show == 0] = np.nan

    def _fit_branches(self,cont,N):

        def func(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        if cont.__class__ is list:
            seg_1 = np.flip(cont[0],axis=1)
        else:
            seg_1 = cont.allsegs[0][0]
            
        # Branch 1 -- end of contour (upward branch)
        xdata_1 = seg_1[-N:,0]
        y_1 = ydata_1 = seg_1[-N:,1]

        # fit
        popt_1, pcov_1 = curve_fit(func, ydata_1, xdata_1,p0=(-10,1,0))
        x_1 = func(ydata_1, *popt_1)
        
        # Branch 2 -- start of contour
        x_2 = xdata_2 = seg_1[:N,0]
        ydata_2 = seg_1[:N,1]

        # fit
        popt_2, pcov_2 = curve_fit(func, xdata_2, ydata_2,p0=(-10,1,0))
        y_2 = func(xdata_2, *popt_2)
        
        return popt_1, x_1, y_1, popt_2, x_2, y_2, func

    def plot(self, mask = True, branch = False, fig=None, ax=None):
        self.make_mask()

        if fig == ax == None :
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.8, 4.85))
        
        Z = self.norm_density.T

        title = f"Normalized density"
        scale = 'log'
        vbds = (1e-3, 1e3)
        cmap = plt.cm.BrBG

        # -- Frame
        ax_show = ax.twinx().twiny()
        ax = set_frame_invlog(ax, self.dist1.ranks, self.dist2.ranks)
        ax.set_xlabel(r"1$^\circ\times 1$day extremes")
        ax.set_ylabel(r"4km-30mn extremes")
        ax.set_title(title)

        # -- Density
        pcm = show_joint_histogram(ax_show, Z, scale=scale, vmin=vbds[0], vmax=vbds[1], cmap=cmap)

        # -- Masks multiscale categories and colorbar
        if mask : 
            pcm_mask = ax_show.imshow(self.mask_show.T,alpha=1,origin='lower')
            cb = fig.colorbar(pcm_mask, ax=ax_show)
            cb.set_label("Multiscale categories") 
            
        else : 
            cb = fig.colorbar(pcm, ax=ax_show)
            cb.set_label("Normalized density" ) 

        if branch : 
            # -- Branches
            cont = measure.find_contours(Z, 1)
            N = 60
            # fit
            popt_1, x_1, y_1, popt_2, x_2, y_2, func = self._fit_branches(cont,N)
            x_branch_2 = y_branch_1 = np.linspace(2,45,45)
            y_branch_2 = func(x_branch_2,*popt_2)
            x_branch_1 = func(y_branch_1,*popt_1)

            # show branches
            ax_show.plot(x_branch_1,y_branch_1,'k--')
            ax_show.plot(x_branch_2,y_branch_2,'k--')

            # show 1-1 line
            ax_show.plot(x_branch_2,x_branch_2,'k--')
        
        return ax, cb
            
    def get_mask_yxt(self, d1, d2, var_days = None): #regional = False, lat_slice = None, lon_slice = None
        dj = self.joint_digit(d1+1, d2+1) # because np.digitize returns 1 for the i_bin 0
        
        if var_days is not None : 
            d3_1, d3_2 = self.compute_conditional_locations(var_days = var_days)
        else : 
            d3_1, d3_2 = self.digit_3d_1, self.digit_3d_2
        dj_3d = self.joint_digit(d3_1, d3_2)
        
        mask = dj_3d == dj
        
        ## TODO : just use masks compatibility for regionality... with .where
        # if regional:
        #     region_mask = np.zeros_like(mask, dtype = bool)
        #     if lat_slice is not None and lon_slice is not None:
        #         if lon_slice.start < lon_slice.stop:
        #             region_mask[lat_slice, lon_slice] = mask[lat_slice, lon_slice] 
        #         elif lon_slice.stop < lon_slice.start:
        #             region_mask[lat_slice, slice(0, lon_slice.stop)]= mask[lat_slice, slice(0, lon_slice.stop)]
        #             region_mask[lat_slice, slice(lon_slice.start, 360)]= mask[lat_slice, slice(lon_slice.start, 360)]
        #     mask = region_mask
            
        return mask
        
    def get_mask_yxt_from_mask_jdist(self, mask_jdist):
        """
        For each joint extreme in bin (i,j), return mask in the spatiotemporal domain.
        """
        
        mask_yxt_all = False
        i_j_mask = np.where(mask_jdist)
        
        for i,j in zip(i_j_mask[0],i_j_mask[1]):
            mask_yxt = self.get_mask_yxt(i,j)
            mask_yxt_all = np.logical_or(mask_yxt_all,mask_yxt)
            
        return mask_yxt_all

    def get_coord_values(self, coordname):
        """
        I don't understand this func wery well, but it seems
        to retrieve 
        """
        if coordname == 'lat':
            x_native = self.grid.lat_centers
            x_regrid = self.prec['lat_global'].values
        elif coordname == 'lon':
            x_native = self.grid.lon_centers
            x_regrid = self.prec['lon_global'].values
        
        Nx = len(x_regrid)
        x_bnds = np.round(x_native[0],2), np.round(x_native[-1],2)
        dx = np.diff(x_bnds)/Nx

        edges = np.arange(x_bnds[0],x_bnds[1]+dx,dx)
        centers = np.convolve(edges,[0.5,0.5],'valid')
        
        return centers  

    def make_map(self, mask_yxt, data = None, func = np.sum, threshold = (-np.inf, np.inf)):
        ## image
        # cmap = plt.cm.bone_r
        # cmap = plt.cm.Blues
        cmap = plt.cm.afmhot_r
        # cmap_mcs = plt.cm.get_cmap('Accent', 10)

        # compute figure size
        dlon = np.diff((self.grid.casestudy.lon_slice.start, self.grid.casestudy.lon_slice.stop))[0] % 360
        if dlon == 0: dlon = 360
        dlat = np.diff((self.grid.casestudy.lat_slice.start, self.grid.casestudy.lat_slice.stop))[0]
        Lx_fig = 15
        Lx_cbar = 1.5
        Ly_title = 1
        Ly_fig = (Lx_fig-Lx_cbar)/dlon*dlat + Ly_title
        print('figure size =',Lx_fig,Ly_fig)
        
        # initialize figure
        fig = plt.figure(figsize=(Lx_fig,Ly_fig))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
        
        # coords
        lat_1d = self.get_coord_values('lat')
        lon_1d = self.get_coord_values('lon')
        
        lon_meshgrid, lat_meshgrid = np.meshgrid(lon_1d, lat_1d)
        
        # data
        if func == np.sum:
            Z = np.sum(mask_yxt,axis=-1) # count
            Next = np.sum(Z)
        elif func == "data_weighted":
            sum_weights = np.sum(data, axis=-1)
            mask = sum_weights == 0
            mask = np.repeat(mask[..., np.newaxis], 20, axis = -1)
            masked_values = np.ma.masked_array(mask_yxt, mask)
            masked_weights = np.ma.masked_array(data, mask)
            weighted_means = np.ma.average(masked_values, axis=-1, weights=masked_weights)
            Z = weighted_means.filled(np.nan)
            Next = np.nansum(Z)
        # show
        # im = ax.pcolormesh(np.ravel(lonarray_dyamond),np.ravel(latarray_dyamond),np.ravel(Z),transform=ccrs.PlateCarree(),alpha=0.9,cmap=cmap)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            im = ax.pcolormesh(lon_meshgrid, lat_meshgrid, Z, transform=ccrs.PlateCarree(), alpha=0.9, cmap=cmap)
        # im.set_clim(*clim)

        # we draw coasts
        ax.coastlines('110m')
        # Adding gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(range(-180, 181, 5))
        gl.ylocator = mticker.FixedLocator(range(-90, 91, 5))
        gl.top_labels = False  # Turn off labels on top x-axis
        gl.right_labels = False  # Turn off labels on right y-axis
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # add personal legend
        ax.text(0.93,0.05,"N = %d"%Next,transform=ax.transAxes, fontsize = 9)
        

        # Colorbar
        x,y,w,h = ax.get_position().bounds
        dx = w/60
        cax = plt.axes([x+w+1.5*dx,y,dx,h])
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.set_ylabel('Bincount (#)')

        return ax
    
## MCS labels

    def labels_in_mask_yxt(self, mask_yxt):
        """
        For each joint extreme in mask for joint extremes, merge MCS labels occurring in their spatiotemporal occurrence.
        LATER: also store their relative area.
        """
        if np.sum(mask_yxt) == 0:
            labels = self.labels_regridded_yxtm[False]
        else:
            labels = self.labels_regridded_yxtm[mask_yxt]
        unique_labels = np.unique(labels)
        
        return unique_labels
    
    def labels_in_joint_bin(self, i_bin, j_bin, region_mask=None):
        """
        For each joint extreme in bin (i,j), merge MCS labels occurring in their spatiotemporal occurrence.
        LATER: also store their relative area.
        """
        mask_yxt = self.get_mask_yxt(i_bin,j_bin)
        if region_mask is not None : 
            mask_yxt = np.logical_and(mask_yxt, region_mask)
        labels_i_j = self.labels_in_mask_yxt(mask_yxt)
        return labels_i_j
    
    def count_mcs_in_jdist(self):
        """
        Return matrix of MCS counts in each bin of the joint distribution
        """

        n_i,n_j = self.bincount.shape
        count_ij = np.full((n_i,n_j),np.nan)

        for i_bin in range(n_i):
            for j_bin in range(n_j):
                # print(i_bin,j_bin)
                labels_bin = self.labels_in_joint_bin(i_bin,j_bin)
                labels_unique_bin = np.unique(labels_bin)
                count_ij[i_bin,j_bin] = labels_unique_bin.size

        return count_ij
    
    def get_mask_yxt_labels(self, labels):
        """
        Returns mask where these labels occur in x-y-t space
        """
        mask_all_labels = False
        for label in labels:
            mask_label = (self.labels_regridded_yxtm == label)
            mask_all_labels = np.logical_or(mask_all_labels,mask_label)
            
        return mask_all_labels
    
    def get_mcs_bin_fraction(self, bin_noise_treshold = 4, region_mask = None):
        n_i, n_j = self.bincount.shape
        bin_fraction_mcs = np.full((n_i,n_j),np.nan)
        bin_noise = np.full((n_i,n_j),np.nan)
        bin_counts = np.full((n_i,n_j),np.nan)

        for i_bin in range(n_i):
            for j_bin in range(n_j):
                # where bin falls in x-y-t
                mask_bin_yxt = self.get_mask_yxt(i_bin,j_bin)
                if region_mask is not None :
                    mask_bin_yxt = np.logical_and(mask_bin_yxt, region_mask)

                # where bin falls in x-y-t and MCS occurs
                mask_bin_with_mcs_yxt = np.logical_and(mask_bin_yxt, self.mask_labels_regridded_yxt)
                # number of points in joint mask
                count_bin_mcs = np.sum(mask_bin_with_mcs_yxt)
                # number of point in bin mask
                count_bin = np.sum(mask_bin_yxt) 
                bin_counts[i_bin,j_bin] = count_bin
                # store this fraction
                if count_bin >= bin_noise_treshold:
                    bin_fraction_mcs[i_bin,j_bin] = count_bin_mcs/count_bin
                elif count_bin > 0:
                    bin_noise[i_bin,j_bin] = 1
                # include noisy points in bin_fraction_mcs
                # if count_bin > 0:
                #     bin_fraction_mcs[i_bin,j_bin] = count_bin_mcs/count_bin
                #     if count_bin <= 4:
                #         bin_noise[i_bin,j_bin] = 1
        
        # return this fraction
        return bin_fraction_mcs, bin_noise, bin_counts

    def plot_data(self, data, data_noise = None, scale = 'linear', cmap = plt.cm.RdBu_r, branch = False, label = '', fig =None ,ax = None, vbds = (None, None), cb_bool = True):
        """
        TODO : mask data, keep bins
        """
        self.make_mask()
        
        if fig==None and ax==None: 
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4.85))
        
        Z_nd = self.norm_density.T
        Z = data.T
        

        # Should be passed as **kwargs
        title = f"Data over Normalized density"
        cmap = cmap

        # -- Frame
        ax_show = ax.twinx().twiny()
        ax = set_frame_invlog(ax, self.dist1.ranks, self.dist2.ranks)
        ax.set_xlabel(self.var_id_1)
        ax.set_ylabel(self.var_id_2)
        # ax.set_title(title)

        # -- Density
        pcm = show_joint_histogram(ax_show, Z, scale=scale, vmin=vbds[0], vmax=vbds[1], cmap=cmap)
        if data_noise is not None :
            Z_noise = data_noise.T
            show_joint_histogram(ax_show, Z_noise, scale=scale, vmin=vbds[0], vmax=vbds[1], cmap=cmap, alpha=0.1)

        # -- Colorbar
        if cb_bool : 
            cb = fig.colorbar(pcm, ax=ax_show)
            # cb.set_label('Normalized density')
            cb.set_label(label)
        else :
            cb = None
        
        # # -- Masks multiscale categories
        # ax_show.imshow(self.mask_show.T,alpha=0.5,origin='lower')

        if branch : 
            # -- Branches
            cont = measure.find_contours(Z, 1)
            N = 40
            # fit
            popt_1, x_1, y_1, popt_2, x_2, y_2, func = self._fit_branches(cont,N)
            x_branch_2 = y_branch_1 = np.linspace(2,45,45)
            y_branch_2 = func(x_branch_2,*popt_2)
            x_branch_1 = func(y_branch_1,*popt_1)

            # show branches
            ax_show.plot(x_branch_1,y_branch_1,'k--')
            ax_show.plot(x_branch_2,y_branch_2,'k--')

            # show 1-1 line
            ax_show.plot(x_branch_2,x_branch_2,'k--')
        
        return ax, cb
    
    def get_labels_per_region(self):
        ### TODO, put this in settings when working on it again. Validate the regions otherwise
        
        start_time = time.time()
        # make it loop ? 
        self.wpwp_labels_in_jdist = self.get_labels_in_jdist_bins(True, slice(30, 50), slice(130, 185))
        self.ep_itcz_labels_in_jdist = self.get_labels_in_jdist_bins(True, slice(35, 50), slice(215, 280))
        self.atl_itcz_labels_in_jdist = self.get_labels_in_jdist_bins(True, slice(35, 45), slice(290, 350))
        self.epcz_labels_in_jdist = self.get_labels_in_jdist_bins(True, slice(10, 30), slice(160, 260))
        self.af_rf_labels_in_jdist = self.get_labels_in_jdist_bins(True, slice(30, 50), slice(340, 40))
        self.io_wp_labels_in_jdist = self.get_labels_in_jdist_bins(True, slice(20, 35), slice(55, 100),)
        self.se_as_labels_in_jdist = self.get_labels_in_jdist_bins(True, slice(30, 45), slice(100, 130))
        self.ct_as_labels_in_jdist = self.get_labels_in_jdist_bins(True, slice(50, 60), slice(80, 120))
        self.regions_labels_in_jdist = [self.wpwp_labels_in_jdist, self.ep_itcz_labels_in_jdist, self.atl_itcz_labels_in_jdist, self.epcz_labels_in_jdist, 
                                    self.af_rf_labels_in_jdist, self.io_wp_labels_in_jdist, self.se_as_labels_in_jdist, self.ct_as_labels_in_jdist]
        print(f"Time elapsed for loading regions labels: {time.time() - start_time:.2f} seconds")
       
    def get_labels_in_jdist_bins(self, regional = None):
        """
        Return matrix N_i,N_j,N_MCS of labels in each bin of the joint distribution
        """
        
        n_i,n_j = self.bincount.shape
        n_mcs = self.labels_regridded_yxtm.shape[3]
        labels_ij = np.full((n_i,n_j, n_mcs),np.nan)
        
        for i_bin in range(n_i):
            for j_bin in range(n_j):
                # print(i_bin,j_bin)
                labels_bin = np.unique(self.labels_in_joint_bin(i_bin,j_bin, regional))
                # number of labels
                n_labs = len(labels_bin)
                n_store = min(n_mcs,n_labs)
                # store
                labels_ij[i_bin,j_bin,:n_store] = labels_bin[:n_store]
                
        return labels_ij
    
    def storm_attributes_on_jdist(self, attr, func, region_mask=None, fast = True):
        # given an attribute that is a variable of st.ds_storm, 
        n_i, n_j = self.bincount.shape
        ## build output
        out_ij = np.full((n_i, n_j), np.nan)
        # Open storm dataset
        storms = xr.open_dataset(self.file_storms, mode='r')
        # Retrieve valid labels because of unknown labels bug
        valid_labels = storms.label.values
        # If fast remove the first 20 bins of each
        if fast : 
            n_i_start, n_j_start = 10, 10
        else : 
            n_i_start, n_j_start = 0, 0
        # Iterate over bins 

        for i_bin in range(n_i_start, n_i):
            if i_bin%1==0 : print(i_bin, end='')
            for j_bin in range(n_j_start, n_j):
                #If a peculiar region defined in settings 
                if region_mask is None : 
                    labels = self.labels_in_joint_bin(i_bin,j_bin)
                else : 
                    labels = self.labels_in_joint_bin(i_bin, j_bin, region_mask)

                # Clean label
                labels = np.unique(labels[~np.isnan(labels)])
                # Select valid ones from dataset available labels
                valid_selected_labels = [label for label in labels if label in valid_labels]
                """
                Missing labels seems to be the ones that I removed from dataset (labels between 1.9e5 to 2.1e5)
                """
                # Print the missing ones
                # unvalid_selected_labels = [label for label in labels if label not in valid_labels]
                # if len(unvalid_selected_labels)>0:
                #     print(f"At joint bin, {i_bin},{j_bin} labels that were not valid are : {unvalid_selected_labels}")
                # Retrieve attr values from dataset
                attr_values = storms.sel(label = valid_selected_labels)[attr].values

                if len(attr_values) > 0:
                    try:
                        out_ij[i_bin,j_bin] = func(attr_values)
                    except ValueError:
                        print(len(labels))
                        print(attr_values)
                        print("Oops!  That was no valid number.  Try again...")

        return out_ij
    
    def process_plot_var_cond_reducing_prec(self, var_id, var_cond_list, mask = True, func = "mean"):
        key = func+'_'+var_id

        if func == 'MCS':
            da_var = self.grid.get_var_id_ds("MCS_label").sortby("days")[var_id]
        # Trying to avoid the prec bug, maybe it's due to prec dataset already being open within jd
        elif var_id == "Prec" : 
            da_var = self.prec.sortby("days")[key]
        else :  
            da_var = self.grid.get_var_id_ds(var_id).sortby("days")[key]
            
        var_days = list(da_var.days.values)
        da_var = da_var.where(mask)
        var = da_var.values.ravel()
        
        reduced_prec = self.prec.sel(days = var_days).where(mask)
        bincount_where_var_cond = []
        labels = []
        for cond_inf, cond_sup in zip(var_cond_list[:-1], var_cond_list[1:]):
            spatial_var_where_cond = list(np.where((da_var.values>cond_inf) & (da_var.values<=cond_sup)))
            # print([(spatial_var_where_cond[i]) for i in range(3)])
            sample1_where_cond = reduced_prec[self.var_id_1].values[spatial_var_where_cond[0], spatial_var_where_cond[1], spatial_var_where_cond[2]] #this flattens
            sample2_where_cond = reduced_prec[self.var_id_2].values[spatial_var_where_cond[0], spatial_var_where_cond[1], spatial_var_where_cond[2]] #this flattens
            bincount_cond, _, _ = np.histogram2d(x=sample1_where_cond, y=sample2_where_cond, bins = (self.bins1, self.bins2), density = False)
            bincount_where_var_cond.append(bincount_cond)
            labels.append(f"{cond_inf} < {key} <= {cond_sup}")
        
        nrows = ceil(len(var_cond_list)/2)#+1 if impair
        fig, axs = plt.subplots(nrows = nrows, ncols = 2, figsize = (12, 4.71*nrows))
        
        ax_hist = axs[0, 0]
        simple_hist(var, f"{key}", bars= var_cond_list, fig = fig, ax = ax_hist) #label = f"Simple hist of {var_id}"
        
        ax_jd = axs[0, 1]
        bincount_reduced_prec, _, _ = np.histogram2d(x = reduced_prec[self.var_id_1].values.flatten(), y = reduced_prec[self.var_id_2].values.flatten(), bins = (self.bins1, self.bins2), density = False)
        self.plot_data(bincount_reduced_prec, scale = 'log', label = "Reduced Prec", cmap=plt.cm.magma_r , fig = fig, ax = ax_jd)
        
        for bincount, ax, label in zip(bincount_where_var_cond, axs.flatten()[2:], labels):
            self.plot_data(bincount/bincount_reduced_prec, scale = 'linear',  cmap=plt.cm.magma_r, vbds = (0, 1), fig = fig, ax = ax, label = label)
        
        return bincount_where_var_cond, bincount_reduced_prec
     
    def compute_conditional_data_over_density(self, data = None, mask = None):         
        var_days = list(data.days.values)  
        n_i, n_j = self.bincount.shape
        
        if data is not None : 
            data_over_density = np.full(shape=(n_i,n_j), fill_value=np.nan)
            if mask is not None : 
                data = data.where(mask)
                
        for d2 in range(n_j): 
                for d1 in range(n_i):
                    data_where_joint_bin = self.get_mask_yxt(d1, d2, var_days=var_days)
                    # return data_where_joint_bin
                    # data_idx = tuple(np.argwhere((digit1==i1) & (digit2==i2)).T)
                    if np.any(data_where_joint_bin==True):
                        to_mean = data.where(data_where_joint_bin)
                        if not np.all(np.isnan(to_mean)):
                            data_over_density[d1, d2] = np.nanmean(to_mean)
        if data is not None:
            return data_over_density
     
    def plot_var_id_func_over_jdist(self, var_id, func, mask, cmap = plt.cm.viridis, vbds = (None, None), fig = None, ax = None):
        key = func+'_'+var_id
            # Trying to avoid the prec bug, maybe it's due to prec dataset already being open within jd
        if var_id == "Prec" : 
            ds_var = self.prec.sortby("days")[key]
        else :  
            ds_var = self.grid.get_var_id_ds(var_id).sortby("days")[key]
            
        var_days = list(ds_var.days.values)
        ds_var = ds_var.sel(days = var_days).where(mask) # redundant ? 
        
        var_over_density = self.compute_conditional_data_over_density(ds_var, mask = mask) #more a da than ds but whatever
        if fig is None : 
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 4.71))
        ax, cb = self.plot_data(var_over_density, data_noise = None, cmap = cmap, branch=False, vbds = vbds, fig = fig, ax = ax, label = key)
        return ax, cb
        
    def add_mcs_var_from_labels(self, var_id):
        """
        Not the same var_id than usual
        """
        with self.grid.get_var_id_ds("MCS_label").sortby("days") as ds_mcs:
            
            ds_var_by_label = xr.open_dataset(self.file_storms)[var_id] 
            arr = np.full(np.shape(ds_mcs["MCS_label"])[:-1], np.nan)
            # Messy stuff bc MCS in here
            for i_lat ,lat in enumerate(ds_mcs.lat_global):
                for i_lon, lon in enumerate(ds_mcs.lon_global):
                    for i_day, day in enumerate(ds_mcs.days):
                        sub_ds = ds_mcs.sel(days = day, lat_global = lat, lon_global = lon)
                        labels = sub_ds["MCS_label"].values
                        rel_surfaces = sub_ds["Rel_surface"].values
                        valid_labels = labels[~np.isnan(labels)].astype('int')
                        if len(valid_labels)>0:
                            labels = valid_labels[np.isin(valid_labels, ds_var_by_label.label)]
                            rel_surfaces = rel_surfaces[:len(valid_labels)][np.isin(valid_labels, ds_var_by_label.label)]
                            if len(labels)>0:
                                var_values = ds_var_by_label.loc[dict(label=labels)].values          
                                rel_surfaces = rel_surfaces[:len(labels)]
                                arr[i_lat, i_lon, i_day] = np.average(var_values, weights = rel_surfaces) 
                                
            
            da_var = xr.DataArray(arr, dims=["lat_global", "lon_global", "days"],
                                    coords={"lat_global": ds_mcs.lat_global,
                                            "lon_global": ds_mcs.lon_global,
                                            "days": ds_mcs.days})
            
            ds_mcs = xr.merge([ds_mcs, da_var.rename(var_id)])
            
            file_mcs_ds = self.grid.get_var_ds_file("MCS_label")
            os.remove(file_mcs_ds)
            ds_mcs.to_netcdf(file_mcs_ds)

        # def process_plot_var_cond(self, var_id, var_cond_list, mask = None, func = "mean"):
   
   
    #     ## TODO catch var_unit somehow for cleaner labels
    #     key = func+'_'+var_id
    #     var_ds = self.grid.get_var_id_ds(var_id)
    #     var = var_ds[key].sortby("days").values.ravel()
        
    #     ## have to make it smarter and add days_to_fill np.nan arrays before var... or reshape ! 
    #     to_fill_var = len(self.sample1) - len(var)
    #     var_padded_flat = np.pad(var, (to_fill_var, 0), mode = 'constant', constant_values = np.nan)
    #     var_padded = np.reshape(var_padded_flat, self.shape)
    #     bincount_where_var_cond = []
    #     labels = []
    #     for cond_inf, cond_sup in zip(var_cond_list[:-1], var_cond_list[1:]):
    #         spatial_var_where_cond = list(np.where((var_padded>cond_inf) & (var_padded<=cond_sup)))
    #         # print([(spatial_var_where_cond[i].dtype) for i in range(3)])
    #         sample1_where_cond = self.prec[self.var_id_1].values[spatial_var_where_cond[0], spatial_var_where_cond[1], spatial_var_where_cond[2]]
    #         sample2_where_cond = self.prec[self.var_id_2].values[spatial_var_where_cond[0], spatial_var_where_cond[1], spatial_var_where_cond[2]]
    #         bincount_cond, _, _ = np.histogram2d(x=sample1_where_cond, y=sample2_where_cond, bins = (self.bins1, self.bins2), density = False)
    #         bincount_where_var_cond.append(bincount_cond)
    #         labels.append(f"{cond_inf} < {key} <= {cond_sup}")
        
    #     nrows = ceil(len(var_cond_list)/2)#+1 if impair
    #     fig, axs = plt.subplots(nrows = nrows, ncols = 2, figsize = (12, 4.71*nrows))
        
    #     ax_hist = axs[0, 0]
    #     simple_hist(var, f"{key}", bars= var_cond_list, fig = fig, ax = ax_hist) #label = f"Simple hist of {var_id}"
        
    #     ax_jd = axs[0, 1]
    #     self.plot_data(self.norm_density, scale = 'log', label = "Vanilla Y", cmap=plt.cm.BrBG , fig =fig, ax = ax_jd, vbds = (1e-3,1e3))
        
    #     for bincount, ax, label in zip(bincount_where_var_cond, axs.flatten()[2:], labels):
    #         self.plot_data(bincount/self.bincount, scale = 'linear',  cmap=plt.cm.magma_r, vbds = (0, 1), fig = fig, ax = ax, label = label)

    #     var_ds.close()
    #     plt.show()
    #     return bincount_where_var_cond