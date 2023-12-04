import numpy as np
import os
import time
import sys
import glob
import csv
import pickle

from math import log10
from skimage import measure # pylance: disable=import-error 
from scipy.optimize import curve_fit
import warnings

from .distribution import Distribution
from .load_toocan import load_toocan

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from .plots.plot2d import set_frame_invlog, show_joint_histogram

class JointDistribution():
    """
    Creates a joint distribution for two precipitations variables based on Grid prec.nc
    """
    
    def __init__(self, grid, nd=4, storm_tracking = True, overwrite=False, verbose = False):
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
        
        self.nd = nd

        self.ditvi = grid.casestudy.days_i_t_per_var_id
        
        self.prec = grid.get_var_id_ds("Prec")
        
        self.shape = np.shape(self.prec["mean_Prec"].to_numpy())
        
        self.sample1 = self.prec["mean_Prec"].to_numpy().ravel()
        self.sample2 = self.prec["max_Prec"].to_numpy().ravel()

        self.overwrite = overwrite

        cwd = os.getcwd()
        rel_dir_out = self.settings["DIR_OUT"]
        abs_dir_out = os.path.join(cwd, self.settings["DIR_OUT"])
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
        
        if storm_tracking : 
            # Should check if regridded MCS labels is already stored in grid
            self.labels_regridded_yxtm = grid.get_var_id_ds("MCS_label")["MCS_label"].values
            self.mask_labels_regridded_yxt = np.any(~np.isnan(self.labels_regridded_yxtm),axis=3)

            # get storm tracking data
            if self.verbose : print("Loading storms...")
            self.storms, self.label_storms, self.dict_i_storms_by_label = self.load_storms_tracking()
            
            ## TODO : you know what to do #settings #cleanthatmess
            if self.verbose : print("Retrieve labels in jdist")
            self.labels_in_jdist = self.get_labels_in_jdist_bins(self)
            self.wpwp_labels_in_jdist = self.get_labels_in_jdist_bins(self, True, slice(30, 50), slice(130, 185))
            self.ep_itcz_labels_in_jdist = self.get_labels_in_jdist_bins(self, True, slice(35, 50), slice(215, 280))
            self.atl_itcz_labels_in_jdist = self.get_labels_in_jdist_bins(self, True, slice(35, 45), slice(290, 350))
            self.epcz_labels_in_jdist = self.get_labels_in_jdist_bins(self, True, slice(10, 30), slice(160, 260))
            self.af_rf_labels_in_jdist = self.get_labels_in_jdist_bins(self, True, slice(30, 50), slice(340, 40))
            self.io_wp_labels_in_jdist = self.get_labels_in_jdist_bins(self, True, slice(20, 35), slice(55, 100),)
            self.se_as_labels_in_jdist = self.get_labels_in_jdist_bins(self, True, slice(30, 45), slice(100, 130))
            self.ct_as_labels_in_jdist = self.get_labels_in_jdist_bins(self, True, slice(50, 60), slice(80, 120))
            self.regions_labels_in_jdist = [self.wpwp_labels_in_jdist, self.ep_itcz_labels_in_jdist, self.atl_itcz_labels_in_jdist, self.epcz_labels_in_jdist, 
                                           self.af_rf_labels_in_jdist, self.io_wp_labels_in_jdist, self.se_as_labels_in_jdist, self.ct_as_labels_in_jdist]

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
        name_dist1 = self.name + '_' + 'mean_Prec'
        path_dist1 = os.path.join(self.dir_out, name_dist1)
        name_dist2 = self.name + '_' + 'max_Prec'
        path_dist2 = os.path.join(self.dir_out, name_dist2)

        if self.overwrite:
            self.dist1 = Distribution(name = name_dist1,  bintype = "invlogQ", nd = self.nd, fill_last_decade=True)
            self.dist1.compute_distribution(self.sample1)
            with open(path_dist1, 'wb') as file:
                pickle.dump(self.dist1, file)

            self.dist2 = Distribution(name = name_dist2,  bintype = "invlogQ", nd = self.nd, fill_last_decade=True)
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

        if verbose:
            print()

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
        
        if sample1.shape == sample2.shape:
            
            if method == 'default':

                # Compute bin count
                self.bincount, _, _ = np.histogram2d(x=sample1.flatten(),y=sample2.flatten(),bins=(self.bins1,self.bins2),density=False)

                # Compute probability density
                self.density, _, _ = np.histogram2d(x=sample1.flatten(),y=sample2.flatten(),bins=(self.bins1,self.bins2),density=True)

            elif method == 'manual':

                digit1 = np.digitize(sample1, self.bins1, right = True)
                digit2 = np.digitize(sample2, self.bins2, right = True)
                # if verbose : print(digit1, digit2)
                Ntot = sample1.size
            
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
    
    def compute_normalized_density(self, sample1, sample2, verbose = False, method='default'):
        """Compute joint density normalized by the expected density of independent variables : N_ij * N_tot / N_i / N_j."""

        if self.bincount is None:
            
            self.compute_joint_density(sample1, sample2, method = method)
        
        l1, l2 = len(self.bins1)-1, len(self.bins2)-1 # BF: adjusted nbins to match np.histogram2d

        digit1 = np.digitize(sample1, self.bins1, right = True)
        digit2 = np.digitize(sample2, self.bins2, right = True)

        N1 = [np.sum(digit1==i1) for i1 in range(l1)]
        N2 = [np.sum(digit2==i2) for i2 in range(l2)]
        Ntot = sample1.size
        with np.errstate(divide='ignore'):
            Norm = Ntot / np.outer(N1, N2)
        Norm[np.isinf(Norm)] = 1

        self.norm_density = Norm * self.bincount
        
    def compute_conditional_locations(self):
            
        digit1 = np.digitize(self.sample1, self.bins1, right = True)
        digit2 = np.digitize(self.sample2, self.bins2, right = True)
        
        digit1_3D = np.reshape(digit1,self.shape)
        digit2_3D = np.reshape(digit2,self.shape)
        
        return digit1_3D,digit2_3D
        
    def compute_conditional_data_over_density(self, sample1, sample2, data = None):
        """
        TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO 
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
                        data_over_density[i1, i2] = np.nanmean(data.flatten()[data_idx])
                    else : data_over_density[i1, i2] = 0

        if data is not None:
            
            return data_over_density
        
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

    def plot(self, branch = False):
        self.make_mask()

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

        # -- Masks multiscale categories
        ax_show.imshow(self.mask_show.T,alpha=0.5,origin='lower')

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
            
    def get_mask_yxt(self, d1, d2, regional = False, lat_slice = None, lon_slice = None):
        dj = self.joint_digit(d1, d2)
        dj_3d = self.joint_digit(self.digit_3d_1, self.digit_3d_2)
        mask = dj_3d == dj
        if regional:
            region_mask = np.zeros_like(mask, dtype = bool)
            if lat_slice is not None and lon_slice is not None:
                if lon_slice.start < lon_slice.stop:
                    region_mask[lat_slice, lon_slice] = mask[lat_slice, lon_slice] 
                elif lon_slice.stop < lon_slice.start:
                    region_mask[lat_slice, slice(0, lon_slice.stop)]= mask[lat_slice, slice(0, lon_slice.stop)]
                    region_mask[lat_slice, slice(lon_slice.start, 360)]= mask[lat_slice, slice(lon_slice.start, 360)]

            mask = region_mask
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

    def make_map(self, mask_yxt):
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
        Z = np.sum(mask_yxt,axis=-1) # count
        Next = np.sum(Z)
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
    
    def labels_in_joint_bin(self, i_bin, j_bin, regional=False, lat_slice=None, lon_slice=None):
        """
        For each joint extreme in bin (i,j), merge MCS labels occurring in their spatiotemporal occurrence.
        LATER: also store their relative area.
        """
        mask_yxt = self.get_mask_yxt(i_bin,j_bin, regional, lat_slice, lon_slice)
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
    
    def get_mcs_bin_fraction(self):
        n_i, n_j = self.bincount.shape
        bin_fraction_mcs = np.full((n_i,n_j),np.nan)
        bin_noise = np.full((n_i,n_j),np.nan)

        for i_bin in range(n_i):
            for j_bin in range(n_j):
        
                # where bin falls in x-y-t
                mask_bin_yxt = self.get_mask_yxt(i_bin,j_bin)
            
                # where bin falls in x-y-t and MCS occurs
                mask_bin_with_mcs_yxt = np.logical_and(mask_bin_yxt,self.mask_labels_regridded_yxt)
                
                # number of points in joint mask
                count_bin_mcs = np.sum(mask_bin_with_mcs_yxt)
                
                # number of point in bin mask
                count_bin = np.sum(mask_bin_yxt)
                
                # store this fraction
                if count_bin >= 4:
                    bin_fraction_mcs[i_bin,j_bin] = count_bin_mcs/count_bin
                elif count_bin > 0:
                    bin_noise[i_bin,j_bin] = 1
                # include noisy points in bin_fraction_mcs
                # if count_bin > 0:
                #     bin_fraction_mcs[i_bin,j_bin] = count_bin_mcs/count_bin
                #     if count_bin <= 4:
                #         bin_noise[i_bin,j_bin] = 1
        
        # return this fraction
        return bin_fraction_mcs, bin_noise

    def plot_data(self, data, data_noise = None, cmap = plt.cm.RdBu_r, branch = False, label = '', fig =None ,ax = None, vbds = (None, None)):
        self.make_mask()
        if fig==None and ax==None: 
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4.85))
        
        Z_nd = self.norm_density.T
        Z = data.T
        Z_noise = data_noise.T

        # Should be passed as **kwargs
        title = f"Data over Normalized density"
        scale = 'linear'
        cmap = cmap

        # -- Frame
        ax_show = ax.twinx().twiny()
        ax = set_frame_invlog(ax, self.dist1.ranks, self.dist2.ranks)
        ax.set_xlabel(r"1$^\circ\times 1$day extremes")
        ax.set_ylabel(r"4km-30mn extremes")
        # ax.set_title(title)

        # -- Density
        pcm = show_joint_histogram(ax_show, Z, scale=scale, vmin=vbds[0], vmax=vbds[1], cmap=cmap)
        if data_noise is not None :
            show_joint_histogram(ax_show, Z_noise, scale=scale, vmin=vbds[0], vmax=vbds[1], cmap=cmap, alpha=0.1)

        # -- Colorbar
        cb = fig.colorbar(pcm, ax=ax_show)
        # cb.set_label('Normalized density')
        cb.set_label(label)
        
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
        
        return ax

    def load_storms_tracking(self):
        paths = glob.glob(os.path.join(self.settings['DIR_STORM_TRACKING'], '*.gz'))
        storms = load_toocan(paths[0])+load_toocan(paths[1])
        label_storms = [storms[i].label for i in range(len(storms))]
        dict_i_storms_by_label = {}
        for i, storm in enumerate(storms):
                if storm.label not in dict_i_storms_by_label.keys():
                    dict_i_storms_by_label[storm.label] = i
        return storms, label_storms, dict_i_storms_by_label
    
    def get_labels_in_jdist_bins(self, jdist, regional = False, lat_slice=None, lon_slice=None):
        """
        Return matrix N_i,N_j,N_MCS of labels in each bin of the joint distribution
        """
        
        n_i,n_j = jdist.bincount.shape
        n_mcs = self.labels_regridded_yxtm.shape[3]
        labels_ij = np.full((n_i,n_j, n_mcs),np.nan)
        
        for i_bin in range(n_i):
            for j_bin in range(n_j):
                
                # print(i_bin,j_bin)
                labels_bin = np.unique(self.labels_in_joint_bin(i_bin,j_bin, regional, lat_slice, lon_slice))
                # number of labels
                n_labs = len(labels_bin)
                n_store = min(n_mcs,n_labs)
                # store
                labels_ij[i_bin,j_bin,:n_store] = labels_bin[:n_store]
                
        return labels_ij
    
    def storm_attributes_on_jdist(self, attr, func, region_labels_in_jdist = None):
        n_i, n_j = self.bincount.shape
        out_ij = np.full((n_i, n_j), np.nan)

        for i_bin in range(1, n_i):
            if i_bin%1==0 : print(i_bin, end='')
            for j_bin in range(1, n_j):
                if region_labels_in_jdist is None : 
                    labels = self.labels_in_jdist[i_bin,j_bin]
                else : 
                    labels = region_labels_in_jdist[i_bin, j_bin]
                labels = np.unique(labels[~np.isnan(labels)])

                i_labels = []
                for label in labels:
                    i_labels.append(self.i_storms[label])

                if attr in self.storms[0].__dict__.keys():
                    attr_list = [getattr(self.storms[i],attr) for i in i_labels]
                elif attr in self.storms[0].clusters.__dict__.keys():
                    attr_list = [np.mean(getattr(self.storms[i].clusters,attr)) for i in i_labels]

                if len(attr_list) > 0:
                    try:
                        out_ij[i_bin,j_bin] = getattr(np,'nan%s'%func)(attr_list)
                    except ValueError:
                        print(len(labels))
                        print(attr_list)
                        print("Oops!  That was no valid number.  Try again...")

        return out_ij