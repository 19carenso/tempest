import numpy as np
import os
import time
import sys


from math import log10
from skimage import measure
from scipy.optimize import curve_fit
import warnings

from .distribution import Distribution

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from .plots.plot2d import set_frame_invlog, show_joint_histogram

class JointDistribution():
    """
    Creates a joint distribution for two precipitations variables based on Grid prec.nc
    """
    
    def __init__(self, grid, nd=4, overwrite=False):
        """Constructor for class Distribution.
        Arguments:
        - name: name of reference variable
        - distribution1, distribution2: marginal distributions of two reference variables
        - overwrite: option to overwrite stored data in object
        """
        
        # Inheritance by hand because __super__() speaks too much 
        self.grid = grid

        self.name = grid.name
        self.settings = grid.settings
        
        self.nd = nd

        self.ditvi = grid.days_i_t_per_var_id
        
        self.prec = grid.get_var_id_ds("Prec")
        
        self.shape = np.shape(self.prec["mean_Prec"].to_numpy())
        
        self.sample1 = self.prec["mean_Prec"].to_numpy().ravel()
        self.sample2 = self.prec["max_Prec"].to_numpy().ravel()

        self.dist1 = Distribution(name = self.name + '_' + 'mean_Prec',  bintype = "invlogQ", nd = self.nd, fill_last_decade=True)
        self.dist1.compute_distribution(self.sample1)

        self.dist2 = Distribution(name = self.name + '_' + 'max_Prec',  bintype = "invlogQ", nd = self.nd, fill_last_decade=True)
        self.dist2.compute_distribution(self.sample2)

        self.bins1 = self.dist1.bins
        self.bins2 = self.dist2.bins

        self.density = None
        self.bin_locations_stored = False
        self.overwrite = overwrite

        self.compute_distribution(self.sample1, self.sample2)
        self.compute_normalized_density(self.sample1, self.sample2)
        
        self.digit_3d_1, self.digit_3d_2 = self.compute_conditional_locations()      
    
    def __repr__(self):
        """Creates a printable version of the Distribution object. Only prints the 
        attribute value when its string fits is small enough."""

        out = '< JointDistribution object:\n'
        # print keys
        for k in self.__dict__.keys():
            out = out+' . %s: '%k
            if sys.getsizeof(getattr(self,k).__str__()) < 80:
                # show value
                out = out+'%s\n'%str(getattr(self,k))
            else:
                # show type
                out = out+'%s\n'%getattr(self,k).__class__
        out = out+' >'

        return out
    
    def __str__(self):
        """Override string function to print attributes
        """
        # method_names = []
        # str_out = '-- Attributes --'
        str_out = ''
        for a in dir(self):
            if '__' not in a:
                a_str = str(getattr(self,a))
                if 'method' not in a_str:
                    str_out = str_out+("%s : %s\n"%(a,a_str))
        #         else:
        #             method_names.append(a)
        # print('-- Methods --')
        # for m in method_names:
        #     print(m)
        return str_out
    
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
        if verbose : print(digit1, digit2)
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
        # ax.set_title(title)

        # -- Density
        pcm = show_joint_histogram(ax_show, Z, scale=scale, vmin=vbds[0], vmax=vbds[1], cmap=cmap)
        cont = measure.find_contours(Z, 1)

        # -- Masks multiscale categories
        ax_show.imshow(self.mask_show.T,alpha=0.5,origin='lower')

        if branch : 
            # -- Branches
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
            
    def get_mask_yxt(self, d1, d2):
        dj = self.joint_digit(d1, d2)
        dj_3d = self.joint_digit(self.digit_3d_1, self.digit_3d_2)
        
        return dj_3d == dj
        
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
        dlon = np.diff((self.grid.lon_slice.start, self.grid.lon_slice.stop))[0] % 360
        if dlon == 0: dlon = 360
        dlat = np.diff((self.grid.lat_slice.start, self.grid.lat_slice.stop))[0]
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

        ax.coastlines('110m')
        ax.gridlines()
        ax.text(0.01,0.05,"N = %d"%Next,transform=ax.transAxes)
        
        # Colorbar
        x,y,w,h = ax.get_position().bounds
        dx = w/60
        cax = plt.axes([x+w+1.5*dx,y,dx,h])
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.set_ylabel('Bincount (#)')

        return ax
        
        
    