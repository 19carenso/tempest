import argparse
import numpy as np
import warnings
import os 

from tempest import casestudy
from tempest import grid
from tempest import storm_tracker
from tempest import joint_distrib
from tempest import handler

import matplotlib.pyplot as plt


settings_path = 'settings/tropics_20d.yaml'

# Parse command-line arguments
parser = argparse.ArgumentParser(description='TEMPEST Script')
parser.add_argument('--i_bin', type=int, default=25, help='X-coordinate for location')
parser.add_argument('--j_bin', type=int, default=25, help='Y-coordinate for location')
args = parser.parse_args()

hdlr = handler.Handler(settings_path)
cs = casestudy.CaseStudy(hdlr, overwrite = False ,verbose = False)
gr = grid.Grid(cs, fast = True, overwrite= False, verbose_steps = False, verbose = False)
st = storm_tracker.StormTracker(gr, overwrite = False) #overwrite = True is super long
jd = joint_distrib.JointDistribution(gr, st, var_id_1 = "mean_unweighted_Prec", var_id_2 = "cond_alpha_50_Prec", nd=5, overwrite = True)


# Filter out RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

arr_yxt = np.argwhere(jd.get_mask_yxt(args.i_bin, args.j_bin)) 

# Once this is definitive, put it in a class, add more arguments to tune the plot
def hovmoller_prec_contour(self, var_id, i_x, i_y, i_day):
    #get coord from get_mask_yxt output
    fig, ax = plt.subplots(1,1)
    day = list(self.grid.casestudy.days_i_t_per_var_id["Prec"].keys())[i_day]
    lat = i_x-30
    lon = i_y
    
    treshold_prec = self.prec.Treshold_cond_alpha_50_Prec[i_x, i_y, i_day].values
    
    first_i_t = cs.days_i_t_per_var_id["Prec"][day][0]
    first_prec = hdlr.load_var(gr, "Prec", first_i_t).sel(lat=slice(lat, lat+1), lon = slice(lon, lon+1)).values
    
    n_lat = first_prec.shape[0]
    n_i_t = len(self.grid.casestudy.days_i_t_per_var_id["Prec"][day])
    hov_prec = np.full((n_lat,n_i_t), np.nan)
    hov_var = np.full((n_lat,n_i_t), np.nan)
    for i, i_t in enumerate(self.grid.casestudy.days_i_t_per_var_id["Prec"][day]):
        print(i_t)
        prec = hdlr.load_var(gr, "Prec", i_t).sel(lat=slice(lat, lat+1), lon = slice(lon, lon+1)).values # shape is (lat, lon)
        prec[prec<treshold_prec]=np.nan
        hov_prec[:,i] = np.nanmean(prec, axis = 1)
        if i_t in cs.days_i_t_per_var_id[var_id][day]:
            var = hdlr.load_var(gr, var_id, i_t).sel(lat=slice(lat, lat+1), lon = slice(lon, lon+1)).values
            hov_var[:,i] = np.nanmean(var, axis = 1)
        else : 
            hov_var[:,i] = np.full((n_lat,), np.nan)
            
    contour = ax.contour(hov_prec.T, origin = 'lower', cmap= 'Blues', alpha = 0.75)
    pcolor = ax.pcolor(hov_var.T, cmap = 'RdGy')
    colorbar = fig.colorbar(pcolor, ax=ax)
    
    fig_name = f"hovmaller_prec_contour_of_{var_id}_at_latxlon_{lat}x{lon}_on_day_{day}.png"
    ax.set_title(fig_name)
    ax.set_xlabel("native lat")
    ax.set_ylabel("timestep within day")
    colorbar.set_label(f"{var_id}")
    abs_dir_fig = os.path.join(os.getcwd(), jd.settings["DIR_FIG"])
    if not os.path.exists(abs_dir_fig):
        os.makedirs(abs_dir_fig)
    fig_dir = os.path.join(abs_dir_fig, self.name)
    fig_path = os.path.join(fig_dir, fig_name)
    plt.savefig(fig_path)

for i,yxt in enumerate(arr_yxt):
    print(f"{100*i/len(arr_yxt)}%")
    ## Should check here that the day given by yxt[2] is actually available in the ditvi of the considered var_id
    var_id = "OM850_cond_Prec"
    day = list(cs.days_i_t_per_var_id["Prec"].keys())[yxt[2]]
    if day in list(cs.days_i_t_per_var_id[var_id].keys()):
        hovmoller_prec_contour(jd, var_id, yxt[0], yxt[1], yxt[2])
    else : 
        print(f"Too bad this joint extreme day {day} doesn't have value for the variable {var_id}")
        
print("100%")