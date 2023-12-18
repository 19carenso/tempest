import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation
from IPython.display import HTML

from tempest import casestudy
from tempest import grid
from tempest import joint_distrib
from tempest import handler
from tempest import storm_tracker

settings_path = 'settings/tropics.yaml'

hdlr = handler.Handler(settings_path)
cs = casestudy.CaseStudy(hdlr, verbose = False)
gr = grid.Grid(cs, verbose = False, overwrite = False)
if __name__ == '__main__':
    st = storm_tracker.StormTracker(gr, overwrite = False) #overwrite = True is super long
    # st_feng = storm_tracker.StormTracker(gr, label_var_id = "MCS_label_Tb_Feng", overwrite = False)
    pass


jd = joint_distrib.JointDistribution(gr, st, nd=5)
# jd_feng = joint_distrib.JointDistribution(gr, st_feng, nd=5)

def _load_plot_var(var_id, gr, i_t):

    if var_id == "MCS_label" or var_id == "MCS_label_Tb_Feng":
        var= hdlr.load_var(gr, var_id, i_t).sel(longitude=slice(lonmin, lonmax), latitude=slice(latmin, latmax))[0, :, :]
        n_lat, n_lon = var.shape
        n = min(var.shape)
    else : 
        var = hdlr.load_var(gr, var_id, i_t).sel(lon=slice(lonmin, lonmax), lat=slice(latmin, latmax))[:, :]
        n_lat, n_lon = var.shape
        n = min(var.shape)
         # approximate square
    var = var[n_lat//2 - n//2 : n_lat//2 + n//2 , n_lon//2 - n//2 : n_lon//2 + n//2] # exact square
    return var

verbose = False

# i_label = 20000
label = 124402
i_label = jd.dict_i_storms_by_label[label]
storm = jd.storms[i_label]

lifecycle = storm.clusters
utc_0, utc_f = storm.Utime_Init, storm.Utime_End
i_0, i_f = int(utc_0/30), int(utc_f/30) # roots
print("storm label", storm.label,  "from", i_0, "to" ,i_f)

### Make it square

length = max(storm.latmin%90 - storm.latmax%90, storm.lonmax - storm.lonmin)
bordersize = 0.5

if np.isclose(storm.latmax - storm.latmin, length):
    latmax, latmin = storm.latmax + bordersize, storm.latmin - bordersize
    lonmid = (storm.lonmin + storm.lonmax) / 2
    lonmax, lonmin = lonmid + bordersize + length / 2, lonmid - bordersize - length / 2
else : 
    latmid = (storm.lonmin + storm.lonmax) / 2
    latmax, latmin = storm.latmax + bordersize + length / 2, storm.latmin - bordersize - length / 2
    lonmax, lonmin = storm.lonmax + bordersize, storm.lonmin - bordersize 
    
print(length)
print("longitude min and max", lonmin, lonmax)
print("latitude min and max", latmin, latmax)
    
var_id = "Prec"
var_unit = "mm/h"
var_cmap = "Blues"
    
for i_t in range(i_0, i_f):

    var = _load_plot_var(var_id, gr, i_t)
    olr = _load_plot_var("LWNTA", gr, i_t)
    seg = _load_plot_var("MCS_label", gr, i_t)
    # # winds
    # u, v = hdlr.load_var(gr, "U10m", i_t), hdlr.load_var(gr, "V10m", i_t)
        
    cloud_mask = (seg.astype(int) == storm.label)
    # Set alpha values for clouds based on the mask
    alpha_values = np.where(cloud_mask, 0.7, 0.2)

    fig, ax = plt.subplots(subplot_kw=dict(projection = ccrs.PlateCarree(central_longitude = lonmax + lonmin / 2)), figsize=(10, 8))
    lon, lat = seg.longitude.values, seg.latitude.values

    # Define contour levels
    contour_levels = np.arange(130, 180, 5) #225K to ?K

    # Plot contours with specified levels
    contour_plot = ax.contour(lon, lat, olr, transform=ccrs.PlateCarree(), levels=contour_levels, colors='k', alpha=0.1)

    # Add contour labels
    contour_labels = plt.clabel(contour_plot, inline=True, fontsize=8, fmt='%1.0f')
    # plot precip (p)
    p = ax.pcolormesh(lon, lat, var, cmap=var_cmap, transform=ccrs.PlateCarree())
    # plot clouds (c)
    c = ax.pcolormesh(lon, lat, seg,  transform=ccrs.PlateCarree(), cmap="viridis", alpha=alpha_values) 
    # plot winds (w)
    # w = ax.quiver(lon, lat, u, v, transform=ccrs.PlateCarree(), regrid_shape = 50)
    ax.coastlines()

    # Adding gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    loc_interval = 1
    gl.xlocator = mticker.FixedLocator(range(-180, 181, loc_interval))
    gl.ylocator = mticker.FixedLocator(range(-90, 91, loc_interval))
    gl.top_labels = False  # Turn off labels on the top x-axis
    gl.right_labels = False  # Turn off labels on the right y-axis
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Create colorbar in a separate axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    label_name = f'{var_id} ({var_unit})'  # Use actual variable name
    cbar = plt.colorbar(p, cax=cax, orientation='vertical', label=label_name)


    # Adjust layout for the colorbars
    fig.suptitle(f"Storm {label} : {i_t}", fontsize=16)

    plt.tight_layout() 
    # saving frames and making dirs in cache
    cache = os.path.join(cs.settings["DIR_TEMPDATA"], cs.name)

    if not os.path.exists(cache):
        os.makedirs(cache)
        print("Could be made at class level")

    figname = f"storm_{label}_{var_id}" #maybe show a better date than it here but will do
    dir_path = os.path.join(cache, figname)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"made {dir_path}, don't forget to delete")

    stamp = f"{i_t}".zfill(5)
    path_stamp = os.path.join(dir_path, stamp)
    plt.savefig(path_stamp)
    print(f"Timestamp {i_t} saved")
    plt.close(fig)
    



# Output video file name
output_file = f"/home/mcarenso/code/tempest/figures/{label}.mp4"

# List all PNG files in the input directory
image_files = [file for file in os.listdir(dir_path) if file.endswith('.png')]
image_files.sort()  # Ensure sorted order

# Create a list to store image paths
image_paths = [os.path.join(dir_path, file) for file in image_files]

# Function to update the figure for each frame
def update_figure(i):
    img = plt.imread(image_paths[i])
    im.set_array(img)
    return [im]

# Create the video using Matplotlib's animation module
fig, ax = plt.subplots()
im = ax.imshow(plt.imread(image_paths[0]))

# You can customize other animation parameters such as interval, repeat, etc.
animation = animation.FuncAnimation(fig, update_figure, frames=len(image_paths), interval=500, blit=True)

animation.save(output_file, writer='ffmpeg', fps=1, dpi=300)

# Display the animation in the notebook
HTML(animation.to_jshtml())
plt.close(fig)

for path in image_paths:
    os.remove(path)