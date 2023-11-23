import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import warnings
from scipy.optimize import OptimizeWarning

# Suppress specific warning
def piecewise_linear(t:np.array,t_breaks:list,s_breaks:list):
    """
    Define piecewise linear RH shape with constant value at top and bottom.

    Args:
        t (np.array): t coordinate
        t_breaks (list): t values of break points
        s_breaks (list): surface values of break points

    Returns:
        np.array: piecewize surface
        
    """
    
    N_breaks = len(t_breaks)
    
    cond_list = [t <= t_breaks[0]]+\
                [np.logical_and(t > t_breaks[i-1],t <= t_breaks[i]) for i in range(1,N_breaks)]+\
                [t > t_breaks[N_breaks-1]]
                
    def make_piece(k):
        def f(t):
            return s_breaks[k-1]+(s_breaks[k]-s_breaks[k-1])/(t_breaks[k]-t_breaks[k-1])*(t-t_breaks[k-1])
        return f 
    func_list = [lambda t: s_breaks[0]]+\
                [make_piece(k) for k in range(1,N_breaks)]+\
                [lambda t: s_breaks[N_breaks-1]]
                
    return np.piecewise(t,cond_list,func_list)

def piecewise_fit(t:np.array,s:np.array,t_breaks_0:list,s_breaks_0:list):    
    """
    Compute piecewise-linear fit of surf(t).

    Args:
        t (np.array): t coordinate
        s (np.array): surface over time
        t_breaks_0 (list): initial t values of break points
        s_breaks_0 (list): initial s values of break points

    Returns:
        t_breaks (list): fitted t values of break points
        s_breaks (list): fitted s values of break points
        s_id (np.array): piecewize s fit

    """

    N_breaks = len(t_breaks_0)

    def piecewise_fun(t,*p):
        return piecewise_linear(t,p[0:N_breaks],p[N_breaks:2*N_breaks])
    
    warnings.filterwarnings("ignore", category=OptimizeWarning)
    p0 = t_breaks_0 + s_breaks_0
    try:
        p, e = curve_fit(piecewise_fun, t, s, p0=p0) 
        s_id = piecewise_linear(t, p[0:N_breaks], p[N_breaks:2*N_breaks])
        s_breaks = list(p[N_breaks:2*N_breaks])
        t_breaks = list(p[0:N_breaks])
    
        return t_breaks,s_breaks,s_id
    
    except Exception as e:
        print(f"Optimization error: {e}") 
        print('Returning initial guess')
        s_id_0 = piecewise_linear(t, p0[0:N_breaks], p0[N_breaks:2*N_breaks])
        return t_breaks_0, s_breaks_0, s_id_0

def set_storm_growth_rate(storm, r_treshold = 0.85, verbose = False, plot = False):
    """
    Given a storm object, update it's growth_rate attribute 
    Returns an ax object to plot the fit
    """
    surf = np.array(storm.clusters.surfPix_172Wm2) * 16
    
    if len(surf) <= 6 : 
        storm.growth_rate = np.nan
        return None
    else : 
        time = np.arange(0, len(surf))

        surf_breaks = [0, max(surf), 0]
        time_breaks = [0, len(surf)//2, len(surf)]

        t_breaks, s_breaks, s_id = piecewise_fit(time, surf, time_breaks, surf_breaks)
        r_squared = r2_score(surf, s_id)
        growth_rate = (s_breaks[1] - s_breaks[0]) / (t_breaks[1] - t_breaks[0])
        
        if verbose : print(f"For storm with label {storm.label}, the growth rate computed by fitting a triangle is {growth_rate} with an r-score of {r_squared}")

        if r_squared > r_treshold : 
            setattr(storm, 'growth_rate', growth_rate)
            setattr(storm, 'r_score_growth_rate', growth_rate)
        else : 
            setattr(storm, 'growth_rate', np.nan)
            setattr(storm, 'r_score_growth_rate', r_squared)

        
        if plot : 
            # Return ax object if plotting is necessary
            fig, ax = plt.subplots()
            ax.scatter(time, surf, label='Surface')
            time_plot = np.linspace(0, time.max(), 1000)
            #ax.plot(time_plot, piecewise_linear(time_plot, t_breaks, s_breaks), 'r-', label='Idealized Surface')
            ax.legend()
            ax.set_xlabel('Time')
            ax.set_ylabel('Surface Values')
            ax.set_title('Fitting a Triangle Function to Surface Values over Time')

            return ax