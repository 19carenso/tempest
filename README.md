# TEMPEST

Tallying Engine for Multiscale Precipitation Events with Storm Tracking. 


### Prerequisites

- Python 3.8

### Setting Up a Virtual Environment (Recommended)

### Update tempest/utils.py to fetch correctly the data specified in settings.yaml
Typically contains functions that are able to correctly read your data, typically the output from tracking.

### Update settings/your_settings
Adapt to your paths

### modify handler and casestudy to your paths
This class is full of functions to tackle different scenario. I'd reccomand creating your own function, or smartly adapting those in place
Handler manages paths and slicing of netcdf on region of interest (in some cases)
casestudy builds the dictionnary that gives the variable available timesteps.

### Build tempest with pip install -e . 
will allow the imports at the beginning of each notebooks. and the main.py and wrapped_main.py

### Then launch scripts/main.py for each model/region/period
This will populate the according data/region
region is equivalent to the coordinates of interest, the timeline, and the model used (as well as the tracking)
it will populate the directory data that can be used for making figures afterward.

It will also retrieve the variables names from settings DIR_DATA_IN and fetch their corresponding days and timestamps available 
that will be store under DIR_DATA_OUT/var_id_days_i_t.json

### Authors

Maxime Carenso & Benjamin Fildier

### License

LMD/IPSL, ENS, PSL Research University, Ecole Polytechnique, Institut Polytechnique de Paris, ´
7 Sorbonne Universit´e, CNRS, Paris France 
