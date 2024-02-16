# TEMPEST

Tallying Engine for Multiscale Precipitation Events with Storm Tracking. 


### Prerequisites

- Python 3.8

### Setting Up a Virtual Environment (Recommended)

pip install -r requirements.txt

### setup the environment by configuring settings/settings.yaml
Recreate a .yaml for your own configuration. 
Here is a guide on how to do it : 

### Update tempest/utils.py to fetch correctly the data specified in settings.yaml
Typically contains functions that are able to correctly read your data. 

### Then launch setup.py for each region (as a job)
This will populate the according data/region
region is equivalent to the coordinates of interest, the timeline, and the model used (as well as the tracking)
it will populate the directory data, accordingly storing 
the first tempest.nc 

It will also retrieve the variables names from settings DIR_DATA_IN and fetch their corresponding days and timestamps available 
that will be store under DIR_DATA_OUT/var_id_days_i_t.json


!!! 
    Currently there is a bug, so that you need to relaunch setup.py multiple time 
    unless you sort your in new_variablesin json file in order of creation 
    or until you have propagated your new_variables to the saved .json files 
    to the point that every new_variables depends only on variables precendently saved by CaseStudy instance 
!!!

### Launch main script (as a job) 
Use command-line arguments to specify the required inputs.
Should create pickle files in data/region/key/by_day.pkl
Will also update tempest.nc


### Do the /test/intro.ipynb

You can extract the appropriate sub_data from DYAMOND with the script tests/intro/extract_data.py. 
From there you can even choose the time index and box of interest and variables. 
Then you have to move them from cache to the place you want to store them (already in input/intro/data normally)
If then intro.yamml is set-up correctly you should be able to play the notebook test/intro/intro.ipynb

### Other Important Notes
PYTHONPATH: If you prefer using absolute imports, you can add the project's 
root directory to your PYTHONPATH environment variable:
in bash : 
export PYTHONPATH=/path/to/your_project:$PYTHONPATH

This is really reccommanded.

### When adding a new storm class 

You must modify Handler the add_storm_tracking_variables in casestudy
Grid must also be modified as it catches storm functions based on string names 

### Authors

Maxime Carenso & Benjamin Fildier

### License

CNRS ? IPSL ? LEGOS ? 