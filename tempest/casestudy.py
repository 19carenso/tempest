import pickle
import os
import sys
import glob
import tarfile
import shutil 
import json 

import numpy as np
import datetime as dt
import re

from tempest.utils import load_rel_table, extract_digit_after_sign

from functools import reduce

class CaseStudy():
    
    """
    Documentation for class CaseStudy
    """
        
    def __init__(self, settings, verbose = False, overwrite = False):
        
        self.settings = settings

        self.verbose = verbose
        
        self.overwrite = overwrite
        
        self.region = settings['REGION']
        self.model = settings['MODEL']
        
        self.name = '%s_%s' % (self.model, self.region)

        self.data_in = settings['DIR_DATA_IN']
        
        self.rel_table = load_rel_table(settings['REL_TABLE'])
        
        self.data_out = os.path.join(settings['DIR_DATA_OUT'], self.name)
        
        self._set_region_coord_and_period()
        
        #Maybe make a more robust intern function out of that 
        if not os.path.exists(self.data_out):
            os.makedirs(self.data_out)
            print(f"First instance of {self.name}. It's directory has been created at : {self.data_out}")
            ## TODO create the nc
        else:
            pass # nothing to say really , am I wrong? The thing is that because of grid.__super__() inheritance we reprint it it's boring... How to override that.
            # print(f"Directory already exists: {self.data_out} \n")
            ## load it. to it cleanly so read netcdf first 
        
        
        ### Here we load the variables from the dir freshly createds
        json_filename = 'var_id_days_i_t.json'
        json_path = os.path.join(self.data_out, json_filename)
        if not os.path.exists(json_path):
            print("Json file of variables, days, i_t not found, so loading them...")
            self.variables_names, self.days_i_t_per_var_id = self._load_var_id_in_data_in()
            print(f"Variables data retrieved. Saving them in {json_path}")
            self.save_var_id_as_json(self.variables_names, self.days_i_t_per_var_id, json_path)
        else :
            # print('Found json file at {json_path}, loading it..')
            self.variables_names, self.days_i_t_per_var_id = self.load_var_id_from_json(json_path) #dates are in "year-date-month" for now
            
        if verbose : self._chek_variables_days_and_i_t()

        json_filename = 'new_variables.json'
        new_var_json_path = os.path.join(self.data_out, json_filename)
        if not os.path.exists(new_var_json_path):
            print(f"Json file of new variables not found, so please create it at path {self.data_out} in same fashion than specified in README.md !!!" )
        elif os.path.exists(new_var_json_path):
            print("Loading the new variables and their corresponding function for loading them")
            self.new_variables_names, self.new_var_dependencies, self.new_var_functions = self.add_new_var_id(new_var_json_path)
            self.save_var_id_as_json(self.variables_names, self.days_i_t_per_var_id, json_path)

    def __repr__(self):
        """Creates a printable version of the Distribution object. Only prints the 
        attribute value when its string fits is small enough."""

        out = '< Tempest instance:\n'
        # print keys
        for k in self.__dict__.keys():
            out = out+' . %s: '%k
            if k in ['dist_chunks','chunks_to_ignore']:
                # show type
                out = out+'%s\n'%str(getattr(self,k).__class__)
            else:
                if len(str(getattr(self,k))) < 80:
                    # show value
                    out = out+'%s\n'%str(getattr(self,k))
                else:
                    # show type
                    out = out+'%s\n'%getattr(self,k).__class__
        out = out+' >'

        return out

    def _set_region_coord_and_period(self):
        """
        Simply load the Region of Analysis sections from settings.yaml into class attributes
        """
        # could actually be updated with dtvi 
        self.i_t_min = self.settings['TIME_RANGE'][0]
        self.i_t_max = self.settings['TIME_RANGE'][1]
        self.range_t = range(self.i_t_min, self.i_t_max+1)
        # BOX is [lat_min, lat_max, lon_min, lon_max]
        self.lat_slice = slice(self.settings['BOX'][0], self.settings['BOX'][1])
        self.lon_slice = slice(self.settings['BOX'][2], self.settings['BOX'][3])

    def _load_var_id_in_data_in(self):
        """
        this functions loads the data from your DIR_DATA_IN settings

        :param dir: The path to your DIR_DATA_IN in .yaml

        :return
            var_id: list of variables found
            days_i_t_per_var_id: a dictionnary that contains the days and correspong indexes per var_id    
                            days_i_t_per_var_id[var_id] = tuple(list of days, dict of i_t per key the full datetime in format datetime.datetime(2016, 8, 31, 0, 0))
        """
        dir = self.settings['DIR_DATA_IN']
        variables_id = []
        days_i_t_per_var_id = {}

        def get_day_and_i_t(filename):
                # Define a regex pattern to extract the timestamp from your file path
                timestamp_pattern = r'(\d{10})\.\w+\.2D\.nc'

                # starting date
                date_ref = dt.datetime(year=2016, month=8, day=1)
                
                def get_datetime_and_i_t(filename):
                    # Extract the timestamp from the file path
                    match = re.search(timestamp_pattern, filename)
                    if match:
                        timestamp = int(match.group(1))
                        # Calculate the delta in seconds
                        delta_t = dt.timedelta(seconds=timestamp * 7.5)
                        # Calculate the current date
                        date_current = date_ref + delta_t

                        i_t = int(timestamp / 240) 
                        return date_current, i_t
                    else:
                        return None  # Handle cases where the timestamp couldn't be extracted

                # time dimension
                
                date_time, i_t = get_datetime_and_i_t(filename)
                days_i_t = {} ## maybe here work on storing the full datetimes with corresponding i_t. 
                day = date_time.strftime("%y-%m-%d")
                return day, i_t

        # Define a regular expression pattern to extract variable names from filenames.
        variable_pattern = re.compile(r'\.([A-Za-z0-9]+)\.2D\.nc$')

        for root, _, files in os.walk(dir):
            for filename in sorted(files):
                match = variable_pattern.search(filename)
                if match:
                    var_id = match.group(1)
                    if var_id not in variables_id : 
                        variables_id.append(var_id)
                        days_i_t_per_var_id[var_id] = {}
                        
                    day, i_t = get_day_and_i_t(filename)
                    if day not in list(days_i_t_per_var_id[var_id].keys()):
                        days_i_t_per_var_id[var_id][day] = [i_t]
                    else :
                        days_i_t_per_var_id[var_id][day].append(i_t)
                        
        return variables_id, days_i_t_per_var_id

    def _chek_variables_days_and_i_t(self):
        for var_id in self.variables_names:
            print(var_id)

            days = list(self.days_i_t_per_var_id[var_id].keys())
            print('day:      (#t)  t_i-t_f')
            for day in days:
                i_t_day = self.days_i_t_per_var_id[var_id][day]
                print('%s: (%d) %d-%d'%(day,len(i_t_day), i_t_day[0], i_t_day[-1]))
        print('\n')

    def save_var_id_as_json(self, variables_id, days_i_t_per_var_id, json_filename):
        """
        Save the variables_id and days_i_t_per_var_id as a JSON file.

        :param variables_id: list of variables found
        :param days_i_t_per_var_id: a dictionary containing the days and corresponding indexes per var_id
        :param json_filename: The name of the JSON file to save the data.
        """
        data_to_save = {
            "variables_id": variables_id,
            "days_i_t_per_var_id": days_i_t_per_var_id
        }

        with open(json_filename, 'w') as json_file:
            json.dump(data_to_save, json_file, indent=4)
        print(f"Data saved as {json_filename}")
        
    def load_var_id_from_json(self, json_filename):
        """
        Load the variables_id and days_i_t_per_var_id from a JSON file.

        :param json_filename: The name of the JSON file to load the data from.
        :return: A tuple containing variables_id and days_i_t_per_var_id.
        """
        try:
            with open(json_filename, 'r') as json_file:
                data = json.load(json_file)
                variables_id = data.get("variables_id", [])
                days_i_t_per_var_id = data.get("days_i_t_per_var_id", {})
                print(f"Data loaded from {json_filename}")
                return variables_id, days_i_t_per_var_id
        except FileNotFoundError:
            print(f"File {json_filename} not found.")
            return None, None
        
    def add_new_var_id(self, filepath):
        """
        This function clearly needs a description. In essence it reads the new_variables.json and update var_i_d_days_i_t.json and updates the class attributes accordingly
        It does so by taking into the account the dependencies of the new variables (to select the valid i_t) and points to the function that has to be called 
        to load this variables. The loading func must be in utils.py (future Handler class)
        input : filepath towards new_variables.json
        output : list of new variables names for later usage, dict of dependencies by new var name for debugging, dict of functions to load by key new var names
        """
        def _update_ditvi(var_id, dependency):
            """
            getting passed the new variables names var_id, and the variables names it depends of as a list of string dependency 
            this functions make the intersection of dates and i_t for the new variable.

            dependency is a list of either directly original or new var_id, but it can also contain a +n or -n as to specify an offset in its indexes 
            e.g. :  Prec = Precac - Precac-1 ; dependencies["Prec"] = ["Precac", "Precac-1"]
            """
            ddates = []
            for dvar_id in dependency:
                ## Maybe if finally the date is empty of any i_t should be removed from the keys! even if it should not propagate any issue
                if "+" in dvar_id or "-" in dvar_id:
                    continue 
                else : 
                    ddates.append(list(self.days_i_t_per_var_id[dvar_id].keys()))
            dates = reduce(lambda x, y: list(set(x) & set(y)), ddates)

            i_t_per_date = []
            for i_date, date in enumerate(dates):
                dindexes = []
                for dvar_id in dependency:
                    if "-" in dvar_id:
                        offset = extract_digit_after_sign(dvar_id)
                        if i_date!= 0 : ## Ce n'est pas le 1er jour on peut récupérer celui d'avant
                            prev_date = dates[i_date-1]
                            prev_date_indexes = []
                            for i in range(offset):
                                prev_date_indexes.append(self.days_i_t_per_var_id[dvar_id[:-2]][prev_date][-(i+1)]+offset)
                        this_date_indexes = [self.days_i_t_per_var_id[dvar_id[:-2]][date][i]+1 for i in range(1+len(self.days_i_t_per_var_id[dvar_id[:-2]][date])-offset)]
                        
                        if i_date !=0 : 
                            dindexes.append(prev_date_indexes+this_date_indexes)
                        else : 
                            dindexes.append(this_date_indexes)
                    else : 
                        dindexes.append(list(self.days_i_t_per_var_id[dvar_id][date]))
                i_t_per_date.append(reduce(lambda x, y: list(set(x) & set(y)), dindexes))

            
            self.days_i_t_per_var_id[var_id] = {}
            for i,date in enumerate(dates): 
                self.days_i_t_per_var_id[var_id][date] = i_t_per_date[i]
            return self.days_i_t_per_var_id

        try:
            with open(filepath, 'r') as json_file:
                data=json.load(json_file)
                new_var_names = data.get("variables_id", []) # explicit, will be stored in same dir than other var_id
                dependencies = data.get("dependencies", {}) # dependencies of these variables with available ones.
                                                            # should also be used with a certain keyword for seg mask usage
                functions = data.get("functions", {}) # the name of the functions in utils, that uses the dependent variables
                                                      # from dependencies to compute the new one. Should clean RAM.



                for var_id in new_var_names:
                    if var_id not in self.variables_names:
                        dependency = dependencies[var_id]

                        self.variables_names.append(var_id)
                        self.days_i_t_per_var_id = _update_ditvi(var_id, dependency)
                        print(var_id, self.days_i_t_per_var_id[var_id].keys())

                return new_var_names, dependencies, functions
            
        except FileNotFoundError:
            print(f"File {filepath} not found. Very weird")
            return None, None, None
        

            
        


# Example usage:
# Load the data from "data.json"
# loaded_variables_id, loaded_days_i_t_per_var_id = load_data_from_json("data.json")
