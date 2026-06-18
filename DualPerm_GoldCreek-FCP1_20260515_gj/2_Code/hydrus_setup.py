# hydrus_setup.py
from MS_CUA import MscuaSetup, nse_md
from main import run_with_timeout
import numpy as np
import pandas as pd
import spotpy
import os

class HydrusSetup(MscuaSetup):

    # Define parameters here (before __init__) as variables
    thetar = spotpy.parameter.Uniform(low=0.05, high=0.1)
    thetas = spotpy.parameter.Uniform(low=0.35, high=0.5)
    alpha = spotpy.parameter.Uniform(low=0.005, high=0.04)
    n = spotpy.parameter.Uniform(low=1.1, high=2.0)
    ksat = spotpy.parameter.Uniform(low=5.0, high=500)
    #l = 0.5 #spotpy.parameter.Uniform(low=0.5, high=0.5)
    thetar_macro = spotpy.parameter.Uniform(low=0, high=0.1)
    thetas_macro = spotpy.parameter.Uniform(low=0.2, high=0.8)
    alpha_macro = spotpy.parameter.Uniform(low=0.001, high=0.8)
    n_macro = spotpy.parameter.Uniform(low=1.01, high=3)
    ksat_macro = spotpy.parameter.Uniform(low=1, high=1000)
    #l_macro = spotpy.parameter.Uniform(low=0.5, high=0.5)
    w = spotpy.parameter.Uniform(low=0.001, high=0.25)
    beta = spotpy.parameter.Uniform(low=1, high=20)
    #gamma = 0.4 #spotpy.parameter.Uniform(low=0.4, high=0.4)
    a = spotpy.parameter.Uniform(low=0.1, high=10)
    ka = spotpy.parameter.Uniform(low=0.00001, high=0.2)


    def __init__(self, depths, calib, atm, atm_hydrus, obs, days, timesteps2, 
                 calib_cols, settings, trmin, trmax, xrmin, xrmax, params=None):
        super().__init__(objective_funcs=[nse_md])

        # store prepared inputs so simulation() can use them
        self.depths = depths
        self.calib = calib
        self.atm = atm
        self.atm_hydrus = atm_hydrus
        self.obs = obs
        self.days = days
        self.timesteps2 = timesteps2
        self.calib_cols = calib_cols
        self.settings = settings
        self.trmin = trmin
        self.trmax = trmax
        self.xrmin = xrmin
        self.xrmax = xrmax
        self.parameter_dimension = {            
            "thetar": 1, 
            "thetas": 1, 
            "alpha": 1, 
            "n": 1, 
            "ksat": 1,
            #"l": 1,
            "thetar_macro": 1, 
            "thetas_macro": 1, 
            "alpha_macro": 1, 
            "n_macro": 1, 
            "ksat_macro": 1,
            #"l_macro": 1,
            "w": 1,
            "beta": 1,
            #"gamma": 1,
            "a": 1,
            "ka": 1}  # adjust to your param count
        self.param_dim_names = {
            "thetar": "field", 
            "thetas": "field", 
            "alpha": "field", 
            "n": "field", 
            "ksat": "field",
            #"l": "field",
            "thetar_macro": "field", 
            "thetas_macro": "field", 
            "alpha_macro": "field", 
            "n_macro": "field", 
            "ksat_macro": "field",
            #"l_macro": "field",
            "w": "field",
            "beta": "field",
            #"gamma": "field",
            "a": "field",
            "ka": "field"
        }

        # If refined params are passed in, override class-level definitions
        if params is not None:
            for name, values in params.items():
                setattr(self, name, spotpy.parameter.List(name, values[~np.isnan(values)]))

    def simulation(self, params):
        # load data
        main_level = os.path.normpath(os.path.join(os.getcwd(), os.pardir))
        input_files = [f for f in os.listdir(os.path.join(main_level, '1_Input',))]
        vg_csv = [s for s in input_files if "vg" in s.lower()][0]
        vgs_df = pd.read_csv(os.path.join(main_level, '1_Input', vg_csv))

        params_df = pd.DataFrame(params, index=[0])
        #thetar = 0.0691
        #thetas = 0.4234
        #alpha = 0.0061
        #n = 1.5783
        #ksat = 14.6
        l = 0.5
        l_macro = 0.5
        gamma = 0.4
        params_array = []
        for i in range(len(vgs_df)):
            if vgs_df.loc[i, "CalibrateYN"] == "N":
                params_array.append(list(vgs_df.loc[i,"thetar":]))
            elif vgs_df.loc[i, "CalibrateYN"] == "Y":
                for row in range(len(params_df)):
                    row_array = list(params_df.iloc[row, : ])
                    #row_array.insert(0, thetar)
                    #row_array.insert(1, thetas)
                    #row_array.insert(2, alpha)
                    #row_array.insert(3, n)
                    #row_array.insert(4, ksat)
                    row_array.insert(5, l)
                    row_array.insert(11, l_macro)
                    row_array.insert(14, gamma)
                    params_array.append(list(row_array))
        
        #print(params_array)

        fallback = self.calib.copy().drop(columns=["Date"])
        for col, val in zip(self.settings["FieldDataColumn"], self.settings["FailValue"]):
            fallback[col] = val
        fallback = np.array(fallback).reshape(len(fallback), -1)
        
        hydrus_output = run_with_timeout(
            params_array, self.depths, self.calib, self.atm, self.atm_hydrus, 
            self.obs, self.days, self.timesteps2, self.calib_cols, self.settings,
            self.trmin, self.trmax, self.xrmin, self.xrmax,
            fallback=fallback,
            timeout=300
        )
        #print("simulation output shape:", hydrus_output.shape)
        return hydrus_output

    def evaluation(self):
        return np.array(self.calib.drop(columns=["Date"]))

    def objectivefunction(self, observation, simulation):
        return nse_md(observation, simulation, return_dict=True, axis=1)