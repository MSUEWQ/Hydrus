# hydrus_setup.py
from MS_CUA import MscuaSetup, nse_md
from main import run_with_timeout
import numpy as np
import pandas as pd
import spotpy

class HydrusSetup(MscuaSetup):

    # Define parameters here (before __init__) as variables
    thetar = spotpy.parameter.Uniform(low=0.05, high=0.1)
    thetas = spotpy.parameter.Uniform(low=0.35, high=0.5)
    alpha = spotpy.parameter.Uniform(low=0.005, high=0.04)
    n = spotpy.parameter.Uniform(low=1.1, high=2.0)
    ksat = spotpy.parameter.Uniform(low=5.0, high=500)
    l = spotpy.parameter.Uniform(low=0.5, high=0.5)
    thetar_macro = spotpy.parameter.Uniform(low=0, high=0.1)
    thetas_macro = spotpy.parameter.Uniform(low=0.2, high=0.8)
    alpha_macro = spotpy.parameter.Uniform(low=0.001, high=0.8)
    n_macro = spotpy.parameter.Uniform(low=1.01, high=7)
    ksat_macro = spotpy.parameter.Uniform(low=1, high=1000)
    l_macro = spotpy.parameter.Uniform(low=0.5, high=0.5)
    w = spotpy.parameter.Uniform(low=0.001, high=0.25)
    beta = spotpy.parameter.Uniform(low=1, high=20)
    gamma = spotpy.parameter.Uniform(low=0.4, high=0.4)
    a = spotpy.parameter.Uniform(low=0.1, high=10)
    ka = spotpy.parameter.Uniform(low=0.00001, high=10)

    def __init__(self, calib, atm, atm_hydrus, obs, days, timesteps2):
        super().__init__(objective_funcs=[nse_md])
        # store prepared inputs so simulation() can use them
        self.calib = calib
        self.atm = atm
        self.atm_hydrus = atm_hydrus
        self.obs = obs
        self.days = days
        self.timesteps2 = timesteps2
        self.parameter_dimension = {            
            "thetar": 1, 
            "thetas": 1, 
            "alpha": 1, 
            "n": 1, 
            "ksat": 1,
            "l": 1,
            "thetar_macro": 1, 
            "thetas_macro": 1, 
            "alpha_macro": 1, 
            "n_macro": 1, 
            "ksat_macro": 1,
            "l_macro": 1,
            "w": 1,
            "beta": 1,
            "gamma": 1,
            "a": 1,
            "ka": 1}  # adjust to your param count
        self.param_dim_names = {
            "thetar": "field", 
            "thetas": "field", 
            "alpha": "field", 
            "n": "field", 
            "ksat": "field",
            "l": "field",
            "thetar_macro": "field", 
            "thetas_macro": "field", 
            "alpha_macro": "field", 
            "n_macro": "field", 
            "ksat_macro": "field",
            "l_macro": "field",
            "w": "field",
            "beta": "field",
            "gamma": "field",
            "a": "field",
            "ka": "field"
        }

    def simulation(self, params):
        params_df = pd.DataFrame(params, index=[0])
        params_array = []
        for row in range(len(params_df)):
            df = params_df.iloc[row, : ]
            params_array.append(list(df))
        print(params_array)
        fallback = np.full((np.array(self.calib.drop(columns=["Date"])).shape), np.nan)
        _, hydrus_output = run_with_timeout(
            params_array, self.calib, self.atm, self.atm_hydrus, self.obs, 
            self.days, self.timesteps2,
            fallback=fallback,
            timeout=30
        )
        return hydrus_output

    def evaluation(self):
        return np.array(self.calib.drop(columns=["Date"]))

    def objectivefunction(self, observation, simulation):
        return nse_md(observation, simulation, return_dict=True)