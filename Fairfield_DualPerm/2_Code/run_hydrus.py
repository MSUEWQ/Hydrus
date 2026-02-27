import os
import datetime
import pandas as pd
import numpy as np
from EditedFunctions_260226_mr import Model #Meghan's edits for roots
import ProfileFunction as pl #Profile function from GitHub
import multiprocessing as mp



def run_dual_perm(vgs, field):

    #give path to exe file
    exe = os.path.join(os.getcwd(), "H1D_Dual.exe")

    #name the folder files will fill
    main_level = os.path.normpath(os.path.join(os.getcwd(), os.pardir))
    
    newfolder = ''.join(e for e in str(datetime.datetime.today()) if e.isalnum())
    ws = os.path.join(main_level, '3_Results', newfolder)

    #Atmospheric Input
    atm = pd.read_csv(os.path.join(main_level, '1_Input', 'WeatherData.csv'))
    days = len(atm['Date']) #save length of days column for later steps
    irrigation = pd.read_csv(os.path.join(main_level, '1_Input', 'IrrigationData.csv'))

    #merge weather and irrigation and add for a total water input column
    allatm = atm.join(irrigation.set_index('Date'), on='Date')
    allatm['WaterIn_cm'] = allatm['Precip'] + allatm['Amount_cm']

    #set potential evaporation and transpiration equal to zero by default (will edit in loop below)
    allatm['PotenitalEvap'] = 0
    allatm['PotentialTransp'] = 0

    #add month
    allatm['month'] = pd.to_datetime(allatm['Date']).dt.month

    #partition ET to either all evaporation or all transpiration based on month (April-Oct all transpiration, everything else evap)
    #this part is a candidate for editing for more complexity
    for i in range(1,len(allatm)):

        if allatm.loc[i, 'month'] in range(4, 10):
            allatm.loc[i,'PotentialTransp'] = allatm.loc[i, 'ET_cm']
        else:
            allatm.loc[i, 'PotenitalEvap'] = allatm.loc[i,'ET_cm']

        
    #read in roots data
    roots = pd.read_csv(os.path.join(main_level, '1_Input', 'RootData1.csv'))
    allatm = allatm.join(roots.set_index('month'), on= 'month')

    #create new dataframe to manipulate to be exactly what is needed for PHydrus (keeping old version w/ temperature info for now)
    atm2 = allatm[['WaterIn_cm', 'PotenitalEvap', 'PotentialTransp', 'RootDepth_cm']]

    #these are required column names: 'tAtm', 'Prec', 'rSoil', 'rRoot', 'hCritA', 'rB', 'hB', 'hT'
    atm2 = atm2.rename(columns={'PotenitalEvap':'rSoil', 'WaterIn_cm':'Prec', 'PotentialTransp': 'rRoot','RootDepth_cm':'RootDepth' }) #rename columns that are there
    atm2['tAtm'] = range(1,len(atm2)+1)
    atm2['hCritA']= 1000000 #max allowed surface head, this is hydrus default
    atm2['rB']= 0 #bottom flux, is zero for free drainage boundary condition
    atm2['hB']= 0 #groundwater level, is zero if model is not being run with groundwater
    atm2['hT']= 0 #surface pressure head, zero if model uses free drainage

    #rearrange to match format input
    atm2 = atm2[['tAtm', 'Prec', 'rSoil', 'rRoot', 'hCritA', 'rB', 'hB', 'hT', 'RootDepth']]

    #Van Genuchten parameters
    vgs_array = np.array(vgs).reshape(1,len(vgs),17)[0]
    vg_lists = []
    for j in range(len(vgs_array)):
        vgs_1 = vgs_array[j] #take one row 
        vglist1= list(vgs_1)
        vg_lists.append(vglist1)

    #Observation Nodes
    obs = pd.read_csv(os.path.join(main_level, '1_Input', 'ObservationInput.csv'))
    obspts = obs.iloc[:, 2]
    obspts= obspts*-1
    obslist = list(obspts)

    #Timestep list 
    timesteps2 = [0.4, 0.001, 0.003, 0.005, 0.006, 0.0001, 0.0003, 0.0004, 0.01, 0.00002,
                0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00009, 0.0002, 0.0007, 0.0009,
                0.0055, 0.0065, 0.00015, 0.00055, 0.00095] #actual good time steps
    
    #loop iterating through time steps

    for i in range(len(timesteps2)):
        #basic model info (name, units)
        ml = Model(exe_name=exe, #telling it to use single porosity exe
                ws_name=ws, #folder to create/fill (in 3_Results)
                name="model",
                mass_units="mmol", time_unit="days", length_unit="cm")
        
        #add time info
        times = ml.add_time_info(tinit=0, #first day
                            tmax=days, #last day
                            print_times=True, #true if want model to print t level info every day
                            # printinit= 1,
                            #printmax= 40,
                            #dtprint= 1,
                            dt=timesteps2[i], #initial time increment (this is what R iteratres through)
                            dtmax=0.5, #max time increment (didn't change)
                            #printinit=120 #this would specify when to start printing t level info
                            )
        
        #add waterflow information for selector on model type, boundary conditions, can add groundwater here (didn't)
        ml.add_waterflow(model= 9, #dual permeability
                    top_bc=3, #top boundary condition is atmospheric with runoff
                    bot_bc=4, #bottom boundary condition is free drainage
                    )
        
        #this is a dataframe for soil materials, n is number of materials
        m = ml.get_empty_material_df_modified(n=len(vg_lists))
        
        #these are vg parameters for each depth
        m.loc[1:len(vg_lists)] = vg_lists

        #add materials to function
        ml.add_material(m)
        
        #create soil profile
        profile = pl.create_profile(bot= -150, #depth of soil profile
                                    dx= 1, #grid cells 1 cm
                                    h=-15 #intial pressure head
                                    )
                
        #add  profile to model
        ml.add_profile_modified(profile, qtop= 0.1) #qtop is the water going into cracks vs normal soil, 0-1 range, this means 10% entering cracks
        
        #observations nodes
        ml.add_obs_nodes(obslist)
        
        poptms= [-1 for i in range(len(vg_lists))] # -1 for each vg layer
        ml.add_root_uptake(model= 0, #feddes water uptake
                    p0= -1, #pressure head below which roots can extract water from soil
                    p2h= -500, #pressure head below which roots cannot extract water at maximum rate
                    p2l= -900, #manual says same as p2h exceot with potential transpiration of 'r2L'
                    p3= -1600, #wilting point: no root water extraction
                    r2h= 0.5,
                    r2l= 0.1,
                    poptm= poptms, #pressure head below which roots extract water at max rate, 1 per soil type
                    pexp= 3, #exponent for stress response function, accepting default of 3
                    crootmax= 1 #this is max concentration of solute for root uptake, this model doesn't have solutes but isn't working without
                    )
        
        #make table of root data
        ml.add_root_growth(irootin= 2, #this has to be 2 for dual permeability
                        irfak= 0, #root growth is calculated from given data
                        trmin= 1, #day when roots start growing
                        trmed= 150, #time of a specific root growth benchmark
                        trmax= 365, #end of root growth period
                        xrmin= 20, #root depth at start of growth period
                        xrmed= 25, #root depth at trmed time
                        xrmax= 30, #max rooting depth
                        trperiod= 365 #number of days in root growth cycle
                        )
    
        #add atmosphere input to Hydrus
        ml.add_atmospheric_bc(atm2,
                            hcrits=0)
        
        #write out input files
        ml.write_input_modified()

        #run Hydrus!
        rs = ml.simulate()

        #check if model ran completely
        #first step: read in output
        tlevel = ml.read_tlevel_modified()

        #check that output length equals input length, if TRUE stop loop, if FALSE keep going
        if len(tlevel) == days:
        # ml.plots.profile()
            break
        else:
            if i == len(timesteps2)-1:
                failed = pd.DataFrame()
                failed.to_csv(os.path.join(main_level, '3_Results', newfolder, 'FAILED.csv'))
    
    
    #Read in data 
    if ("FAILED.csv" in os.listdir(os.path.join(main_level, '3_Results', newfolder))) == False:
        
        #read in t level and obs node data
        tlevel = ml.read_tlevel_modified(usecols= ['Time','rTopT', 'rRootT', 'vTopT', 'vRootT', 'vBotT', 'sum(rTopT)', \
                                                    'sum(rRoot)', 'sum(vTopT)', 'sum(vRoot)', 'sum(vBotT)', 'hTop',  \
                                                    'hRoot', 'hBot', 'vTopF', 'vTopM', 'sum(vTopM)', 'vFrac', \
                                                    'sum(vFrac)', 'vBotF', 'sum(vBotF)', 'vBotM', 'sum(vBotM)'])
        obsnode = ml.read_obs_node()

        #make obs node data into a dataframe
        allobs = pd.DataFrame()
        
        for i in range(len(obslist)):
            nodenum = obslist[i]*-1+1
            obstemp =obsnode[nodenum]
            obstemp= obstemp.rename(columns={'h':f"h{nodenum}",
                                    'theta':f"theta{nodenum}",
                                    'Temp':f"temp{nodenum}"})
            allobs = pd.concat([allobs, obstemp], axis=1)

        #merge data to create csv file
        output = pd.concat([tlevel, allobs], axis=1)
        outputr = output.reset_index(drop=True)
        alldata = pd.concat([atm, outputr], axis= 1)
        alldata["Date"] = pd.to_datetime(alldata["Date"])

        # save csv as HydrusMergedData.csv in 3_Results
        alldata.to_csv(os.path.join(main_level, '3_Results', newfolder, 'HydrusMergedData.csv'))

        # Filter hydrus_output for date and theta at observation node depth(s)
        cols_to_keep = ["Date"] # Need 'Date'
        for i in range(len(obs)): # loop through all observation depths
            cols_to_keep.append("theta" + str(obs.loc[i,"depth"]+1)) # figure out corresponding column in hydrus_ouput
        hydrus_output = alldata.loc[:,cols_to_keep] # ditch columns that aren't date or observation depth VWC
        hydrus_output = hydrus_output.loc[hydrus_output.Date <= max(field.Date)] # ditch points outside of observation period
        hydrus_output = hydrus_output.loc[hydrus_output.Date >= min(field.Date)]
        hydrus_output = hydrus_output.drop(columns=["Date"])

        print("run complete")

        # If run returns static VWC values, replace all values with 1.0
        if len(np.unique(hydrus_output)) == 1:
            hydrus_output = np.full((np.array(field.drop(columns=["Date"])).shape), 1.0)

        return alldata, np.array(hydrus_output)


    else: 
        print("Run failure")
        #return array with same shape as observation data, filled with 1s
        return alldata, np.full((np.array(field.drop(columns=["Date"])).shape), 1.0)
    


def _worker(queue, params_array, field):
    try:
        result = run_dual_perm(params_array, field)
        queue.put(result)
    except Exception:
        queue.put(None)



def run_with_timeout(params_array, field, fallback_shape, timeout=300):

    queue = mp.Queue()
    p = mp.Process(target=_worker, args=(queue, params_array, field))
    p.start()

    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return np.full(fallback_shape, 1.0)

    if queue.empty():
        return np.full(fallback_shape, 1.0)

    result = queue.get()
    return result if result is not None else np.full(fallback_shape, 1.0)