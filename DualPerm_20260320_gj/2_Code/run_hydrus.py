import os
import datetime
import pandas as pd
import numpy as np
import phydrus as ph #Meghan's edits for roots


def run_dual_perm(vgs, calib, atm, atm_hydrus, obs, days, timesteps2):

    #give path to exe file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    exe = os.path.join(BASE_DIR, "H1D_Dual.exe")

    #name the folder files will fill
    main_level = os.path.normpath(os.path.join(BASE_DIR, os.pardir))
    
    newfolder = ''.join(e for e in str(datetime.datetime.today()) if e.isalnum())
    ws = os.path.join(main_level, '3_Results', newfolder)

    #Van Genuchten parameters
    vgs_array = np.array(vgs).reshape(1,len(vgs),17)[0]
    vg_lists = []
    for j in range(len(vgs_array)):
        vgs_1 = vgs_array[j] #take one row 
        vglist1= list(vgs_1)
        vg_lists.append(vglist1)

    #Observation Nodes
    obspts = obs.iloc[:, 2]
    obspts= obspts*-1
    obslist = list(obspts)

    #loop iterating through time steps

    for i in range(len(timesteps2)):
        #basic model info (name, units)
        ml = ph.Model(exe_name=exe, #telling it to use single porosity exe
                ws_name=ws, #folder to create/fill (in 3_Results)
                name="model",
                mass_units="mmol", time_unit="days", length_unit="cm")
        
        ml.basic_info["lFlux"] = True
        
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
        m = ml.get_empty_material_df(n=len(vg_lists))
        
        #these are vg parameters for each depth
        m.loc[1:len(vg_lists)] = vg_lists

        #add materials to function
        ml.add_material(m)
        
        #create soil profile
        profile = ph.create_profile(bot= -150, #depth of soil profile
                                    dx= 1, #grid cells 1 cm
                                    h=-15 #intial pressure head
                                    )
                
        #add  profile to model
        ml.add_profile(profile, qtop= 0.1) #qtop is the water going into cracks vs normal soil, 0-1 range, this means 10% entering cracks
        
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
        ml.add_atmospheric_bc(atm_hydrus,
                            hcrits=0)
        
        #write out input files
        ml.write_input()

        #run Hydrus!
        rs = ml.simulate()

        #check if model ran completely
        #first step: read in output
        tlevel = ml.read_tlevel()

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
        tlevel = ml.read_tlevel(usecols= ['Time','rTopT', 'rRootT', 'vTopT', 'vRootT', 'vBotT', 'sum(rTopT)', \
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
        hydrus_output = hydrus_output.loc[hydrus_output.Date <= max(calib.Date)] # ditch points outside of observation period
        hydrus_output = hydrus_output.loc[hydrus_output.Date >= min(calib.Date)]
        hydrus_output = hydrus_output.drop(columns=["Date"])

        print("run complete")

        # If run returns static VWC values, replace all values with 1s
        if len(np.unique(hydrus_output)) == 1:
            hydrus_output = np.full((np.array(calib.drop(columns=["Date"])).shape), 1.0)

        return np.array(hydrus_output)


    else: 
        print("Run failure")
        #return array with same shape as observation data, filled with 1s
        return np.full((np.array(calib.drop(columns=["Date"])).shape), 1.0)
    


def run_single_por(vgs, calib, atm, atm_hydrus, obs, days, timesteps2):

    #give path to exe file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    exe = os.path.join(BASE_DIR, "H1D_Calc.exe")

    #name the folder files will fill
    main_level = os.path.normpath(os.path.join(BASE_DIR, os.pardir))
    newfolder = ''.join(e for e in str(datetime.datetime.today()) if e.isalnum())
    ws = os.path.join(main_level, '3_Results', newfolder)

    #Van Genuchten parameters
    vg_lists = []
    for j in range(len(vgs)):
        vgs_1 = vgs[j] #take one row 
        vglist1= list(vgs_1)
        vg_lists.append(vglist1)
    #print('vg_lists:', vg_lists)

    #Observation Nodes
    obspts = obs.iloc[:, 2]
    obspts= obspts*-1
    obslist = list(obspts)

    
    #loop iterating through time steps
    for i in range(len(timesteps2)):
   
        #basic model info (name, units)
        ml = ph.Model(exe_name=exe, #telling it to use single porosity exe
                ws_name=ws, #folder to create/fill (in 3_Results)
                name="model",
                mass_units="mmol", time_unit="days", length_unit="cm")
        #add time info
        ml.add_time_info(tinit=0, #first day
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
        ml.add_waterflow(model= 0, #single porosity
                    top_bc=3, #top boundary condition is atmospheric with runoff
                    bot_bc=4, #bottom boundary condition is free drainage
                    )
        
        #this is a dataframe for soil materials, n=1 is number of materials
        m = ml.get_empty_material_df(n=len(vg_lists))
        
        #these are vg parameters for each depth
        m.loc[1:len(vg_lists)] = vg_lists

        #add materials to function
        ml.add_material(m)
        
        #create soil profile
        profile = ph.create_profile(bot= -150, #depth of soil profile
                                    dx= 1, #grid cells 1 cm
                                    h=-15 #intial pressure head
                                    )
               
        #add  profile to model
        ml.add_profile(profile)
        
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
        
        '''
        #make table of root data
        ml.add_root_growth(irootin = 0, 
                        #ngrowth= len(atm_hydrus), #number of data points
                        #tgrowth= atm_hydrus['tAtm'], #days
                        rootdepth= atm4b['RootDepth'] #root depths
                        )

        '''
        ml.add_root_growth(irootin= 2,
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
        ml.add_atmospheric_bc(atm_hydrus)
        
        #write out input files
        ml.write_input()
     
        #run Hydrus!
        ml.simulate()        

        #check if model ran completely
        #first step: read in output
        tlevel = ml.read_tlevel()

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
        tlevel = ml.read_tlevel(usecols= ['Time','rTop', 'rRoot', 'vTop', 'vRoot', 'vBot', 'sum(rTop)', 'sum(rRoot)', 'sum(vTop)',
            'sum(vRoot)', 'sum(vBot)', 'hTop', 'hRoot', 'hBot', 'RunOff', 'sum(RunOff)', 'Volume', 'sum(Infil)', 'sum(Evap)', 'Cum(WTrans)'])
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
        
        # Filter hydrus_output for date and VWC at observation node depth(s)
        cols_to_keep = ["Date"] # Need 'Date'
        for i in range(len(obs)): # loop through all observation depths
            cols_to_keep.append("h" + str(obs.loc[i,"depth"]+1)) # figure out corresponding column in hydrus_ouput
        hydrus_output = alldata.loc[:,cols_to_keep] # ditch columns that aren't date or observation depth VWC
        hydrus_output = hydrus_output.loc[hydrus_output.Date <= max(calib.Date)] # ditch points outside of observation period
        hydrus_output = hydrus_output.loc[hydrus_output.Date >= min(calib.Date)]
        hydrus_output = hydrus_output.drop(columns=["Date"])

        print("run complete")

        # If run returns static VWC values, replace all values with 1.0s
        if len(np.unique(hydrus_output)) == 1:
            hydrus_output = np.full((np.array(calib.drop(columns=["Date"])).shape), 1.0)
        return np.array(hydrus_output)
    
    else: 
        print("Run failure")
        #return array with same shape as observation data, filled with 1.0s
        return np.full((np.array(calib.drop(columns=["Date"])).shape), 1.0)
