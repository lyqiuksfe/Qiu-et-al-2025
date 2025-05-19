# -*- coding: utf-8 -*-
"""
Qiu updated on Mar 5, 2025;
"""
import sys
import itertools
import os
import numpy as np
import pandas as pd
sys.path.append("..")
import Modules as Modules
from Setting import Setting


# Optimization configuration
Setting.wall_clock_time_lim = 10000  # seconds
Setting.solver_gap = 0.001  # x100 percent
Setting.print_results_header = 1
Setting.print_detailed_results = 1
Setting.dispatch = False 
# True: run dispatch optimization; False: run investment optimization
Setting.dispatch_unlimited   = False # limit the dispatchable capacity of CCGT to renewable target or not
###############Input data and parameters######################################
Setting.RE_plant_types = ['solar-UPV', 'wind-onshore']
Setting.gas_plant_types = ['CCGT']
Setting.storage_types = ['Li-ion']
Setting.plant_types = Setting.RE_plant_types + Setting.gas_plant_types
######################
Setting.lost_load_thres = np.inf
Setting.val_lost_load=25*1000
experiment_setting=pd.read_csv('../Params/namelist.csv',index_col=0)
experiment_setting=experiment_setting.loc['Standard']
Setting.WACC = float(experiment_setting['WACC'])
Setting.gas_price = float(experiment_setting['gas_price'])
Setting.area_buffer=float(experiment_setting['area_buffer'])
Setting.landr = int(experiment_setting['landr'])
Setting.RE_cell_size= float(experiment_setting['RE_cell_size'])
Setting.weather_model = str(experiment_setting['weather_model'])
Setting.minCF = float(experiment_setting['minCF'])

#######################################################################################
####input from command line
Setting.iso = str(sys.argv[1]) 
isoalias={'ISONE':'ISNE','ERCOT':'ERCO','CAISO':'CISO'} # ['ISONE', 'ERCOT', 'CAISO']
Setting.isoa=isoalias[Setting.iso]

Setting.scenario = str(sys.argv[2]) #['historic', 'rcp85hotter']
Setting.UB_dispatchable_cap['CCGT'] = float(sys.argv[3])  # maximum capacity of CCGT (0-1)
Setting.num_y = int(sys.argv[4]) # number of investment years
ensid = int(sys.argv[5]) # 0 means runs all
Setting.demandsce=str(sys.argv[6]) #'ne': NE alogirhtm; 'mlp';'tell'
#Setting.elec=str(sys.argv[7]) # 'baseline','0.6','0.8','1.0'

# Setting.iso='ISONE'
# Setting.scenario='historic'
# Setting.UB_dispatchable_cap['CCGT'] = 0
# num_y = 3
# ensid = 1
# demandsce='mlp'

# file paths
Setting.popsce='ssp5'
Setting.datadir = '/orcd/nese/mhowland/001/lyqiu/GODEEP/data/'
if Setting.demandsce=='ne':
    Setting.demanddatadir = '/orcd/nese/mhowland/001/lyqiu/GODEEP/demand_ninja/'
    if Setting.scenario=='historic':
        Setting.demandfile = '%s/%s/%s/%s_demand_'%(Setting.demanddatadir,Setting.scenario,Setting.iso,Setting.iso)
    else:
        Setting.demandfile = '%s/%s_%s/%s/%s_demand_'%(Setting.demanddatadir,Setting.scenario,Setting.popsce,Setting.iso,Setting.iso)
elif (Setting.demandsce=='mlp'):
    Setting.demanddatadir = '/orcd/nese/mhowland/001/lyqiu/GODEEP/Demand/Demand_TELL/outputs/'
    if Setting.scenario=='historic':
        Setting.demandfile = '%s/%s_output/%s/'%(Setting.demanddatadir,Setting.demandsce,Setting.scenario)
    else:
        Setting.demandfile = '%s/%s_output/%s_%s/'%(Setting.demanddatadir,Setting.demandsce,Setting.scenario,Setting.popsce)
elif (Setting.demandsce=='tell'):
    Setting.demanddatadir = '/orcd/nese/mhowland/001/lyqiu/GODEEP/Demand/Demand_TELL/outputs/'
    if Setting.scenario=='historic':
        Setting.demandfile = '%s/%s_output/%s/'%(Setting.demanddatadir,Setting.demandsce,Setting.scenario)

Setting.REfile['wind-onshore'] = '%s/%s/wind/%s/wind_gen_cf_' % (Setting.datadir, Setting.scenario, Setting.iso)
Setting.REfile['solar-UPV'] = '%s/%s/solar/%s/solar_gen_cf_' % (Setting.datadir, Setting.scenario, Setting.iso)
Setting.param_dir='../Params/'
if Setting.scenario=='historic':
    #fullyear=list(range(2000,2020))
    #Setting.year_list = random.sample(fullyear, 10)
    #test list is the years not in the year_list
    #Setting.test_list=[x for x in fullyear if x not in Setting.year_list]
    if Setting.demandsce=='ne':
        Setting.year_list=list(range(2001,2008))
    elif Setting.demandsce=='mlp':
        Setting.year_list=list(range(2001,2011))
else:
    if Setting.demandsce=='ne':
        Setting.year_list=list(range(2040,2047))
    elif Setting.demandsce=='mlp':
        Setting.year_list=list(range(2050,2060))

year_lists = list(itertools.combinations(Setting.year_list, Setting.num_y))

Setting.odir='%s/Results_%s_%s1/' % (os.getcwd(),Setting.iso, Setting.demandsce)
if not os.path.exists(Setting.odir):
    os.makedirs(Setting.odir)
suffix='demand_%s_%s_cc_%d_landr_%d_%s_%s_Load.csv' % (Setting.demandsce,Setting.iso,Setting.UB_dispatchable_cap['CCGT']*100, Setting.landr, Setting.weather_model, Setting.scenario)

if ensid != 0:
    csvfilename = '%s/sub%dyrs_ens%d_%s' % (Setting.odir, Setting.num_y, ensid, suffix)
    print(csvfilename)
    if not os.path.exists(csvfilename):
        print('running')
        Setting.sub_year_list = year_lists[ensid-1]
        Setting.ens_id = ensid
        optimization_model = Modules.Model(Setting)
        optimization_model.run_and_save_results()
else:
    for en in range(len(year_lists)):
        ensid = en + 1
        csvfilename = '%s/sub%dyrs_ens%d_%s' % (Setting.odir, Setting.num_y, ensid, suffix)
        print(csvfilename)
        if not os.path.exists(csvfilename):
            print('running')
            Setting.sub_year_list = year_lists[en]
            optimization_model = Modules.Model(Setting)
            optimization_model.run_and_save_results()
        
