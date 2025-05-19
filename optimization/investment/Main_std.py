# -*- coding: utf-8 -*-
"""
Qiu updated on Mar 5, 2025;
"""
import pandas as pd
import sys
import xarray as xr
import os

sys.path.append("..")
import NCDataCost as NCData
import Modules as Modules

class Setting:
    demandfile = str()
    RE_cell_size = float()  # degree
    RE_plant_types = list()  # set of RE plant types considered
    REfile = dict()
    solver_gap = float()
    wall_clock_time_lim = int()
    weather_model = str()
    print_results_header = bool()
    print_detailed_results = bool()
    test_name = str()
    datadir = str()
    UB_dispatchable_cap = dict()
    lost_load_thres = float()
    gas_price = float()
    storage_types = list()
    plant_types = list()
    wake = int()
    gas_plant_types = list()
    val_lost_load = float()
    val_curtail = float()
    demandfile = str()
    csvdir = str()
    test_years = list()
    suffix = str()
    CF = dict()

# Optimization configuration
Setting.wall_clock_time_lim = 10000  # seconds
Setting.solver_gap = 0.001  # x100 percent
Setting.print_results_header = 1
Setting.print_detailed_results = 1
Setting.startrun=False
Setting.getcf=True
Setting.dispatch=True
Setting.dispatch_unlimited   = False
###############Input data and parameters######################################
Setting.RE_plant_types = ['solar-UPV', 'wind-onshore']
Setting.gas_plant_types = ['CCGT']
Setting.storage_types = ['Li-ion']
Setting.plant_types = Setting.RE_plant_types + Setting.gas_plant_types

######################
Setting.val_lost_load = 10*1000;
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
isoalias={'ISONE':'ISNE','ERCOT':'ERCO','CAISO':'CISO'}
Setting.isoa=isoalias[Setting.iso]

Setting.invest_scenario = str(sys.argv[2]) #invest scenario
Setting.UB_dispatchable_cap['CCGT'] = float(sys.argv[3])  # maximum capacity of CCGT (0-1)
Setting.invest_num_y=int(sys.argv[4])
Setting.invest_ens_id_s=int(sys.argv[5])
Setting.invest_ens_id_e=int(sys.argv[6])
Setting.demandsce=str(sys.argv[7]) #'ne': NE alogirhtm; 'mlp';'tell'
Setting.scenario = str(sys.argv[8]) #test scenario
Setting.capacity_factor=float(sys.argv[9])



# Setting.iso ='ISONE'
# isoalias={'ISONE':'ISNE','ERCOT':'ERCO','CAISO':'CISO'}
# Setting.isoa=isoalias[Setting.iso]
# Setting.invest_scenario='historic'
# Setting.UB_dispatchable_cap['CCGT'] = 0
# Setting.invest_num_y=3
# Setting.invest_ens_id_s=1
# Setting.invest_ens_id_e=5
# Setting.demandsce='mlp'
# Setting.scenario = 'rcp85hotter'
# Setting.capacity_factor=1.0

Setting.popsce='ssp5'
Setting.mdir = '/orcd/nese/mhowland/001/lyqiu/GODEEP/'

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
    Setting.demanddatadir = '/orcd/nese/mhowland/001/lyqiu/tell_data/outputs/'
    if Setting.scenario=='historic':
        Setting.demandfile = '%s/%s_output/%s/'%(Setting.demanddatadir,Setting.demandsce,Setting.scenario)

Setting.REfile['wind-onshore'] = '%s/%s/wind/%s/wind_gen_cf_' % (Setting.datadir, Setting.scenario, Setting.iso)
Setting.REfile['solar-UPV'] = '%s/%s/solar/%s/solar_gen_cf_' % (Setting.datadir, Setting.scenario, Setting.iso)
Setting.param_dir='../Params/'

# test years
if Setting.scenario == 'historic':
    Setting.test_year = list(range(2001, 2011))
else:
    Setting.test_year = list(range(2020, 2030))

Setting.num_y = len(Setting.test_year)

Setting.REfile['wind-onshore'] = '%s/data/%s/wind/%s/wind_gen_cf_' % (Setting.mdir, Setting.scenario, Setting.iso)
Setting.REfile['solar-UPV'] = '%s/data/%s/solar/%s/solar_gen_cf_' % (Setting.mdir, Setting.scenario, Setting.iso)


#import CF data
for tech in Setting.RE_plant_types:
    for iy in Setting.test_year:
        CF_orig = xr.open_dataset(Setting.REfile[tech]+str(iy)+'.nc')['capacity_factor']
        CF_orig['lat'] = CF_orig['lat'].round(2)
        CF_orig['lon'] = CF_orig['lon'].round(2)
        if iy==Setting.test_year[0]:
            CF=CF_orig
        else:
            CF=xr.concat([CF,CF_orig],dim='Time')
    CF = CF.stack(z=('y', 'x')).dropna(dim='z')
    raw_time = CF['Time'].values
    dates = raw_time.astype(str)
    try:
        base_dates = dates.astype(float).astype(int)  # e.g., 20200101
        frac_days = dates.astype(float) - base_dates  # e.g., 0.04166667
        base_datetimes = pd.to_datetime(base_dates, format="%Y%m%d")
        final_times = base_datetimes + pd.to_timedelta(frac_days, unit='D')
        final_times = final_times.round('h')
    except:
        final_times=pd.to_datetime(dates)
    CF['Time'] = final_times
    #round to the nearest hour
    CF_df = CF.to_dataframe()[['capacity_factor', 'lat', 'lon']].reset_index()
    Setting.CF[tech] = CF_df
Setting.odir='%s/Results_%s_%s_cf/' % (os.getcwd(),Setting.iso, Setting.demandsce)
if not os.path.exists(Setting.odir):
    os.makedirs(Setting.odir)
Setting.csvdir='%s/investment/Results_%s_%s/' % ("../",Setting.iso, Setting.demandsce)
for ensid in range(Setting.invest_ens_id_s,Setting.invest_ens_id_e+1):
    Setting.invest_ens_id = ensid
    Setting.suffix='sub%dyrs_ens%d_demand_%s_%s_cc_%d_landr_%d_%s_%s'%(Setting.invest_num_y,Setting.invest_ens_id,
                                                            Setting.demandsce,Setting.iso,
                                                            Setting.UB_dispatchable_cap['CCGT']*100,
                                                            Setting.landr,Setting.weather_model,
                                                            Setting.invest_scenario)

    if Setting.startrun==True:
        outputfile = "%s/%s_test_%s_%.3f_Load.csv" % (Setting.odir, Setting.suffix, Setting.scenario, Setting.capacity_factor)
        print(outputfile)
        if not os.path.exists(outputfile):
            print('dispatching')
            model=Modules.Model(Setting)
            model.run_and_save_results()
    if Setting.getcf==True:
        dat = NCData.Data(Setting)
        
        for r, replant in enumerate(Setting.RE_plant_types):
            pindex = Setting.plant_types.index(replant)
            sptialvar=dat.Plants[pindex].prod_sptialvar.reset_index()
            sptialvarf=f'{Setting.odir}/{Setting.suffix}_{replant}_sptialvar.csv'
            sptialvar.to_csv(sptialvarf,index=False)
            temporalvar=dat.Plants[pindex].prod_temporalvar.reset_index()
            temporalvarf=f'{Setting.odir}/{Setting.suffix}_{replant}_temporalvar.csv'
            temporalvar.to_csv(temporalvarf,index=False)
