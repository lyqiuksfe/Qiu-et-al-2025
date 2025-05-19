# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:56:36 2023
Read problem data for the wind-solar siting optimization problem
@author: Rahman Khorramfar
"""
import numpy as np
import pandas as pd
import xarray as xr
import logging

def get_df(df, land=None, cg=None, size=None, ny=None, ensid=None,scenario=None,demand_data=None):
    df = df.reset_index(drop=True)
    df = df.set_index(['landres_allowed', 'upper_bound_CCGT','cell_size_solar-UPV', 'inv_num_y', 'inv_ens_id', 'investment_sce','demand_data'])
    if ny is not None:
        dfa = df.iloc[df.index.get_level_values('inv_num_y') == ny]
    else:
        dfa = df
    if land is not None:
        dfa = dfa.iloc[dfa.index.get_level_values('landres_allowed') == land]
    if cg is not None:
        dfa = dfa.iloc[dfa.index.get_level_values('upper_bound_CCGT') == cg]
    if size is not None:
        dfa = dfa.iloc[dfa.index.get_level_values('cell_size_solar-UPV') == size]
    if ensid is not None:
        dfa = dfa.iloc[dfa.index.get_level_values('inv_ens_id') == ensid]
    if scenario is not None:
        dfa = dfa.iloc[dfa.index.get_level_values('investment_sce') == scenario]
    if demand_data is not None:
        dfa = dfa.iloc[dfa.index.get_level_values('demand_data') == demand_data]
    return dfa

class Storage:
    def __init__(self):  # no constructor
        super().__init__()
        self.Type = str()
        self.CAPEX = float()
        self.eff_round = float()
        self.FOM = float()
        self.lifetime = int()
        self.est_coef = float()
        self.num_storages = int()
        self.duration = int()
        self.decay_eff = float()
        ####diapatch mode only
        self.sLev=float()
        self.invcost=float()
    def populate_storage_data(self, Setting,cp_data=None):
        self.Storages = list()
        df = pd.read_csv(f'{Setting.param_dir}/Storage_params.csv', index_col=0)
        for storage_type in Setting.storage_types:
            strg = Storage()
            strg.Type = storage_type
            strg.CAPEX = (df.loc[storage_type,'EnergyCAPEX($/kWh)']+df.loc[storage_type,'PowerCAPEX($/kW)']
                          /df.loc[storage_type,'duration(hr)'])*1000  # per MWh
            strg.eff_round = df.loc[storage_type,'Round-tripEfficiency']
            strg.FOM = strg.CAPEX*0.025
            strg.lifetime = int(df.loc[storage_type,'lifetime'])  # yr
            strg.est_coef = Setting.WACC * \
                ((1+Setting.WACC)**strg.lifetime) / \
                ((1+Setting.WACC)**strg.lifetime-1)
            strg.duration = int(df.loc[storage_type,'duration(hr)'])  # hr
            strg.decay_eff = df.loc[storage_type,'DecayRate']
            if Setting.dispatch:
                strg.sLev = cp_data[f'capacity_strg'].values[0]
                strg.invcost = cp_data[f'total_cost_strg'].values[0]
            self.Storages.append(strg)
        self.num_storages = len(self.Storages)


class Plant:
    def __init__(self):  # no constructor
        self.Type = str()  # name of the plant type, e.g., solar, wind-offshore
        self.CAPEX = float()
        self.FOM = float()
        self.VOM = float()
        self.heat_rate = float()
        self.lifetime = int()
        self.est_coef = float()
        self.stable = float()
        self.density = float()

        self.CF = list()
        self.lat = list()
        self.lon = list()
        self.area = list()
        self.num_loc = int()

        self.capacity=float()
        self.invcost = float()
        self.prod=list()
        self.prod_seperate=pd.DataFrame()

    def populate_plant_data(self, Setting,cp_data=None):
        self.Plants = list()
        df = pd.read_csv(f'{Setting.param_dir}/Plant_params.csv', index_col=0)
        for plant_type in Setting.plant_types:
            plt = Plant()
            plt.Type = plant_type
            plt.CAPEX = df.loc[plant_type,'CAPEX($/kw)']*1000  # to MW
            plt.FOM = df.loc[plant_type,'FOM($/kW-yr)']*1000  # to MW
            plt.VOM = df.loc[plant_type,'VOM($/MWh)']
            plt.heat_rate = df.loc[plant_type,'HeatRate(MMBtu/MWh)']
            plt.lifetime = int(df.loc[plant_type,'Lifetime(year)'])
            plt.est_coef = Setting.WACC * \
                ((1+Setting.WACC)**plt.lifetime) / \
                ((1+Setting.WACC)**plt.lifetime-1)
            plt.stable = df.loc[plant_type,'MinOutput(%)']
            plt.density = df.loc[plant_type,'CapacityDensity(MW/m2)']
            if Setting.dispatch:
                plt.capacity = cp_data[f'capacity_{plant_type}'].values[0]*Setting.capacity_factor
                if plant_type in Setting.gas_plant_types:
                    plt.invcost = cp_data[f'inv_cost_{plt.Type}'].values[0]*Setting.capacity_factor
                elif plant_type in Setting.RE_plant_types:
                    plt.invcost = cp_data[f'total_cost_{plt.Type}'].values[0]

            if plant_type in Setting.RE_plant_types:
                if (Setting.dispatch):
                    if Setting.demandsce == 'mlpdr':
                        suffix='sub%dyrs_ens%d_demand_%s_%s_cc_%d_landr_%d_%s_%s'%(Setting.invest_num_y,Setting.invest_ens_id,
                                                            'mlp',Setting.iso,Setting.UB_dispatchable_cap['CCGT']*100,
                                                            Setting.landr,Setting.weather_model,Setting.invest_scenario)
                        locations = pd.read_csv(Setting.csvdir+suffix+f'_{plant_type}_locations.csv')
                    else:
                        locations = pd.read_csv(Setting.csvdir + Setting.suffix + f'_{plant_type}_locations.csv')
                    locations['lat'] = locations['lat'].round(2)
                    locations['lon'] = locations['lon'].round(2)
                    locations['capacity'] = locations['capacity']*Setting.capacity_factor
                    power_DF = pd.merge(locations, Setting.CF[plant_type], how='left', on=['lat', 'lon'])
                    power_DF['power']=power_DF['capacity']*power_DF['capacity_factor']
                    power_sum = power_DF.groupby('Time').sum()['power']
                    if Setting.getcf:
                        print(power_DF.shape)
                        power_DF['Month']=power_DF['Time'].dt.month
                        #plt.prod_seperate = power_DF
                        plt.prod_sptialvar= power_DF.groupby(['Time']).std()['capacity_factor']
                        plt.prod_temporalvar= power_DF.groupby(['lat','lon','Month']).std()['capacity_factor']
                    plt.prod = power_sum.to_numpy()
                    #variability of the capacity factor
                    
                    print(plt.prod.shape)
                else:
                    for iy in Setting.sub_year_list:
                        CF_orig = xr.open_dataset(Setting.REfile[plant_type] + str(iy) + '.nc')['capacity_factor']
                        data = CF_orig.stack(z=("y", "x")).dropna('z', how='all')
                        plt.CF.append(data.data)
                    plt.CF = np.concatenate(plt.CF, axis=0)
                    plt.CF = np.where(plt.CF < Setting.minCF, 0, plt.CF)
                    plt.area = Setting.RE_cell_size * Setting.RE_cell_size  # km2
                    plt.lat = data.lat.data
                    plt.lon = data.lon.data
                    plt.num_loc = plt.CF.shape[1]
                    logging.info(f"{plt.Type} data has {plt.num_loc} potential locations")
            self.Plants.append(plt)
        self.num_plants = len(self.Plants)


class Data(Plant, Storage):
    def __init__(self, Setting):  
        super().__init__()
        self.Plants = []
        self.Storages = []# no constructor

        if Setting.dispatch==True:
            cp_data = pd.read_csv(f'{Setting.csvdir}/{Setting.weather_model}_{Setting.iso}_General_Results.csv')      
            cp_data = get_df(cp_data,land= Setting.landr,cg=Setting.UB_dispatchable_cap['CCGT'],size=Setting.RE_cell_size,ny=Setting.invest_num_y,ensid=Setting.invest_ens_id,scenario=Setting.invest_scenario)
            num=len(cp_data)
            if num>1:
                logging.error('More than one row is returned for the dispatchable capacity')
                exit()
            self.populate_plant_data(Setting,cp_data)
            self.populate_storage_data(Setting,cp_data)
        else:
            self.populate_plant_data(Setting)
            self.populate_storage_data(Setting)

        dfs=[]
        if Setting.dispatch==True:
            years_to_read=Setting.test_year
        else:
            years_to_read=Setting.sub_year_list
        for iy in years_to_read:
            if Setting.demandsce == 'ne':
                dft = pd.read_csv(Setting.demandfile+str(iy)+'_'+Setting.elec+'.csv',index_col=0)[['total_demand']]
                dft['total_demand'] = dft['total_demand']*1000. # to MW
            elif Setting.demandsce in ['mlp','tell']:
                dft = pd.read_csv("%s/%d/%s_%d_%s_output.csv"%(Setting.demandfile,iy,Setting.isoa,iy,Setting.demandsce),index_col=0)[['Load']]
                dft=dft.rename(columns={'Load':'total_demand'})
            elif Setting.demandsce == 'mlpdr':
                dft = pd.read_csv("%s/%d/%s_%d_mlp_output.csv" % (Setting.demandfile, iy, Setting.isoa, iy), index_col=0)[['Load']]
                dft = dft.rename(columns={'Load': 'total_demand'})
            dfs.append(dft)
        df = pd.concat(dfs, axis=0)
        if Setting.demandsce == 'ne':
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            df.index = pd.to_datetime(df.index)
        df = df[~((df.index.month == 2) & (df.index.day == 29))]
        self.demand = df['total_demand'].to_numpy()
        self.num_plan_periods = len(self.demand)
        self.dates=df.index
