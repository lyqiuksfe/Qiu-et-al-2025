# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 08:36:23 2023
Last Update: 2023 Oct 17
@original author: Rahman Khorramfar
@modified by: Liying Qiu
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum, LinExpr
from DVs import DV, DV_val
import time
import os
import NCDataCost as NCDataCost


class Model:
    def __init__(self, Setting):
        self.dat = NCDataCost.Data(Setting)
        self.Setting = Setting
        self.opt_data = self.fetch_data()
        self.Model = gp.Model()

    def fetch_data(self):
        opt_data={
            'nT': self.dat.demand.shape[0],
            'nPlt': self.dat.num_plants,
            'nREPlt': len(self.Setting.RE_plant_types),
            'nStr': self.dat.num_storages,
            'nYear': self.Setting.num_y,
            'nDPlt': len(self.Setting.gas_plant_types),
        }
        if self.Setting.dispatch:
            gas_inv_cost = []
            Xg=[]
            Xr=[]
            RE_cost = []
            for pindex,aplant in enumerate(self.Setting.plant_types):
                plant = self.dat.Plants[pindex]
                single_cost = (plant.est_coef*plant.CAPEX+plant.FOM)
                if aplant in self.Setting.gas_plant_types:
                    gas_inv_cost.append(single_cost * plant.capacity)
                    Xg.append(plant.capacity)
                elif aplant in self.Setting.RE_plant_types:
                    RE_cost.append(single_cost * plant.capacity)
                    Xr.append(plant.capacity)

            plant = self.dat.Storages[0]
            Xstr = plant.sLev
            strg_inv_cost = (plant.est_coef * plant.CAPEX + plant.FOM) * Xstr
            opt_data['RE_cost'] = RE_cost
            opt_data['gas_inv_cost'] = gas_inv_cost
            opt_data['strg_inv_cost'] = strg_inv_cost
            opt_data['Xr'] = Xr
            opt_data['Xg'] = Xg
            opt_data['Xstr'] = Xstr
            print(opt_data['Xg'])
        else:
            RE_nlocations = []
            for replant in self.Setting.RE_plant_types:
                pindex = self.Setting.plant_types.index(replant)
                RE_nlocations.append(self.dat.Plants[pindex].
                                    num_loc)
            opt_data['RE_nlocations'] = RE_nlocations
        
        return opt_data
        

    # define decision variables
    def define_DVs(self):
        if not self.Setting.dispatch:
            DV.Xr = [self.Model.addVars(nloc, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS) for nloc in self.opt_data['RE_nlocations']]
            DV.Xstr = self.Model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
            DV.Xg = self.Model.addVars(self.opt_data['nDPlt'], lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
        # Other Decisions
        DV.prod = self.Model.addVars(self.opt_data['nPlt'], self.opt_data['nT'], lb=0, vtype=GRB.CONTINUOUS)
        DV.load = self.Model.addVars(self.opt_data['nPlt'], self.opt_data['nT'], lb=0, vtype=GRB.CONTINUOUS)
        DV.LL = self.Model.addVars(self.opt_data['nT'], lb=0, vtype=GRB.CONTINUOUS)
        DV.sCh = self.Model.addVars(self.opt_data['nT'], lb=0, vtype=GRB.CONTINUOUS)
        DV.sDis = self.Model.addVars(self.opt_data['nT'], lb=0, vtype=GRB.CONTINUOUS)
        DV.sLev = self.Model.addVars(self.opt_data['nT'], lb=0, vtype=GRB.CONTINUOUS)

    def obj_investments(self):
        obj_inv=gp.LinExpr()
        DV.RE_cost = self.Model.addVars(self.opt_data['nREPlt'], vtype=GRB.CONTINUOUS)
        DV.gas_inv_cost = self.Model.addVars(self.opt_data['nDPlt'], vtype=GRB.CONTINUOUS)
        DV.strg_inv_cost = self.Model.addVar(vtype=GRB.CONTINUOUS)
        DV.total_inv_cost = self.Model.addVar(vtype=GRB.CONTINUOUS)
        RE_cost = list()
        for r, replant in enumerate(self.Setting.RE_plant_types):
            pindex = self.Setting.plant_types.index(replant)
            plant = self.dat.Plants[pindex]
            single_cost = (plant.est_coef * plant.CAPEX+plant.FOM)
            r_cost = gp.LinExpr()
            for i in range(self.opt_data['RE_nlocations'][r]):
                r_cost.addTerms(single_cost, DV.Xr[r][i])
            obj_inv += r_cost
            RE_cost.append(r_cost)

        gas_inv_cost = list()
        for d, dplant in enumerate(self.Setting.gas_plant_types):
            pindex = self.Setting.plant_types.index(dplant)
            plant = self.dat.Plants[pindex]
            single_cost = (plant.est_coef*plant.CAPEX+plant.FOM)
            gg_cost = gp.LinExpr()
            gg_cost.addTerms(single_cost, DV.Xg[d])
            obj_inv += gg_cost
            gas_inv_cost.append(gg_cost)
        
        strg_inv_cost = gp.LinExpr()
        plant = self.dat.Storages[0]
        strg_inv_cost.addTerms((plant.est_coef*plant.CAPEX+plant.FOM), DV.Xstr)
        obj_inv += strg_inv_cost

        for r in range(self.opt_data['nREPlt']):
            self.Model.addConstr(DV.RE_cost[r] == RE_cost[r])
        for d in range(self.opt_data['nDPlt']):
            self.Model.addConstr(DV.gas_inv_cost[d] == gas_inv_cost[d])
        self.Model.addConstr(DV.strg_inv_cost == strg_inv_cost)
        self.Model.addConstr(DV.total_inv_cost == obj_inv)
        return obj_inv

    def obj_dispatch(self):
        obj_dispatch = gp.LinExpr()
        DV.fuel_cost = self.Model.addVars(self.opt_data['nDPlt'], vtype=GRB.CONTINUOUS)
        DV.var_cost = self.Model.addVars(self.opt_data['nDPlt'], vtype=GRB.CONTINUOUS)
        DV.lost_load_cost = self.Model.addVar(vtype=GRB.CONTINUOUS)
        DV.total_dispatch_cost = self.Model.addVar(vtype=GRB.CONTINUOUS)
        
        var_cost = list()
        fuel_cost = list()
        for d, dplant in enumerate(self.Setting.gas_plant_types):
            pindex = self.Setting.plant_types.index(dplant)
            plant = self.dat.Plants[pindex]
            vv_cost = gp.LinExpr()
            ff_cost = gp.LinExpr()
            for t in range(self.opt_data['nT']):
                vv_cost.addTerms(plant.VOM/self.opt_data['nYear'], DV.prod[pindex, t])
                ff_cost.addTerms(self.Setting.gas_price*plant.heat_rate/self.opt_data['nYear'], DV.prod[pindex, t])
            obj_dispatch += vv_cost
            obj_dispatch += ff_cost
            var_cost.append(vv_cost)
            fuel_cost.append(ff_cost)

        lost_load_cost = gp.LinExpr()
        for t in range(self.opt_data['nT']):
            lost_load_cost.addTerms(self.Setting.val_lost_load/self.opt_data['nYear'], DV.LL[t])
        obj_dispatch += lost_load_cost

        for d in range(self.opt_data['nDPlt']):
            self.Model.addConstr(DV.var_cost[d] == var_cost[d])
            self.Model.addConstr(DV.fuel_cost[d] == fuel_cost[d])
        self.Model.addConstr(DV.lost_load_cost == lost_load_cost)
        self.Model.addConstr(DV.total_dispatch_cost == obj_dispatch)
        return obj_dispatch
    
    def add_obj_func(self):
        DV.total_cost =self.Model.addVar(vtype=GRB.CONTINUOUS)           
        obj = gp.LinExpr()
        if self.Setting.dispatch:
            obj_dispatch = self.obj_dispatch()
            obj=obj_dispatch
        else:
            obj_dispatch = self.obj_dispatch()
            obj_investment = self.obj_investments()
            obj = obj_investment + obj_dispatch
        self.Model.addConstr(DV.total_cost == obj)
        self.Model.setObjective(obj, GRB.MINIMIZE)


    def add_constraints(self):
        # c1: natural gas production < natural gas capacity
        for d, dplant in enumerate(self.Setting.gas_plant_types):
            pindex = self.Setting.plant_types.index(dplant)
            if self.Setting.dispatch:  
                self.Model.addConstrs(DV.prod[pindex, t] <= self.opt_data['Xg'][d] for t in range(self.opt_data['nT']))
            else:
                self.Model.addConstrs(DV.prod[pindex, t] <= DV.Xg[d] for t in range(self.opt_data['nT']))
        
        
        # c2: renewable: production = CF*capacity
        for r, replant in enumerate(self.Setting.RE_plant_types):
            pindex = self.Setting.plant_types.index(replant)
            for t in range(self.opt_data['nT']):
                if self.Setting.dispatch:
                    self.Model.addConstr(DV.prod[pindex, t] == self.dat.Plants[pindex].prod[t],name=f'c_REprod_{t}_{replant}')
                else:
                    lhs_expr = gp.LinExpr()
                    for i in range(self.opt_data['RE_nlocations'][r]):
                        lhs_expr.addTerms(self.dat.Plants[pindex].CF[t, i], DV.Xr[r][i])
                    self.Model.addConstr(DV.prod[pindex, t] == lhs_expr,name=f'c_REprod_{t}_{replant}')
        if self.Setting.dispatch_unlimited   ==False:
            # c3: renewable target
            for d, dplant in enumerate(self.Setting.gas_plant_types):
                pindex = self.Setting.plant_types.index(dplant)
                ff_load = gp.LinExpr()
                for t in range(self.opt_data['nT']):
                    ff_load.addTerms(1, DV.load[pindex, t])
                self.Model.addConstr(ff_load <= self.dat.demand.sum()*self.Setting.UB_dispatchable_cap[dplant], name=f'c_ffload_cap_{dplant}')

        # c4: balance constraints
        for t in range(self.opt_data['nT']):
            lhs_expr = gp.LinExpr()
            for p in range(self.opt_data['nPlt']):
                lhs_expr.addTerms(1, DV.load[p, t])
            lhs_expr.addTerms(1, DV.sDis[t])
            lhs_expr.addTerms(1, DV.LL[t])
            lhs_expr.addTerms(-1, DV.sCh[t])
            self.Model.addConstr(lhs_expr == self.dat.demand[t], name=f'c_demand_{t}')

        # c5: storage constraints
        plant = self.dat.Storages[0]
        #update 2025
        for t in range(self.opt_data['nT']):
            lhs_expr = gp.LinExpr()
            for p in range(self.opt_data['nPlt']):
                lhs_expr.addTerms(1, DV.load[p, t])
            self.Model.addConstr(DV.sCh[t] <= lhs_expr)
        ###########
        if self.Setting.dispatch:
            self.Model.addConstrs(DV.sCh[t] <= (self.opt_data['Xstr']/plant.duration) for t in range(self.opt_data['nT']))
            self.Model.addConstrs(DV.sDis[t] <= (self.opt_data['Xstr']/plant.duration) for t in range(self.opt_data['nT']))
            self.Model.addConstrs(DV.sLev[t] <= self.opt_data['Xstr'] for t in range(0, self.opt_data['nT']))
            self.Model.addConstr(DV.sLev[0] == self.opt_data['Xstr']/2)
        else:
            self.Model.addConstrs(DV.sCh[t] <= (DV.Xstr/plant.duration) for t in range(self.opt_data['nT']))
            self.Model.addConstrs(DV.sDis[t] <= (DV.Xstr/plant.duration) for t in range(self.opt_data['nT']))
            self.Model.addConstrs(DV.sLev[t] <= DV.Xstr for t in range(1,self.opt_data['nT']))
            self.Model.addConstr(DV.sLev[0] == DV.Xstr/2)

        self.Model.addConstrs(DV.sDis[t] <= DV.sLev[t-1]-DV.sLev[t-1]*plant.decay_eff for t in range(1, self.opt_data['nT']))
        self.Model.addConstrs(DV.sLev[t] == DV.sLev[t-1]-DV.sLev[t-1]*plant.decay_eff + plant.eff_round *
                        DV.sCh[t]-DV.sDis[t] for t in range(1, self.opt_data['nT']))

        if not self.Setting.dispatch:
            # c6: RE farm size
            for i in range(self.opt_data['RE_nlocations'][0]):
                total_area=gp.LinExpr()
                for r, replant in enumerate(self.Setting.RE_plant_types):
                    pindex = self.Setting.plant_types.index(replant)
                    plant = self.dat.Plants[pindex]
                    self.Model.addConstr(DV.Xr[r][i] <= plant.area*self.Setting.area_buffer*plant.density, name=f'c_REsize_{i}_{replant}')
                    total_area.addTerms(1/plant.density/self.Setting.area_buffer, DV.Xr[r][i])
                self.Model.addConstr(total_area <= plant.area, name=f'c_REsize_{i}_total')
            
            # c7: allowed lost load
            self.Model.addConstrs(DV.LL[t] <= self.Setting.lost_load_thres for t in range(self.opt_data['nT']))

        # c8: actual load: smaller than generations
        for pindex in range(self.opt_data['nPlt']):
            for t in range(self.opt_data['nT']):
                self.Model.addConstr(DV.load[pindex, t] <= DV.prod[pindex, t], name=f'c_load_{t}_{pindex}')
        
    def get_DV_vals(self):
        if not self.Setting.dispatch:
            DV_val.Xg_val = self.Model.getAttr('x', DV.Xg)
            DV_val.gas_inv_cost_val = self.Model.getAttr('x', DV.gas_inv_cost)
            DV_val.total_cost_val = DV.total_cost.X 
            DV_val.RE_cost_val = self.Model.getAttr('x', DV.RE_cost)
            DV_val.Xr_val = list()
            for r in range(self.opt_data['nREPlt']):
                DV_val.Xr_val.append(self.Model.getAttr('x', DV.Xr[r]))
        else:
            DV_val.total_cost_val = DV.total_cost.X + sum(self.opt_data['RE_cost']) + sum(self.opt_data['gas_inv_cost']) + self.opt_data['strg_inv_cost']
        
        DV_val.prod_val = self.Model.getAttr('x', DV.prod)
        DV_val.load_val = self.Model.getAttr('x', DV.load)
        DV_val.var_cost_val = self.Model.getAttr('x', DV.var_cost)
        DV_val.fuel_cost_val = self.Model.getAttr('x', DV.fuel_cost)

    def print_results(self,stime):
        prod_value = np.zeros((self.opt_data['nPlt'], self.opt_data['nT']))
        load_value = np.zeros((self.opt_data['nPlt'], self.opt_data['nT']))
        for p in range(self.opt_data['nPlt']):
            for t in range(self.opt_data['nT']):
                prod_value[p, t] = DV_val.prod_val[p, t]
                load_value[p, t] = DV_val.load_val[p, t]
        total_load = self.dat.demand.sum()/self.Setting.num_y

        # locations
        if not self.Setting.dispatch:
            for r, replant in enumerate(self.Setting.RE_plant_types):
                pindex=self.Setting.plant_types.index(replant)
                RE_Xr_val = [DV.Xr[r][n].X for n in range(self.opt_data['RE_nlocations'][r])]
                RE_Xr_val = np.array(RE_Xr_val)
                lat = self.dat.Plants[pindex].lat[RE_Xr_val != 0]
                lon = self.dat.Plants[pindex].lon[RE_Xr_val != 0]
                capacity = RE_Xr_val[RE_Xr_val != 0]
                dfw = pd.DataFrame(
                    data={'lat': lat, 'lon': lon, 'capacity': capacity})
                csvfilename_suffix = '%s/sub%dyrs_ens%d_demand_%s_%s_cc_%d_landr_%d_%s_%s' % (self.Setting.odir,self.Setting.num_y,
                                                                                               self.Setting.ens_id,
                                                                                self.Setting.demandsce,self.Setting.iso,
                                                                                self.Setting.UB_dispatchable_cap['CCGT']*100,self.Setting.landr,
                                                                                self.Setting.weather_model,self.Setting.scenario)
                dfw.to_csv(csvfilename_suffix+f'_{replant}_locations.csv', encoding='utf-8', index=False)
        
        #load file
        DF_prod = pd.DataFrame()
        for pindex, plant in enumerate(self.Setting.plant_types):
            DF_prod[f'prod_{plant}'] = prod_value[pindex, :].squeeze()
            DF_prod[f'load_{plant}'] = load_value[pindex, :].squeeze()
        DF_prod['sLev'] = self.Model.getAttr('x', DV.sLev)
        DF_prod['sCh'] = self.Model.getAttr('x', DV.sCh)
        DF_prod['sDis'] = self.Model.getAttr('x', DV.sDis)
        DF_prod['demand'] = self.dat.demand
        DF_prod['LL'] = self.Model.getAttr('x',DV.LL)
        DF_prod['dat'] = self.dat.dates
        if self.Setting.dispatch:
            DF_prod.to_csv("%s/%s_test_%s_%.3f_Load.csv" % (self.Setting.odir,self.Setting.suffix, self.Setting.scenario,
                                                            self.Setting.capacity_factor),encoding='utf-8', index=False)
        else:
            DF_prod.to_csv(csvfilename_suffix+'_Load.csv', encoding='utf-8', index=False)

        # general results
        DF = pd.DataFrame(data={'num_periods': self.opt_data['nT'],
                        'Sol_time': np.round(time.time()-stime),
                        'weather_model': self.Setting.weather_model,
                        'landres_allowed': self.Setting.landr,
                        'demand_data':self.Setting.demandsce,
                        'region': self.Setting.iso,
                        'total_cost': DV_val.total_cost_val,
                        'prod_strg': DF_prod['sDis'].sum()/self.opt_data['nYear'],
                        'LL_cost': DV.lost_load_cost.X,
                        'total_LL':DF_prod['LL'].sum()/self.opt_data['nYear'],
                        'total_Curtail': (prod_value.sum()-load_value.sum())/self.opt_data['nYear'],
                        'total_load': total_load,
                        'LCOE': DV_val.total_cost_val/(total_load*1000)}, index=[0])

        if self.Setting.dispatch:
            DF['total_cost_strg']=self.opt_data['strg_inv_cost']
            DF['capacity_strg']=self.opt_data['Xstr']
            DF['investment_sce']=self.Setting.invest_scenario
            DF['inv_num_y']=self.Setting.invest_num_y
            DF['inv_ens_id']=self.Setting.invest_ens_id
            DF['test_sce']=self.Setting.scenario
            DF['capacity_factor']=self.Setting.capacity_factor
        else:
            DF['total_cost_strg']=DV.strg_inv_cost.X
            DF['capacity_strg']=DV.Xstr.X
            DF['investment_sce']=self.Setting.scenario
            DF['year_list']=str(self.Setting.sub_year_list)
            DF['inv_num_y']=self.Setting.num_y
            DF['inv_ens_id']=self.Setting.ens_id

        for d, dplant in enumerate(self.Setting.gas_plant_types):
            pindex = self.Setting.plant_types.index(dplant)
            if self.Setting.dispatch:
                DF[f'inv_cost_{dplant}'] =  self.opt_data['gas_inv_cost'][d]
                DF[f'capacity_{dplant}'] = self.dat.Plants[pindex].capacity
            else:
                DF[f'inv_cost_{dplant}'] = DV_val.gas_inv_cost_val[d]
                DF[f'capacity_{dplant}'] = DV_val.Xg_val[d]
            DF[f'fuel_cost_{dplant}'] = DV_val.fuel_cost_val[d]
            DF[f'var_cost_{dplant}'] = DV_val.var_cost_val[d]
            DF[f'prod_{dplant}'] = prod_value[pindex, :].sum()/self.opt_data['nYear']
            DF[f'load_{dplant}'] = load_value[pindex, :].sum()/self.opt_data['nYear']
            DF[f'total_cost_{dplant}'] = DF[f'inv_cost_{dplant}'] + \
                DF[f'fuel_cost_{dplant}']+DF[f'var_cost_{dplant}']
            DF[f'upper_bound_{dplant}'] = self.Setting.UB_dispatchable_cap[dplant]

        for r,replant in enumerate(self.Setting.RE_plant_types):
            pindex = self.Setting.plant_types.index(replant)
            if self.Setting.dispatch:
                DF[f'total_cost_{replant}'] = self.opt_data['RE_cost'][r]
                #DF[f'capacity_{replant}'] = self.dat.Plants[pindex].capacity
            else:
                DF[f'total_cost_{replant}'] = DV_val.RE_cost_val[r]
                DF[f'capacity_{replant}'] = DV_val.Xr_val[r].sum()
            DF[f'prod_{replant}'] = prod_value[pindex, :].sum()/self.opt_data['nYear']
            DF[f'load_{replant}'] = load_value[pindex, :].sum()/self.opt_data['nYear']
            DF[f'cell_size_{replant}'] = self.Setting.RE_cell_size

        dfvfile=f'{self.Setting.odir}/{self.Setting.weather_model}_{self.Setting.iso}_General_Results.csv'
        if not os.path.exists(dfvfile):
            DF.to_csv(dfvfile, mode='w', header=True, index=False)
        else:
            DF.to_csv(dfvfile, mode='a', header=False, index=False)

    def solve(self):
        self.define_DVs()
        self.add_obj_func()
        self.add_constraints()
        self.Model.setParam('OutputFlag', 1)
        self.Model.setParam('MIPGap', self.Setting.solver_gap)
        self.Model.setParam('Timelimit', self.Setting.wall_clock_time_lim)
        self.Model.setParam('Presolve', 2)  # -1 to 2
        self.Model.optimize()
    
    def run_and_save_results(self):
        stime = time.time()
        self.solve()
        self.get_DV_vals()
        self.print_results(stime)
        total_time = np.round(time.time() - stime)
        print(f"Optimization completed in {total_time} seconds.")
        self.Model.reset()
        del (self.Model)

