# Accounting for climate projections in energy system modeling

[☁️ View on calkit.io](https://calkit.io/petebachant/qiu-2025-energy-modeling)

To reproduce the results, execute:

```sh
calkit run
```

## SitingModel

### Data Input

#### 1. Capacity Factor

- Download: [Hourly Wind and Solar Generation Profiles at 1/8th Degree Resolution from PNNL](https://zenodo.org/records/10214348) [Historical; rcp85hotter; rcp45hotter]
- Data Path:
- Extract data for each ISO [step1_cf_remap.py](step1_cf_remap.py)

#### 2. Demand Data

- Model: [TELL: Total ELectricity Loads Model](https://github.com/IMMM-SFA/tell)
- Code: [demand-tell.ipynb](step2_demand-tell.ipynb)
  - _Trained on ISO electricity consumption data from 2016-2018 and evaluated using data from 2019_
- TGW data avaeraged for each Balancing Authority:  Stored [/orcd/nese/mhowland/001/lyqiu/GODEEP/TGW/Meteorology/ISOmean](./TGW/Meteorology/ISOmean); [Dowload ](https://data.msdlive.org/records/cnsy6-0y610)
- Detrend data: [demand-tell.ipynb](step2_demand-tell.ipynb) - "detrend data"

### Siting Model: Spatially Explicit Renewable Siting Model (Investment Decisions)

The model minimizes the sum of annualized investment costs and hourly
operational costs,
subject to constraints on hourly demand balance,
hourly VRE availability,
spatially disaggregated VRE deployment potential,
firm generator (natural gas) availability,
storage energy balance.
The model outputs VRE, storage, and non-VRE deployment and their operations.
RESPO is implemented with the Gurobi optimizer in Python.

[Model](optimization): [investment](optimization/investment/) mode and [dispatch](optimization/dispatch/) mode

#### Main.py

`Setting.dispatch = False` # True: run dispatch optimization; False: run investment optimization

`Setting.dispatch_unlimited   = False` # limit the dispatchable capacity of CCGT to renewable target or not
