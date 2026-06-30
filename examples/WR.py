import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import cluster

from werim import (
    Dataset, Region, Month,
    Clim, Anom, 
    Preprocess,
    WeatherRegimes,
)
from werim.principal_components import PrincipalComponents

DATASETS_DIR = "/Users/Shared/datasets"
DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    os.path.splitext(os.path.basename(__file__))[0],
    "data"
)
PLOTS_DIR = os.path.join(
    os.path.dirname(__file__),
    os.path.splitext(os.path.basename(__file__))[0],
    "plots",
)

dsy = Dataset("slp_daily_1940_2024_era5data1g.nc", folder=DATASETS_DIR).open()
dsy.slice(Region(
    lat0=20, latf=64, lon0=-40, lonf=39,
    month0=Month.MAR, monthf=Month.MAY,
    year0=1940, yearf=2024,
))

if False:
    Clim(dsy, "map").plot(cmap="bwr", levels=30, figsize=(8, 6))

    anom = Anom(dsy, "map", group_season=False)
    anom.plot(timestamp="1940-03-01", cmap="bwr", levels=30, figsize=(8, 6))

nm = 10
alpha = 0.1

# y = Preprocess(dsy, detrend=True, group_season=False)
# y.save("y_", DATA_DIR)
y = Preprocess.load("pre_", DATA_DIR)
if False:
    y.plot(timestamp="1940-03-01", figsize=(8, 6))

# pc = PrincipalComponents(y, nm=nm, alpha=alpha)
# pc.save("pc_", DATA_DIR)
pc = PrincipalComponents.load("pc_", DATA_DIR, ds=y)

if True:
    pc.plot()
    pc.plot_PCs(nm=4)

dsz = Dataset("Spain02_v5.0_DD_010reg_aa3d_pr.nc", folder=DATASETS_DIR).open().slice(Region(
    lat0=36.2, latf=43, lon0=-2, lonf=4,
    month0=Month.MAR, monthf=Month.MAY,
    year0=1971, yearf=2015,
))

z = Clim(dsz, "ts", group_season=False)
if False:
    z.plot(figsize=(8, 5))

if False:
    pc.plot_IC(
        extremes=z.data.values > np.percentile(z.data.values, 95),
        time_mask=((pc.ds.time.dt.year >= dsz.region.year0) & (pc.ds.time.dt.year <= dsz.region.yearf)).values,
    )

k = 4
wr = WeatherRegimes(pc=pc, k=k)
wr.save("wr_", DATA_DIR)
# wr = WeatherRegimes.load("wr_", DATA_DIR, pc=pc)
if False:
    wr.plot()
    wr.plot_PCs()

wr.plot(plot_composed=True)

plt.show()
