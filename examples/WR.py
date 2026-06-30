import os
import pandas as pd
import matplotlib.pyplot as plt

from werim import (
    Dataset, Region, Month,
    Clim, Anom, 
    Preprocess,
    WeatherRegimes,
)

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

ds = Dataset("slp_daily_1940_2024_era5data1g.nc", folder=DATASETS_DIR).open()
ds.slice(Region(
    lat0=20, latf=64, lon0=-40, lonf=39,
    month0=Month.MAR, monthf=Month.MAY,
    year0=1940, yearf=2024,
))

if False:
    Clim(ds, "map").plot(cmap="bwr", levels=30, figsize=(8, 6))

    anom = Anom(ds, "map", group_season=False)
    anom.plot(timestamp="1940-03-01", cmap="bwr", levels=30, figsize=(8, 6))

nm = 10
alpha = 0.1
k = 4

# y = Preprocess(ds, detrend=True, group_season=False)
# y.save("y_", DATA_DIR)
y = Preprocess.load("pre_", DATA_DIR)
if False:
    y.plot(timestamp="1940-03-01", figsize=(8, 6))

# wr = WeatherRegimes(y, nm=nm, alpha=alpha, k=k)
# wr.save("wr_", DATA_DIR)
wr = WeatherRegimes.load("wr_", DATA_DIR, ds=y)

wr.plot_PCs()
wr.plot_EOFs()  # EOFs
wr.plot_clusters()

z = Preprocess(
    Dataset("slp_daily_1940_2024_era5data1g.nc", folder=DATASETS_DIR)
    .open()
    .slice(Region(
        lat0=20, latf=64, lon0=-40, lonf=39,
        month0=Month.MAR, monthf=Month.MAY,
        year0=1940, yearf=2024,
     )),
    group_season=False,
)
if False:
    z.plot(timestamp="1940-03-01", figsize=(8, 6))

plt.show()
