import sys
import os
import math

from typing import (
    Optional as Op,
    Any, Tuple, Literal, List,
    Sequence, Union,
    cast,
)

import xarray as xr
import numpy as np
import numpy.typing as npt
import cartopy.crs as ccrs

import scipy.linalg as scipy_linalg
import scipy.sparse.linalg as sparse_linalg
import scipy.cluster as scipy_cluster

import matplotlib.pyplot as plt
from matplotlib.figure import Figure as Fig
from matplotlib.axes import Axes as Ax
from matplotlib import gridspec, ticker

import spy4cast as sp
import spy4cast.meteo as sp_meteo
import spy4cast.spy4cast as sp_sp
import spy4cast._procedure as sp_pr

FArray = npt.NDArray[np.float32]
BArray = npt.NDArray[np.bool]

