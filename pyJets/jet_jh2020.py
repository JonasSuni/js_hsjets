import pytools as pt
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import plot_contours as pc
import jet_analyser as ja
import jet_io as jio
import jet_aux as jx

def get_transient_xseries(runid,jetid,transient="jet"):

    if type(jetid) is not str:
        jetid = str(jetid).zfill(5)

    try:
        props = jio.PropReader(jetid,runid,transient=transient)
    except:
        return 1

    time_arr = props.read("time")
    x_arr = props.read("x_mean")
    y_arr = props.read("y_mean")
    pd_arr = props.read("pd_avg")

    bs_dist_arr = jx.bs_dist_markus(runid,time_arr,x_arr,y_arr)

    return (bs_dist_arr,pd_arr)
