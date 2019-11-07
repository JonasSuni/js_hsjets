import numpy as np
from Merka_BS_model import BS_distance_Merka2005
from Shue_Mpause_model import Shue_Mpause_model

medium_blue = '#006DDB'
crimson = '#920000'
violet = '#B66DFF'
dark_blue = '#490092'

def BS_xy():
    theta = np.arange(-60,60,0.5)
    R_bs = np.zeros_like(theta)
    for a in theta:
        index = np.where(theta==a)[0][0]
        R_bs[index] = BS_distance_Merka2005(a,0,6e6,400e3,8,[])

    x_bs = R_bs*np.cos(np.deg2rad(theta))
    y_bs = R_bs*np.sin(np.deg2rad(theta))

    return [x_bs,y_bs]


def MP_xy():
    theta = np.arange(-60,60,0.5)
    R_mp = np.zeros_like(theta)
    for a in theta:
        index = np.where(theta==a)[0][0]
        R_mp[index] = Shue_Mpause_model(400e3*400e3*6e6*1e-9,0.0,[a],[0])

    x_mp = R_mp*np.cos(np.deg2rad(theta))
    y_mp = R_mp*np.sin(np.deg2rad(theta))

    return [x_mp,y_mp]
