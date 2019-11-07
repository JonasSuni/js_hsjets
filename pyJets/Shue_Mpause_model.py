import numpy as np

def Shue_Mpause_model(Dp_SW,Bz_SW,theta,r_MP):

    # Magnetopause model from Shue et al. (1998)

    # Magnetopause flaring parameter:
    alpha_MP = (0.58 - 0.007*Bz_SW)*(1. + 0.024*np.log(Dp_SW))
    # Position of the magnetopause subsolar point
    Rssp_MP = (10.22 + 1.29*np.tanh(0.184*(Bz_SW + 8.14)))*np.power(
                                                       Dp_SW,(-1./6.6))

    # Magnetopause geocentric distance in the direction defined by
    # the polar angle theta
    # (Shue et al. model is symmetrical about the Sun-Earth line)

    for i in range(0,len(theta)):
        r_MP[i] = Rssp_MP*np.power((2./(1. + np.cos(theta[i]))),alpha_MP)

    return r_MP[0]
