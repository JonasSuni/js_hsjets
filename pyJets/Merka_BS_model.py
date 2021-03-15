import numpy as np
import numpy.linalg

#**************************************************;
#  Bow shock distance (Merka et al. (2005) model)  ;
#**************************************************;
# Created : 2017-05-09
# Lucile Turc - University of Helsinki
#**************************************************;
# This function calculates the distance of the BS
# from the Earth in the direction defined by the
# angles (theta, phi), using the BS model
# developped by Merka et al. (2005).
#**************************************************;
#**************************************************;
# The equation of the bow shock surface is :
# A1*X^2 + A2*Y^2 + A3*Z^2 + 2*A4*X*Y + 2*A7*X +
# 2*A8*Y + A10 = 0
# where the Ai are functions of the upstream Mach
# number.
#**************************************************;
#**************************************************;
# We can show that X is solution of a quadratic
# equation and that Y and Z depend linearly on X.
# First we calculate X using the discriminant of the
# quadratic equation : AX^2 + BX + C = 0
#**************************************************;

#***************************************************************;
#--------------------------- Inputs ----------------------------;
#***************************************************************;
# - theta, phi : two angles to define the direction in which
# we want to calculate the bow shock position
# - theta is measured from the z axis and phi from the x axis,
# and lies in the x-y plane (cos theta = z/r and cos phi =
# x/sqrt(x^2 + y^2)
# - n : solar wind density (/cc)
# - V : solar wind speed (km/s)
# - Ma : solar wind Alfven Mach number
#***************************************************************;
#--------------------------- Outputs ---------------------------;
#***************************************************************;
# R_BS : position of the bow shock in the direction defined by
# (theta,phi)
#***************************************************************;

def BS_distance_Merka2005(theta,phi,n,V,Ma,R_BS):


    #******************************;
    #   Merka05 model parameters
    #******************************;
    #Scaling factor of the coordinate system
    FAC=np.power((n/7.0)*(V/457.5)*(V/457.5),1./6.)

    #*******************************************************;
    # Fitting parameters for Alfven Mach number dependence
    #*******************************************************;
    B11 =0.0063
    B12 =-0.0098
    B31 =0.8351
    B32 =0.0102
    B41 =-0.02980
    B42 =0.0040
    B71 =16.39
    B72 =0.2087
    B73 =108.3
    B81 =-0.9241
    B82 =0.0721
    B101 =-444.0
    B102 =-2.935
    B103 =-1930.0

    #Fitting parameters of the shock surface
    A1=B11+B12*Ma
    A2=1.0                 # As in Peredo
    A3=B31+B32*Ma
    A4=B41+B42*Ma
    A5=0.0                 # Assume north-south symmetry -> A5 = A6 = A9 = 0
    A6=0.0
    A7=B71+B72*Ma+B73/((Ma-1.)*(Ma-1.))
    A8=B81+B82*Ma
    A9=0.0
    A10=B101+B102*Ma+B103/((Ma-1.)*(Ma-1.))


    #************************************************************************#
    # Transform into a scaled GPE coordinate system (4 degree aberrated GSE) #
    #************************************************************************#
    phi_GPE = phi + 4.*np.pi/180. #rad
    theta_GPE = theta

    tan_phi = np.tan(phi_GPE)
    tan_theta = np.tan(theta_GPE)

    if np.isfinite(Ma):

        A = A1 + A2*tan_phi*tan_phi + A3*(1.+tan_phi*tan_phi)/(tan_theta*tan_theta) + 2.*A4*tan_phi
        B = 2*A7 + 2*A8*tan_phi
        C = A10
        delta = B*B - 4*C*A

        if(delta >= 0) :
            if(np.abs(phi_GPE) < np.pi/2.):
                Xbs_gpe = (-B + np.sqrt(delta))/(2.*A)
            else:
                Xbs_gpe = (-B - np.sqrt(delta))/(2.*A)

            Ybs_gpe = Xbs_gpe*tan_phi
            Zbs_gpe = np.abs(Xbs_gpe)*np.sqrt(1.+tan_phi*tan_phi)/tan_theta

            # Rescaling the GPE coordinates using the scaling factor FAC depending on upstream solar wind parameters:

            Xbs_gpe = Xbs_gpe/FAC
            Ybs_gpe = Ybs_gpe/FAC
            Zbs_gpe = Zbs_gpe/FAC
            R_BS.append(np.sqrt(Xbs_gpe*Xbs_gpe + Ybs_gpe*Ybs_gpe + Zbs_gpe*Zbs_gpe))

        else:
            print('Delta < 0 : no real solution for the reference surface position in Merka et al. model')
            R_BS.append(np.nan)

    else:
        R_BS.append(np.nan)

    return R_BS[0]



def Merka_BS_normal(Xbs,Ybs,Zbs,n,V,Ma,BS_normal):

    #*****************************************
    # Calculation of the bow shock normal at
    # the point (Xbs,Ybs,Zbs)
    #-----------------------------------------
    # We use the gradient of f(x,y,z) which
    # describes the bow shock shape in GPE
    # coordinates to calculate the shock
    # normal vector.
    #*****************************************

    #******************************;
    #   Merka05 model parameters
    #******************************;
    #Scaling factor of the coordinate system
    FAC=np.power((n/7.0)*(V/457.5)*(V/457.5),1./6.)

    #*******************************************************;
    # Fitting parameters for Alfven Mach number dependence
    #*******************************************************;

    B11 =0.0063
    B12 =-0.0098
    B31 =0.8351
    B32 =0.0102
    B41 =-0.02980
    B42 =0.0040
    B71 =16.39
    B72 =0.2087
    B73 =108.3
    B81 =-0.9241
    B82 =0.0721
    B101 =-444.0
    B102 =-2.935
    B103 =-1930.0

    #Fitting parameters of the shock surface
    A1=B11+B12*Ma
    A2=1.0              # As in Peredo
    A3=B31+B32*Ma
    A4=B41+B42*Ma
    A5=0.0              # Assume north-south symmetry -> A5 = A6 = A9 = 0
    A6=0.0
    A7=B71+B72*Ma+B73/((Ma-1.)*(Ma-1.))
    A8=B81+B82*Ma
    A9=0.0
    A10=B101+B102*Ma+B103/((Ma-1.)*(Ma-1.))

    phi_GSE2GPE = -4.*np.pi/180.

    vec_GSE = np.array([Xbs,Ybs,Zbs])

    rotation_matrix_GPE2GSE  = np.array([[np.cos(phi_GSE2GPE),-np.sin(phi_GSE2GPE),0],
                                         [np.sin(phi_GSE2GPE),np.cos(phi_GSE2GPE),0],
                                         [0,0,1]])
    rotation_matrix_GSE2GPE = np.linalg.inv(rotation_matrix_GPE2GSE)

    #print(np.dot(rotation_matrix_GSE2GPE,np.array([1,0,0])))
    vec_GPE = np.dot(rotation_matrix_GSE2GPE,vec_GSE)

    Xbs_GPE = vec_GPE[0]
    Ybs_GPE = vec_GPE[1]
    Zbs_GPE = vec_GPE[2]

    dfx = 2*A1*Xbs_GPE + 2*A4*Ybs_GPE + 2*A7/FAC
    dfy = 2*A2*Ybs_GPE + 2*A4*Xbs_GPE + 2*A8/FAC
    dfz = 2*A3*Zbs_GPE

    norm_grad = np.sqrt(dfx*dfx+dfy*dfy+dfz*dfz)

    normal_gpe = np.zeros(3)
    normal_gpe[0] = dfx/norm_grad
    normal_gpe[1] = dfy/norm_grad
    normal_gpe[2] = dfz/norm_grad


    normal_GSE = np.dot(rotation_matrix_GPE2GSE,normal_gpe)

    BS_normal.append(normal_GSE)

    return BS_normal[0]
