import numpy as np
import pytools as pt
import scipy
import pandas as pd
import jet_scripts as js
import jet_analyser as ja
import jetfile_make as jfm
import os
import jet_scripts as js
import copy
import matplotlib.pyplot as plt
import plot_contours as pc
import scipy.constants as sc

m_p = 1.672621898e-27
r_e = 6.371e+6

'''
This file should contain all functions and scripts related to finding, sorting, tracking and statistically analysing SLAMS.
'''

def visual_slams_finder(runid,filenumber,boxre=[6,18,-8,6],vmax=1.5,plaschke=1.0,sw=1.0):

    if runid in ["AEA","AEC"]:
        B_sw = 10.0e-9
    else:
        B_sw = 5.0e-9

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    if runid == "AED":
        bulkname = "bulk.old."+str(filenumber).zfill(7)+".vlsv"
    else:
        bulkname = "bulk."+str(filenumber).zfill(7)+".vlsv"

    if runid == "BFD":
        pass_vars=["proton/rho","proton/V","CellID","B"]
    else:
        pass_vars=["rho","v","CellID","B"]

    pt.plot.plot_colormap(filename=bulkpath+bulkname,draw=1,usesci=0,lin=1,cbtitle="",boxre=boxre,colormap="parula",vmin=0,vmax=vmax,expression=pc.expr_pdyn,external=ext_slams,pass_vars=pass_vars,ext_pars=[runid,filenumber,plaschke,sw,B_sw])

def ext_slams(ax,XmeshPass,YmeshPass,extmaps,ext_pars):

    rho = extmaps[0]
    v = extmaps[1]
    cellids = extmaps[2]
    B = extmaps[3]

    vx = v[:,:,0]
    vmag = np.linalg.norm(v,axis=-1)

    Bmag = np.linalg.norm(B,axis=-1)

    shp = rho.shape

    runid = ext_pars[0]
    filenumber = ext_pars[1]

    sw_pars = ja.sw_par_dict()[runid]

    tpdynavg = np.loadtxt("/wrk/sunijona/DONOTREMOVE/tavg/"+runid+"/"+str(filenumber)+"_pdyn.tavg")

    vmag = vmag.flatten()
    vx = vx.flatten()
    rho = rho.flatten()
    cellids = cellids.flatten()
    Bmag = Bmag.flatten()

    pdyn = m_p*rho*(vmag**2)
    pdyn_x = m_p*rho*(vx**2)

    spdyn = pdyn[np.argsort(cellids)]
    spdyn_x = pdyn_x[np.argsort(cellids)]
    srho = rho[np.argsort(cellids)]
    sBmag = Bmag[np.argsort(cellids)]

    pdyn_sw = m_p*sw_pars[0]*(sw_pars[1]**2)

    slams = np.ma.masked_greater(sBmag,ext_pars[3]*ext_pars[4])
    slams.mask[spdyn < ext_pars[2]*pdyn_sw] = False
    slams.mask[srho > 3*sw_pars[0]] = False
    slams.fill_value = 0
    slams[slams.mask == False] = 1

    contour = ax.contour(XmeshPass,YmeshPass,np.reshape(slams.filled(),shp),[0.5],linewidths=1.0, colors="black")

    return None

def make_slams_mask(filenumber,runid,boxre=[6,18,-8,6],avgfile=False):
    # finds cellids of cells that fulfill the specified criterion and the specified
    # X,Y-limits

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    if runid == "AED":
        bulkname = "bulk.old."+str(filenumber).zfill(7)+".vlsv"
    else:
        bulkname = "bulk."+str(filenumber).zfill(7)+".vlsv"

    if bulkname not in os.listdir(bulkpath):
        print("Bulk file "+str(filenumber)+" not found, exiting.")
        return 1

    # open vlsv file for reading
    vlsvreader = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

    origid = vlsvreader.read_variable("CellID")
    sorigid = origid[np.argsort(origid)]

    # if file has separate populations, read proton population
    if type(vlsvreader.read_variable("rho")) is not np.ndarray:
        rho = vlsvreader.read_variable("proton/rho")[np.argsort(origid)]
        v = vlsvreader.read_variable("proton/V")[np.argsort(origid)]
    else:
        rho = vlsvreader.read_variable("rho")[np.argsort(origid)]
        v = vlsvreader.read_variable("v")[np.argsort(origid)]

    # x-directional dynamic pressure
    spdynx = m_p*rho*(v[:,0]**2)

    # dynamic pressure
    pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

    sw_pars = sw_par_dict()[runid]
    rho_sw = sw_pars[0]
    v_sw = sw_pars[1]
    pdyn_sw = m_p*rho_sw*(v_sw**2)

    npdynx = spdynx/pdyn_sw
    nrho = rho/rho_sw

    # initialise time average of dynamic pressure
    tpdynavg = np.zeros(pdyn.shape)

    # range of timesteps to calculate average of
    timerange = xrange(filenumber-180,filenumber+180+1)

    missing_file_counter = 0

    for n_t in timerange:

        if avgfile:
            continue

        # exclude the main timestep
        if n_t == filenumber:
            continue

        # find correct file path for current time step
        if runid == "AED":
            tfile_name = "bulk.old."+str(n_t).zfill(7)+".vlsv"
        else:
            tfile_name = "bulk."+str(n_t).zfill(7)+".vlsv"

        if tfile_name not in os.listdir(bulkpath):
            missing_file_counter += 1
            print("Bulk file "+str(n_t)+" not found, continuing")
            continue

        # open file for current time step
        f = pt.vlsvfile.VlsvReader(bulkpath+tfile_name)
        
        # if file has separate populations, read proton population
        if type(f.read_variable("rho")) is not np.ndarray:
            trho = f.read_variable("proton/rho")
            tv = f.read_variable("proton/V")
        else:
            trho = f.read_variable("rho")
            tv = f.read_variable("v")

        # read cellids for current time step
        cellids = f.read_variable("CellID")
        
        # dynamic pressure for current time step
        tpdyn = m_p*trho*(np.linalg.norm(tv,axis=-1)**2)

        # sort dynamic pressures
        otpdyn = tpdyn[cellids.argsort()]
        
        tpdynavg = np.add(tpdynavg,otpdyn)

    # calculate time average of dynamic pressure
    tpdynavg /= len(timerange)-1-missing_file_counter

    # prevent divide by zero errors
    tpdynavg[tpdynavg == 0.0] = 1.0e-27

    if avgfile:
        tpdynavg = np.loadtxt("/wrk/sunijona/DONOTREMOVE/tavg/"+runid+"/"+str(filenumber)+"_pdyn.tavg")

    # ratio of dynamic pressure to its time average
    tapdyn = pdyn/tpdynavg

    # make custom SLAMS mask
    jet = np.ma.masked_greater(npdynx,0.25)
    jet.mask[nrho < 3.5] = False
    jet.mask[tapdyn > 2] = True

    # discard unmasked cellids
    masked_ci = np.ma.array(sorigid,mask=~jet.mask).compressed()

    if not os.path.exists("SLAMS/masks/"+runid+"/"):
        try:
            os.makedirs("SLAMS/masks/"+runid+"/")
        except OSError:
            pass

    # if boundaries have been set, discard cellids outside boundaries
    if not not boxre:
        masked_ci = np.intersect1d(masked_ci,restrict_area(vlsvreader,boxre[0:2],boxre[2:4]))
        np.savetxt("SLAMS/masks/"+runid+"/"+str(filenumber)+".mask",masked_ci)
        return masked_ci
    else:
        np.savetxt("SLAMS/masks/"+runid+"/"+str(filenumber)+".mask",masked_ci)
        return masked_ci