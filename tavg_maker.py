import numpy as np
import pytools as pt
import linecache as lc
import pandas as pd
import os

m_p = 1.672621898e-27

def avg_maker(runid,start,stop):

    outputdir = "/wrk/sunijona/DONOTREMOVE/tavg/"+runid+"/"

    # make outputdir if it doesn't already exist
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    pdyn_arr = np.array([])
    rho_arr = np.array([])

    for n in xrange(start-180,stop+1+180):

        # find correct file for current time step
        if runid == "AED":
            bulkname = "bulk.old."+str(n).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(n).zfill(7)+".vlsv"

        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
        cellids = vlsvobj.read_variable("CellID")

        if vlsvobj.check_variable("rho"):
            rho = vlsvobj.read_variable("rho")[cellids.argsort()]
            v = vlsvobj.read_variable("v")[cellids.argsort()]
        else:
            rho = vlsvobj.read_variable("proton/rho")[cellids.argsort()]
            v = vlsvobj.read_variable("proton/V")[cellids.argsort()]

        pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

        if pdyn_arr.size == 0:
            rho_arr = np.append(rho_arr,rho)
            pdyn_arr = np.append(pdyn_arr,pdyn)
        else:
            rho_arr = np.vstack((rho_arr,rho))
            pdyn_arr = np.vstack((pdyn_arr,pdyn))

    tpdyn_arr = np.array([])
    trho_arr = np.array([])

    for itr in xrange(start,stop+1):

        trho = (np.sum(rho_arr[itr-start:itr+361-start],axis=0)-rho_arr[itr+180-start])/360
        tpdyn = (np.sum(pdyn_arr[itr-start:itr+361-start],axis=0)-pdyn_arr[itr+180-start])/360

        if tpdyn_arr.size == 0:
            trho_arr = np.append(trho_arr,trho)
            tpdyn_arr = np.append(tpdyn_arr,tpdyn)
        else:
            trho_arr = np.vstack((trho_arr,trho))
            tpdyn_arr = np.vstack((tpdyn_arr,tpdyn))

    for ind in xrange(len(tpdyn_arr)):

        np.savetxt(outputdir+str(ind+start)+"_rho.tavg",trho_arr[ind])
        np.savetxt(outputdir+str(ind+start)+"_pdyn.tavg",tpdyn_arr[ind])

    return None

def TP_maker(runid,start,stop):

    outputdir = "/wrk/sunijona/DONOTREMOVE/TP/"+runid+"/"

    # make outputdir if it doesn't already exist
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    for n in xrange(start,stop+1):

        # find correct file for current time step
        if runid == "AED":
            bulkname = "bulk.old."+str(n).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(n).zfill(7)+".vlsv"

        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
        cellids = vlsvobj.read_variable("CellID")

        if vlsvobj.check_variable("rho"):
            TPar = vlsvobj.read_variable("TParallel")
            TPerp = vlsvobj.read_variable("TPerpendicular")
        else:
            TPar = vlsvobj.read_variable("proton/TParallel")
            TPerp = vlsvobj.read_variable("proton/TPerpendicular")

        TPar = TPar[cellids.argsort()]
        TPerp = TPerp[cellids.argsort()]

        np.savetxt(outputdir+str(n)+".tpar",TPar)
        np.savetxt(outputdir+str(n)+".tperp",TPerp)

    return None