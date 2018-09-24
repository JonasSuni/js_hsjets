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
import jet_io as jio

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

def make_slams_mask(filenumber,runid,boxre=[6,18,-8,6]):
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

    B = vlsvreader.read_variable("B")[np.argsort(origid)]
    Bmag = np.linalg.norm(B,axis=-1)

    # dynamic pressure
    pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

    sw_pars = ja.sw_par_dict()[runid]
    rho_sw = sw_pars[0]
    v_sw = sw_pars[1]
    pdyn_sw = m_p*rho_sw*(v_sw**2)
    if runid in ["AEA","AEC"]:
        B_sw = 10.0e-9
    else:
        B_sw = 5.0e-9

    # make custom SLAMS mask
    slams = np.ma.masked_greater(Bmag,1.25*B_sw)
    slams.mask[pdyn < 1.25*pdyn_sw] = False
    slams.mask[rho > 3*rho_sw] = False

    # discard unmasked cellids
    masked_ci = np.ma.array(sorigid,mask=~slams.mask).compressed()

    if not os.path.exists("SLAMS/masks/"+runid+"/"):
        try:
            os.makedirs("SLAMS/masks/"+runid+"/")
        except OSError:
            pass

    # if boundaries have been set, discard cellids outside boundaries
    if not not boxre:
        masked_ci = np.intersect1d(masked_ci,ja.restrict_area(vlsvreader,boxre[0:2],boxre[2:4]))
        np.savetxt("SLAMS/masks/"+runid+"/"+str(filenumber)+".mask",masked_ci)
        return masked_ci
    else:
        np.savetxt("SLAMS/masks/"+runid+"/"+str(filenumber)+".mask",masked_ci)
        return masked_ci

def sort_slams(vlsvobj,cells,min_size=0,max_size=3000,neighborhood_reach=[1,1]):
    # sort masked cells into events based on proximity in X,Y-space

    # initialise list of events and current event
    events = []
    curr_event = np.array([],dtype=int)

    for cell in cells:

        # check if cell already in list of events
        bl_a = False
        for event in events:
            if cell in event:
                bl_a = True
        if bl_a:
            continue

        # number of times to search for more neighbors
        it_range = xrange(200)

        # initialise current event
        curr_event = np.array([cell])

        for n in it_range:

            curr_event_size = curr_event.size

            # find neighbors within the confines of the mask
            curr_event = np.unique(np.append(curr_event,np.intersect1d(cells,ja.get_neighbors(vlsvobj,curr_event,neighborhood_reach))))

            # exit loop if all valid neighbors found
            if curr_event_size == curr_event.size:
                break

        # cast cellids of current event to int and append to list of events
        curr_event = curr_event.astype(int)
        events.append(curr_event)

    # remove events smaller than the minimum size and larger than maximum size
    events_culled = [slams for slams in events if slams.size >= min_size and slams.size <= max_size]

    return events_culled

def slams_maker(runid,start,stop,boxre=[6,18,-8,6],maskfile=False):

    outputdir = "/homeappl/home/sunijona/SLAMS/events/"+runid+"/"

    # make outputdir if it doesn't already exist
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    for file_nr in xrange(start,stop+1):

        # find correct file based on file number and run id
        if runid in ["AEC","AEF","BEA","BEB"]:
            bulkpath = "/proj/vlasov/2D/"+runid+"/"
        elif runid == "AEA":
            bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
        else:
            bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

        if runid == "AED":
            bulkname = "bulk.old."+str(file_nr).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(file_nr).zfill(7)+".vlsv"

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file "+str(file_nr)+" not found, continuing")
            continue

        # open vlsv file for reading
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

        # create mask
        if maskfile:
            msk = np.loadtxt("SLAMS/masks/"+runid+"/"+str(file_nr)+".mask").astype(int)
        else:
            msk = make_slams_mask(file_nr,runid,boxre)

        print(len(msk))
        print("Current file number is " + str(file_nr))

        # sort jets
        slams = sort_slams(vlsvobj,msk,10,1000,[2,2])

        # erase contents of output file
        open(outputdir+str(file_nr)+".events","w").close()

        # open output file
        fileobj = open(outputdir+str(file_nr)+".events","a")

        # write jets to outputfile
        for slam in slams:

            fileobj.write(",".join(map(str,slam))+"\n")

        fileobj.close()

    return None

def eventfile_read(runid,filenr):
    # Read array of arrays of cellids from file

    outputlist = []

    ef = open("/homeappl/home/sunijona/SLAMS/events/"+runid+"/"+str(filenr)+".events","r")
    contents = ef.read().strip("\n")
    if contents == "":
        return []
    lines = contents.split("\n")

    for line in lines:

        outputlist.append(map(int,line.split(",")))

    return outputlist

def track_slams(runid,start,stop,threshold=0.5):

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    if runid == "AED":
        bulkname = "bulk.old."+str(start).zfill(7)+".vlsv"
    else:
        bulkname = "bulk."+str(start).zfill(7)+".vlsv"

    if bulkname not in os.listdir(bulkpath):
        print("Bulk file "+str(start)+" not found, exiting")
        return 1

    # Create outputdir if it doesn't already exist
    if not os.path.exists("/homeappl/home/sunijona/SLAMS/slams/"+runid):
        try:
            os.makedirs("/homeappl/home/sunijona/SLAMS/slams/"+runid)
        except OSError:
            pass

    # Get solar wind parameters
    sw_pars = ja.sw_par_dict()[runid]
    rho_sw = sw_pars[0]
    v_sw = sw_pars[1]

    # Open file, get Cell IDs and sort them
    vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
    sorigid = vlsvobj.read_variable("CellID")
    sorigid = sorigid[sorigid.argsort()]
    
    # Find bow shock cells and area of one cell
    bs_cells = ja.bow_shock_finder(vlsvobj,rho_sw,v_sw)
    dA = ja.get_cell_area(vlsvobj)

    # Read initial event files
    events_old = eventfile_read(runid,start)
    events = eventfile_read(runid,start+1)

    # do nothing
    bs_events = []
    for old_event in events_old:
        bs_events.append(old_event)

    # Initialise list of jet objects
    jetobj_list = []
    dead_jetobj_list = []

    # Initialise unique ID counter
    counter = 1

    # Print current time
    print("t = "+str(float(start+1)/2)+"s")

    # Look for slams
    for event in events:

        for bs_event in bs_events:

            if np.intersect1d(bs_event,event).size > threshold*len(event):

                # Create unique ID
                curr_id = str(counter).zfill(5)

                # Create new jet object
                jetobj_list.append(jio.Jet(curr_id,runid,float(start)/2))

                # Append current events to jet object properties
                jetobj_list[-1].cellids.append(bs_event)
                jetobj_list[-1].cellids.append(event)
                jetobj_list[-1].times.append(float(start+1)/2)

                # Iterate counter
                counter += 1

                break

    # Track jets
    for n in xrange(start+2,stop+1):

        for jetobj in jetobj_list:
            if float(n)/2 - jetobj.times[-1] > 5:
                dead_jetobj_list.append(jetobj)
                jetobj_list.remove(jetobj)

        # Print  current time
        print("t = "+str(float(n)/2)+"s")

        # Find correct bulkname
        if runid == "AED":
            bulkname = "bulk.old."+str(n).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(n).zfill(7)+".vlsv"

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file "+str(n)+" not found, continuing")
            events = []
            continue

        # Open bulkfile and get bow shock cells
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
        bs_cells = ja.bow_shock_finder(vlsvobj,rho_sw,v_sw)

        # List of old events
        bs_events = []
        for old_event in events:
            bs_events.append(old_event)

        # Initialise flags for finding splintering jets
        flags = []

        # Read event file for current time step
        events = eventfile_read(runid,n)
        events.sort(key=len)
        events = events[::-1]

        # Iniatilise list of cells currently being tracked
        curr_jet_temp_list = []

        # Update existing jets
        for event in events:

            for jetobj in jetobj_list:

                if jetobj.ID in flags:
                    
                    if np.intersect1d(jetobj.cellids[-2],event).size > threshold*len(event):

                        curr_id = str(counter).zfill(5)

                        # Create new jet
                        jetobj_new = jio.Jet(curr_id,runid,float(n)/2)
                        jetobj_new.cellids.append(event)
                        jetobj_list.append(jetobj_new)
                        curr_jet_temp_list.append(event)

                        #Clone jet
                        #jetobj_new = copy.deepcopy(jetobj)
                        #jetobj_new.ID = str(counter).zfill(5)
                        #print("Cloned jet to new one with ID "+jetobj_new.ID)
                        #jetobj_new.cellids = jetobj_new.cellids[:-1]
                        #jetobj_new.cellids.append(event)
                        #jetobj_list.append(jetobj_new)
                        #curr_jet_temp_list.append(event)

                        # Iterate counter
                        counter += 1

                        break

                else:

                    if np.intersect1d(jetobj.cellids[-1],event).size > threshold*len(event):

                        # Append event to jet object properties
                        jetobj.cellids.append(event)
                        jetobj.times.append(float(n)/2)
                        print("Updated jet with ID "+jetobj.ID)

                        # Flag jet object
                        flags.append(jetobj.ID)
                        curr_jet_temp_list.append(event)

                        break

        # Look for new jets at bow shock
        for event in events:

            if event not in curr_jet_temp_list:

                for bs_event in bs_events:

                    if np.intersect1d(bs_event,event).size > threshold*len(event):

                        # Create unique ID
                        curr_id = str(counter).zfill(5)

                        # Create new jet object
                        jetobj_list.append(jio.Jet(curr_id,runid,float(n-1)/2))

                        # Append current events to jet object properties
                        jetobj_list[-1].cellids.append(bs_event)
                        jetobj_list[-1].cellids.append(event)
                        jetobj_list[-1].times.append(float(n)/2)

                        # Iterate counter
                        counter += 1

                        break

    jetobj_list = jetobj_list + dead_jetobj_list

    for jetobj in jetobj_list:

        # Write jet object cellids and times to files
        jetfile = open("/homeappl/home/sunijona/SLAMS/slams/"+jetobj.runid+"/"+str(start)+"."+jetobj.ID+".slams","w")
        timefile = open("/homeappl/home/sunijona/SLAMS/slams/"+jetobj.runid+"/"+str(start)+"."+jetobj.ID+".times","w")

        jetfile.write(jetobj.return_cellid_string())
        timefile.write(jetobj.return_time_string())

        jetfile.close()
        timefile.close()

    return None

def timefile_read(runid,filenr,key):
    # Read array of times from file

    tf = open("/homeappl/home/sunijona/SLAMS/slams/"+runid+"/"+str(filenr)+"."+key+".times","r")
    contents = tf.read().split("\n")
    tf.close()

    return map(float,contents)

def jetfile_read(runid,filenr,key):
    # Read array of cellids from file

    outputlist = []

    jf = open("/homeappl/home/sunijona/SLAMS/slams/"+runid+"/"+str(filenr)+"."+key+".slams","r")
    contents = jf.read()
    lines = contents.split("\n")

    for line in lines:

        outputlist.append(map(int,line.split(",")))

    return outputlist

def propfile_write(runid,filenr,key,props):
    # Write jet properties to file

    open("/homeappl/home/sunijona/SLAMS/slams/"+runid+"/"+str(filenr)+"."+key+".props","w").close()
    pf = open("/homeappl/home/sunijona/SLAMS/slams/"+runid+"/"+str(filenr)+"."+key+".props","a")
    pf.write("time [s],x_mean [R_e],y_mean [R_e],z_mean [R_e],A [R_e^2],Nr_cells,r_mean [R_e],theta_mean [deg],phi_mean [deg],size_rad [R_e],size_tan [R_e],x_max [R_e],y_max [R_e],z_max [R_e],n_avg [1/cm^3],n_med [1/cm^3],n_max [1/cm^3],v_avg [km/s],v_med [km/s],v_max [km/s],B_avg [nT],B_med [nT],B_max [nT],T_avg [MK],T_med [MK],T_max [MK],TPar_avg [MK],TPar_med [MK],TPar_max [MK],TPerp_avg [MK],TPerp_med [MK],TPerp_max [MK],beta_avg,beta_med,beta_max,x_min [R_e],rho_vmax [1/cm^3],b_vmax"+"\n")
    pf.write("\n".join([",".join(map(str,line)) for line in props]))
    pf.close()
    print("Wrote to /homeappl/home/sunijona/SLAMS/slams/"+runid+"/"+str(filenr)+"."+key+".props")

def plotmake_script(runid,start,stop,vmax=1.5,boxre=[6,16,-8,6]):
    # Create plots of the dynamic pressure with contours of jets as well as their geometric centers

    if not os.path.exists("SLAMS/contours/"+runid):
        try:
            os.makedirs("SLAMS/contours/"+runid)
        except OSError:
            pass
    # Find names of property files
    filenames = os.listdir("SLAMS/slams/"+runid)
    prop_fns = []
    for filename in filenames:
        if ".props" in filename:
            prop_fns.append(filename)
    prop_fns.sort()

    # Create dictionaries of jet positions with their times as keys
    tpos_dict_list = []
    for fname in prop_fns:
        props = pd.read_csv("/homeappl/home/sunijona/SLAMS/slams/"+runid+"/"+fname,index_col=False).as_matrix()
        t=props[:,0]
        X=props[:,1]
        Y=props[:,2]
        xmax=props[:,11]
        ymax=props[:,12]
        tpos_dict_list.append(dict(zip(t,np.array([X,Y,xmax,ymax]).T)))

    # Find names of event files
    filenames = os.listdir("SLAMS/events/"+runid)
    nrs = [int(s[:-7]) for s in filenames]
    filenames=np.array(filenames)[np.argsort(nrs)].tolist()

    # Create list of arrays of cellids to use as contour mask
    cells_list = []
    for filename in filenames:

        fileobj = open("SLAMS/slams/"+runid+"/"+filename,"r")
        contents = fileobj.read()
        cells = map(int,contents.replace("\n",",").split(",")[:-1])
        cells_list.append(cells)

    fullmask_list = []
    for itr2 in xrange(start,stop+1):

        try:
            fullmask = np.loadtxt("SLAMS/masks/"+runid+"/"+str(itr2)+".mask").astype(int)
        except IOError:
            fullmask = np.array([])
        fullmask_list.append(fullmask)

    # Find correct bulk path
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    for itr in xrange(start,stop+1):

        # Find positions of all jets for current time step
        x_list = []
        y_list = []
        xmax_list = []
        ymax_list = []
        for tpos_dict in tpos_dict_list:
            if float(itr)/2 in tpos_dict:
                x_list.append(tpos_dict[float(itr)/2][0])
                y_list.append(tpos_dict[float(itr)/2][1])
                xmax_list.append(tpos_dict[float(itr)/2][2])
                ymax_list.append(tpos_dict[float(itr)/2][3])

        bulkname = "bulk."+str(itr).zfill(7)+".vlsv"

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file "+str(itr)+" not found, continuing")
            continue

        # Create plot
        pt.plot.plot_colormap(filename=bulkpath+bulkname,outputdir="SLAMS/contours/"+runid+"/",step=itr,run=runid,usesci=0,lin=1,boxre=boxre,vmin=0,vmax=vmax,colormap="parula",cbtitle="",external=pms_ext,expression=pc.expr_pdyn,pass_vars=["rho","v","CellID"],ext_pars=[x_list,y_list,cells_list[itr-start],fullmask_list[itr-start],xmax_list,ymax_list])

    return None

def pms_ext(ax,XmeshXY,YmeshXY,extmaps,ext_pars):

    rho,v,cellids = extmaps

    x_list,y_list,cells,fullmask,xmax_list,ymax_list = ext_pars

    # Create mask
    msk = np.in1d(cellids,cells).astype(int)
    msk = np.reshape(msk,rho.shape)

    fullmsk = np.in1d(cellids,fullmask).astype(int)
    fullmsk = np.reshape(fullmsk,rho.shape)

    # Draw contours
    fullcont = ax.contour(XmeshXY,YmeshXY,fullmsk,[0.5],linewidths=1.0,colors="magenta")
    cont = ax.contour(XmeshXY,YmeshXY,msk,[0.5],linewidths=1.0,colors="black")

    # Plot jet positions
    ax.plot(x_list,y_list,"o",color="red",markersize=4)
    ax.plot(xmax_list,ymax_list,"o",color="white",markersize=4)

def calc_slams_properties(runid,start,jetid,tp_files=False):

    if str(start)+"."+jetid+".slams" not in os.listdir("SLAMS/slams/"+runid):
        print("SLAMS with ID "+jetid+" does not exist, exiting.")
        return 1

    # Read jet cellids and times
    jet_list = jetfile_read(runid,start,jetid)
    time_list = timefile_read(runid,start,jetid)

    # Discard jet if it's very short-lived
    if len(time_list) < 5:
        print("Jet not sufficiently long-lived, exiting.")
        return 1

    # Discard jet if it has large gaps in the times
    dt = (np.pad(np.array(time_list),(0,1),"constant")-np.pad(np.array(time_list),(1,0),"constant"))[1:-1]
    if max(dt) > 5:
        print("Jet not sufficiently continuous, exiting.")
        return 1

    # Find correct bulk path
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    # Convert times to file numbers
    nr_list = [int(t*2) for t in time_list]

    # Initialise property array
    prop_arr = np.array([])

    for n in xrange(len(nr_list)):

        curr_list = jet_list[n]
        curr_list.sort()

        # Find correct file name
        if runid == "AED":
            bulkname = "bulk.old."+str(nr_list[n]).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(nr_list[n]).zfill(7)+".vlsv"

        # Open VLSV file
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

        origid = vlsvobj.read_variable("CellID")
        sorigid = origid[origid.argsort()]

        # read variables
        if vlsvobj.check_variable("X"):
            X = vlsvobj.read_variable("X")[origid.argsort()]
            Y = vlsvobj.read_variable("Y")[origid.argsort()]
            Z = vlsvobj.read_variable("Z")[origid.argsort()]
            X,Y,Z = ja.ci2vars_nofile([X,Y,Z],sorigid,curr_list)
        else:
            X,Y,Z = ja.ci2vars_nofile(ja.xyz_reconstruct(vlsvobj),sorigid,curr_list)

        # Calculate area of one cell
        if n == 0 and vlsvobj.check_variable("DX"):
            dA = vlsvobj.read_variable("DX")[0]*vlsvobj.read_variable("DY")[0]
        elif n == 0 and not vlsvobj.check_variable("DX"):
            dA = ja.get_cell_area(vlsvobj)

        # If file has more than one population, choose proton population
        var_list = ["rho","v","B","Temperature","CellID","beta","TParallel","TPerpendicular"]
        var_list_alt = ["proton/rho","proton/V","B","proton/Temperature","CellID","proton/beta","proton/TParallel","proton/TPerpendicular"]
        if not vlsvobj.check_variable("rho"):
            var_list = var_list_alt

        # If temperature files exist, read parallel and perpendicular temperature from those instead
        if tp_files:

            var_list = var_list[:-2]

            rho,v,B,T,cellids,beta = ja.read_mult_vars(vlsvobj,var_list,cells=-1)
            cellids = cellids[cellids.argsort()]
            TParallel = jio.tpar_reader(runid,nr_list[n],cellids,curr_list)
            TPerpendicular = jio.tperp_reader(runid,nr_list[n],cellids,curr_list)
            
            rho = rho[origid.argsort()]
            v = v[origid.argsort()]
            B = B[origid.argsort()]
            T = T[origid.argsort()]
            beta = beta[origid.argsort()]

            rho,v,B,T,beta = ja.ci2vars_nofile([rho,v,B,T,beta],sorigid,curr_list)

        else:

            rho,v,B,T,cellids,beta,TParallel,TPerpendicular = ja.read_mult_vars(vlsvobj,var_list,cells=-1)
            rho = rho[origid.argsort()]
            v = v[origid.argsort()]
            B = B[origid.argsort()]
            T = T[origid.argsort()]
            beta = beta[origid.argsort()]
            TParallel = TParallel[origid.argsort()]
            TPerpendicular = TPerpendicular[origid.argsort()]

            rho,v,B,T,beta,TParallel,TPerpendicular = ja.ci2vars_nofile([rho,v,B,T,beta,TParallel,TPerpendicular],sorigid,curr_list)

        # Q: Why are we doing this?
        #cellids = cellids[cellids.argsort()]

        # Scale variables
        rho /= 1.0e+6
        v /= 1.0e+3
        B /= 1.0e-9
        T /= 1.0e+6
        TParallel /= 1.0e+6
        TPerpendicular /= 1.0e+6

        # Calculate magnitudes of v and B
        vmag = np.linalg.norm(v,axis=-1)
        Bmag = np.linalg.norm(B,axis=-1)

        # Calculate means, medians and maximums for rho,vmag,Bmag,T,TParallel,TPerpendicular,beta
        n_avg = np.nanmean(rho)
        n_med = np.median(rho)
        n_max = np.max(rho)

        v_avg = np.nanmean(vmag)
        v_med = np.median(vmag)
        v_max = np.max(vmag)

        B_avg = np.nanmean(Bmag)
        B_med = np.median(Bmag)
        B_max = np.max(Bmag)

        T_avg = np.nanmean(T)
        T_med = np.median(T)
        T_max = np.max(T)

        TPar_avg = np.nanmean(TParallel)
        TPar_med = np.median(TParallel)
        TPar_max = np.max(TParallel)

        TPerp_avg = np.nanmean(TPerpendicular)
        TPerp_med = np.median(TPerpendicular)
        TPerp_max = np.max(TPerpendicular)

        beta_avg = np.nanmean(beta)
        beta_med = np.median(beta)
        beta_max = np.max(beta)

        # Convert X,Y,Z to spherical coordinates
        r = np.linalg.norm(np.array([X,Y,Z]),axis=0)
        theta = np.rad2deg(np.arccos(Z/r))
        phi = np.rad2deg(np.arctan(Y/X))

        # calculate geometric center of jet
        r_mean = np.mean(r)/r_e
        theta_mean = np.mean(theta)
        phi_mean = np.mean(phi)

        # Geometric center of jet in cartesian coordinates
        x_mean = r_mean*np.sin(np.deg2rad(theta_mean))*np.cos(np.deg2rad(phi_mean))
        y_mean = r_mean*np.sin(np.deg2rad(theta_mean))*np.sin(np.deg2rad(phi_mean))
        z_mean = r_mean*np.cos(np.deg2rad(theta_mean))

        # Position of maximum velocity in cartesian coordinates
        x_max = X[vmag==max(vmag)][0]/r_e
        y_max = Y[vmag==max(vmag)][0]/r_e
        z_max = Z[vmag==max(vmag)][0]/r_e

        # Minimum x and density at maximum velocity
        x_min = min(X)/r_e
        rho_vmax = rho[vmag==max(vmag)]
        b_vmax = beta[vmag==max(vmag)]

        #r_max = np.linalg.norm(np.array([x_mean,y_mean,z_mean]))
        #theta_max = np.rad2deg(np.arccos(z_mean/r_mean))
        #phi_max = np.rad2deg(np.arctan(y_mean/x_mean))

        # calculate jet size
        A = dA*len(curr_list)/(r_e**2)
        Nr_cells = len(curr_list)

        # calculate linear sizes of jet
        size_rad = (max(r)-min(r))/r_e
        size_tan = A/size_rad

        # current time
        time = time_list[n]

        ''' 
        0: time [s],
        1: x_mean [R_e],        2: y_mean [R_e],        3: z_mean [R_e],
        4: A [R_e^2],           5: Nr_cells,
        6: r_mean [R_e],        7: theta_mean [deg],    8: phi_mean [deg],
        9: size_rad [R_e],      10: size_tan [R_e],
        11: x_max [R_e],        12: y_max [R_e],        13: z_max [R_e],
        14: n_avg [1/cm^3],     15: n_med [1/cm^3],     16: n_max [1/cm^3],
        17: v_avg [km/s],       18: v_med [km/s],       19: v_max [km/s],
        20: B_avg [nT],         21: B_med [nT],         22: B_max [nT],
        23: T_avg [MK],         24: T_med [MK],         25: T_max [MK],
        26: TPar_avg [MK],      27: TPar_med [MK],      28: TPar_max [MK],
        29: TPerp_avg [MK],     30: TPerp_med [MK],     31: TPerp_max [MK],
        32: beta_avg,           33: beta_med,           34: beta_max
        35: x_min [R_e],        36: rho_vmax [1/cm^3],  37: b_vmax
        '''

        # Create temporary property array
        temp_arr = [time,x_mean,y_mean,z_mean,A,Nr_cells,r_mean,theta_mean,phi_mean,size_rad,size_tan,x_max,y_max,z_max,n_avg,n_med,n_max,v_avg,v_med,v_max,B_avg,B_med,B_max,T_avg,T_med,T_max,TPar_avg,TPar_med,TPar_max,TPerp_avg,TPerp_med,TPerp_max,beta_avg,beta_med,beta_max,x_min,rho_vmax,b_vmax]

        # append properties to property array
        prop_arr = np.append(prop_arr,np.array(temp_arr))

    # reshape property array
    prop_arr = np.reshape(prop_arr,(len(nr_list),len(temp_arr)))

    # write property array to file
    propfile_write(runid,start,jetid,prop_arr)

    return prop_arr