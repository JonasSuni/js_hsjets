import numpy as np
import pytools as pt
import scipy
import pandas as pd
import jet_analyser as ja
import os
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

    sw_pars = ja.sw_par_dict(runid)
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
        masked_ci = np.intersect1d(masked_ci,ja.restrict_area(vlsvreader,boxre))
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
        slams = sort_slams(vlsvobj,msk,10,4500,[2,2])

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
    sw_pars = ja.sw_par_dict(runid)
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
    pf.write("time [s],x_mean [R_e],y_mean [R_e],z_mean [R_e],A [R_e^2],Nr_cells,r_mean [R_e],theta_mean [deg],phi_mean [deg],size_rad [R_e],size_tan [R_e],x_max [R_e],y_max [R_e],z_max [R_e],n_avg [1/cm^3],n_med [1/cm^3],n_max [1/cm^3],v_avg [km/s],v_med [km/s],v_max [km/s],B_avg [nT],B_med [nT],B_max [nT],T_avg [MK],T_med [MK],T_max [MK],TPar_avg [MK],TPar_med [MK],TPar_max [MK],TPerp_avg [MK],TPerp_med [MK],TPerp_max [MK],beta_avg,beta_med,beta_max,x_min [R_e],rho_vmax [1/cm^3],b_vmax,pd_avg [nPa],pd_med [nPa],pd_max [nPa]"+"\n")
    pf.write("\n".join([",".join(map(str,line)) for line in props]))
    pf.close()
    print("Wrote to /homeappl/home/sunijona/SLAMS/slams/"+runid+"/"+str(filenr)+"."+key+".props")

def plotmake_script_BFD(start,stop,runid="BFD",vmax=1.5,boxre=[4,20,-10,4]):

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

    xmean_dict = dict()
    ymean_dict = dict()
    xmax_dict = dict()
    ymax_dict = dict()

    for fname in prop_fns:
        jet_id = fname[4:-6]
        props = PropReader(ID=jet_id,runid=runid)
        time = props.read("time")
        x_mean = props.read("x_mean")
        y_mean = props.read("y_mean")
        z_mean = props.read("z_mean")
        x_vmax = props.read("x_vmax")
        y_vmax = props.read("y_vmax")
        z_vmax = props.read("z_vmax")
        if runid in ["BFD"]:
            y_mean = z_mean
            y_vmax = z_vmax
        for itr in xrange(time.size):
            if time[itr] not in xmean_dict:
                xmean_dict[time[itr]] = [x_mean[itr]]
                ymean_dict[time[itr]] = [y_mean[itr]]
                xmax_dict[time[itr]] = [x_vmax[itr]]
                ymax_dict[time[itr]] = [y_vmax[itr]]
            else:
                xmean_dict[time[itr]].append(x_mean[itr])
                ymean_dict[time[itr]].append(y_mean[itr])
                xmax_dict[time[itr]].append(x_vmax[itr])
                ymax_dict[time[itr]].append(y_vmax[itr])

    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    for itr2 in xrange(start,stop+1):

        t = float(itr2)/2

        bulkname = "bulk."+str(itr2).zfill(7)+".vlsv"

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file "+str(itr2)+" not found, continuing")
            continue

        if runid == "BFD" and itr2 == 961:
            print("Broken file!")
            continue

        if runid in ["BFD"]:
            pass_vars = ["proton/rho","proton/V","CellID"]
        else:
            pass_vars = ["rho","v","CellID"]

        try:
            fullmask = np.loadtxt("SLAMS/masks/"+runid+"/"+str(itr2)+".mask").astype(int)
        except IOError:
            fullmask = np.array([])

        try:
            fileobj = open("SLAMS/events/"+runid+"/"+str(itr2)+".events","r")
            contents = fileobj.read()
            cells = map(int,contents.replace("\n",",").split(",")[:-1])
        except IOError:
            cells = []

        # Create plot
        pt.plot.plot_colormap(filename=bulkpath+bulkname,outputdir="SLAMS/contours/"+runid+"/",step=itr2,run=runid,usesci=0,lin=1,boxre=boxre,vmin=0,vmax=vmax,colormap="parula",cbtitle="",external=pms_ext,expression=pc.expr_pdyn,pass_vars=pass_vars,ext_pars=[xmean_dict[t],ymean_dict[t],cells,fullmask,xmax_dict[t],ymax_dict[t]])

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
    dt = np.ediff1d(time_list)
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

        pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

        # Scale variables
        rho /= 1.0e+6
        v /= 1.0e+3
        B /= 1.0e-9
        pdyn /= 1.0e-9
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

        pd_avg = np.nanmean(pdyn)
        pd_med = np.median(pdyn)
        pd_max = np.max(pdyn)

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
        38: pd_avg [nPa],       39: pd_med [nPa],       40: pd_max [nPa]
        '''

        # Create temporary property array
        temp_arr = [time,x_mean,y_mean,z_mean,A,Nr_cells,r_mean,theta_mean,phi_mean,size_rad,size_tan,x_max,y_max,z_max,n_avg,n_med,n_max,v_avg,v_med,v_max,B_avg,B_med,B_max,T_avg,T_med,T_max,TPar_avg,TPar_med,TPar_max,TPerp_avg,TPerp_med,TPerp_max,beta_avg,beta_med,beta_max,x_min,rho_vmax,b_vmax,pd_avg,pd_med,pd_max]

        # append properties to property array
        prop_arr = np.append(prop_arr,np.array(temp_arr))

    # reshape property array
    prop_arr = np.reshape(prop_arr,(len(nr_list),len(temp_arr)))

    # write property array to file
    propfile_write(runid,start,jetid,prop_arr)

    return prop_arr

def slams_vs_hist(runids,var,time_thresh=10):

    # Get all filenames in folder
    filenames_list = []
    for runid in runids:
        filenames_list.append(os.listdir("SLAMS/slams/"+runid))

    # Filter for property files
    file_list_list = []
    for filenames in filenames_list:
        file_list_list.append([filename for filename in filenames if ".props" in filename])

    # Cutoff dictionary for eliminating false positives
    run_cutoff_dict = dict(zip(["ABA","ABC","AEA","AEC","BFD"],[6,6,6,6,6]))

    # Different colors for different runs
    run_colors_dict = dict(zip([runids[0],runids[1]],["red","blue"]))

    # Dictionary for mapping input variables to parameters
    key_list = ["duration",
    "size_rad","size_tan","size_ratio",
    "pdyn_vmax","pd_avg","pd_med","pd_max",
    "n_max","n_avg","n_med","rho_vmax",
    "v_max","v_avg","v_med",
    "B_max","B_avg","B_med",
    "beta_max","beta_avg","beta_med","b_vmax",
    "T_avg","T_med","T_max",
    "TPar_avg","TPar_med","TPar_max",
    "TPerp_avg","TPerp_med","TPerp_max",
    "A",
    "death_distance"]

    n_list = list(xrange(len(key_list)))
    var_dict = dict(zip(key_list,n_list))

    # Initialise var list
    var_list = [[],[]]

    val_dict = dict(zip(runids,var_list))

    # Append variable values to var lists
    for n in xrange(len(runids)):
        for fname in file_list_list[n]:
            props = PropReader("",runids[n],fname=fname)
            if props.read("time")[-1]-props.read("time")[0] > time_thresh and max(props.read("r_mean")) > run_cutoff_dict[runids[n]]:
                if var == "duration":
                    val_dict[runids[n]].append(props.read("time")[-1]-props.read("time")[0])
                elif var == "size_ratio":
                    val_dict[runids[n]].append(props.read_at_amax("size_rad")/props.read_at_amax("size_tan"))
                elif var in ["n_max","n_avg","n_med","rho_vmax"]:
                    val_dict[runids[n]].append(props.read_at_amax(var)/props.sw_pars[0])
                elif var in ["v_max","v_avg","v_med"]:
                    val_dict[runids[n]].append(props.read_at_amax(var)/props.sw_pars[1])
                elif var in ["B_max","B_avg","B_med"]:
                    val_dict[runids[n]].append(props.read_at_amax(var)/props.sw_pars[2])
                elif var in ["beta_max","beta_avg","beta_med","b_vmax"]:
                    val_dict[runids[n]].append(props.read_at_amax(var)/props.sw_pars[4])
                elif var == "pdyn_vmax":
                    val_dict[runids[n]].append(m_p*(1.0e+6)*props.read_at_amax("rho_vmax")*((props.read_at_amax("v_max")*1.0e+3)**2)/(props.sw_pars[3]*1.0e-9))
                elif var in ["pd_avg","pd_med","pd_max"]:
                    val_dict[runids[n]].append(props.read_at_amax(var)/props.sw_pars[3])
                elif var == "death_distance":
                    val_dict[runids[n]].append(np.linalg.norm([props.read("x_vmax")[-1],props.read("y_vmax")[-1],props.read("z_vmax")[-1]]))
                else:
                    val_dict[runids[n]].append(props.read_at_amax(var))

    # Labels for figure
    label_list = ["Duration [s]",
    "Radial size [R$_{e}$]","Tangential size [R$_{e}$]","Radial size/Tangential size",
    "P$_{dyn,vmax}$ [P$_{dyn,sw}$]","P$_{dyn,avg}$ [P$_{dyn,sw}$]","P$_{dyn,med}$ [P$_{dyn,sw}$]","P$_{dyn,max}$ [P$_{dyn,sw}$]",
    "n$_{max}$ [n$_{sw}$]","n$_{avg}$ [n$_{sw}$]","n$_{med}$ [n$_{sw}$]","n$_{v,max}$ [n$_{sw}$]",
    "v$_{max}$ [v$_{sw}$]","v$_{avg}$ [v$_{sw}$]","v$_{med}$ [v$_{sw}$]",
    "B$_{max}$ [B$_{IMF}$]","B$_{avg}$ [B$_{IMF}$]","B$_{med}$ [B$_{IMF}$]",
    "$\\beta _{max}$ [$\\beta _{sw}$]","$\\beta _{avg}$ [$\\beta _{sw}$]","$\\beta _{med}$ [$\\beta _{sw}$]","$\\beta _{v,max}$ [$\\beta _{sw}$]",
    "T$_{avg}$ [MK]","T$_{med}$ [MK]","T$_{max}$ [MK]",
    "T$_{Parallel,avg}$ [MK]","T$_{Parallel,med}$ [MK]","T$_{Parallel,max}$ [MK]",
    "T$_{Perpendicular,avg}$ [MK]","T$_{Perpendicular,med}$ [MK]","T$_{Perpendicular,max}$ [MK]",
    "Area [R$_{e}^{2}$]",
    "r$_{v,max}$ at time of death [R$_{e}$]"]

    # X limits and bin widths for figure
    xmin_list=[10,
    0,0,0,
    1.25,1.25,1.25,1.25,
    1,1,1,1,
    0.6,0.6,0.6,
    1.25,1.25,1.25,
    1,1,1,1,
    0,0,0,
    0,0,0,
    0,0,0,
    0,
    8]

    xmax_list=[60,
    3,1,7,
    3,3,3,3,
    3,3,3,3,
    1.2,1.2,1.2,
    6,6,6,
    1000,1000,1000,1000,
    25,25,25,
    25,25,25,
    25,25,25,
    1.5,
    18]

    step_list = [2,
    0.25,0.05,0.2,
    0.05,0.05,0.05,0.05,
    0.1,0.1,0.1,0.1,
    0.05,0.05,0.05,
    0.25,0.25,0.25,
    100,100,100,100,
    1,1,1,
    1,1,1,
    1,1,1,
    0.05,
    0.5]

    # Create figure
    plt.ioff()
    #plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(label_list[var_dict[var]],fontsize=20)
    ax.set_ylabel("Fraction of SLAMS",fontsize=20)
    ax.set_xlim(xmin_list[var_dict[var]],xmax_list[var_dict[var]])
    ax.set_ylim(0,1)
    weights = [[1/float(len(val_dict[runids[n]]))]*len(val_dict[runids[n]]) for n in xrange(len(runids))] # Normalise by total number of jets

    # Logarithmic scale for plasma beta
    if var in ["beta_max","beta_avg","beta_med","b_vmax"]:
        bins = np.arange(0,3.25,0.25)
        bins = 10**bins
        plt.xscale("log")
        ax.set_xlim(1,xmax_list[var_dict[var]])
        
        #for n in xrange(len(runids)):
        #    hist = ax.hist(var_list[n],weights=weights[n],bins=bins,color=run_colors_dict[runids[n]],alpha=0.5,label=runids[n])
        
        hist = ax.hist([val_dict[runids[0]],val_dict[runids[1]]],weights=weights,bins=bins,color=[run_colors_dict[runids[0]],run_colors_dict[runids[1]]],label=runids)

    else:
        bins = np.arange(xmin_list[var_dict[var]],xmax_list[var_dict[var]]+step_list[var_dict[var]],step_list[var_dict[var]])
        if var == "death_distance":
            ax.set_xlim(8,xmax_list[var_dict[var]])
        
            bins = np.arange(8,xmax_list[var_dict[var]]+step_list[var_dict[var]],step_list[var_dict[var]])
        #for n in xrange(len(runids)):
        #    hist = ax.hist(var_list[n],bins=bins,weights=weights[n],color=run_colors_dict[runids[n]],alpha=0.5,label=runids[n])

        hist = ax.hist([val_dict[runids[0]],val_dict[runids[1]]],bins=bins,weights=weights,color=[run_colors_dict[runids[0]],run_colors_dict[runids[1]]],label=runids)

    for n in xrange(len(runids)):
        ax.axvline(np.median(val_dict[runids[n]]), linestyle="dashed", linewidth=2, color=run_colors_dict[runids[n]])

    plt.title(",".join(runids),fontsize=20)
    plt.legend()
    plt.tight_layout()

    # Create output directory
    if not os.path.exists("SLAMS/figures/histograms/"+"_vs_".join(runids)+"/"):
        try:
            os.makedirs("SLAMS/figures/histograms/"+"_vs_".join(runids)+"/")
        except OSError:
            pass

    # Save figure
    fig.savefig("SLAMS/figures/histograms/"+"_vs_".join(runids)+"/"+var+"_"+str(time_thresh)+".png")
    print("SLAMS/figures/histograms/"+"_vs_".join(runids)+"/"+var+"_"+str(time_thresh)+".png")

    plt.close(fig)

    return None

def slams_all_hist(runids,var,time_thresh=10):
    # Creates histogram specified var

    # Get all filenames in folder
    filenames_list = []
    for runid in runids:
        filenames_list.append(os.listdir("SLAMS/slams/"+runid))

    # Filter for property files
    file_list_list = []
    for filenames in filenames_list:
        file_list_list.append([filename for filename in filenames if ".props" in filename])

    # Cutoff values for elimination of false positives
    run_cutoff_dict = dict(zip(["ABA","ABC","AEA","AEC","BFD"],[6,6,6,6,6]))

    # Dictionary for mapping input variables to parameters
    key_list = ["duration",
    "size_rad","size_tan","size_ratio",
    "pdyn_vmax","pd_avg","pd_med","pd_max",
    "n_max","n_avg","n_med","rho_vmax",
    "v_max","v_avg","v_med",
    "B_max","B_avg","B_med",
    "beta_max","beta_avg","beta_med","b_vmax",
    "T_avg","T_med","T_max",
    "TPar_avg","TPar_med","TPar_max",
    "TPerp_avg","TPerp_med","TPerp_max",
    "A",
    "death_distance"]

    n_list = list(xrange(len(key_list)))
    var_dict = dict(zip(key_list,n_list))

    # Initialise var list
    var_list = []

    # Append variable values to var list
    for n in xrange(len(runids)):
        for fname in file_list_list[n]:
            props = PropReader("",runids[n],fname=fname)
            if props.read("time")[-1]-props.read("time")[0] > time_thresh and max(props.read("r_mean")) > run_cutoff_dict[runids[n]]:
                if var == "duration":
                    var_list.append(props.read("time")[-1]-props.read("time")[0])
                elif var == "size_ratio":
                    var_list.append(props.read_at_amax("size_rad")/props.read_at_amax("size_tan"))
                elif var in ["n_max","n_avg","n_med","rho_vmax"]:
                    var_list.append(props.read_at_amax(var)/props.sw_pars[0])
                elif var in ["v_max","v_avg","v_med"]:
                    var_list.append(props.read_at_amax(var)/props.sw_pars[1])
                elif var in ["B_max","B_avg","B_med"]:
                    var_list.append(props.read_at_amax(var)/props.sw_pars[2])
                elif var in ["beta_max","beta_avg","beta_med","b_vmax"]:
                    var_list.append(props.read_at_amax(var)/props.sw_pars[4])
                elif var == "pdyn_vmax":
                    var_list.append(m_p*(1.0e+6)*props.read_at_amax("rho_vmax")*((props.read_at_amax("v_max")*1.0e+3)**2)/(props.sw_pars[3]*1.0e-9))
                elif var in ["pd_avg","pd_med","pd_max"]:
                    var_list.append(props.read_at_amax(var)/props.sw_pars[3])
                elif var == "death_distance":
                    var_list.append(np.linalg.norm([props.read("x_vmax")[-1],props.read("y_vmax")[-1],props.read("z_vmax")[-1]]))
                else:
                    var_list.append(props.read_at_amax(var))

    var_list = np.asarray(var_list)

    # Labels for figure
    label_list = ["Duration [s]",
    "Radial size [R$_{e}$]","Tangential size [R$_{e}$]","Radial size/Tangential size",
    "P$_{dyn,vmax}$ [P$_{dyn,sw}$]","P$_{dyn,avg}$ [P$_{dyn,sw}$]","P$_{dyn,med}$ [P$_{dyn,sw}$]","P$_{dyn,max}$ [P$_{dyn,sw}$]",
    "n$_{max}$ [n$_{sw}$]","n$_{avg}$ [n$_{sw}$]","n$_{med}$ [n$_{sw}$]","n$_{v,max}$ [n$_{sw}$]",
    "v$_{max}$ [v$_{sw}$]","v$_{avg}$ [v$_{sw}$]","v$_{med}$ [v$_{sw}$]",
    "B$_{max}$ [B$_{IMF}$]","B$_{avg}$ [B$_{IMF}$]","B$_{med}$ [B$_{IMF}$]",
    "$\\beta _{max}$ [$\\beta _{sw}$]","$\\beta _{avg}$ [$\\beta _{sw}$]","$\\beta _{med}$ [$\\beta _{sw}$]","$\\beta _{v,max}$ [$\\beta _{sw}$]",
    "T$_{avg}$ [MK]","T$_{med}$ [MK]","T$_{max}$ [MK]",
    "T$_{Parallel,avg}$ [MK]","T$_{Parallel,med}$ [MK]","T$_{Parallel,max}$ [MK]",
    "T$_{Perpendicular,avg}$ [MK]","T$_{Perpendicular,med}$ [MK]","T$_{Perpendicular,max}$ [MK]",
    "Area [R$_{e}^{2}$]",
    "r$_{v,max}$ at time of death [R$_{e}$]"]

    # X-limits and bin widths for figure
    xmin_list=[10,
    0,0,0,
    1.25,1.25,1.25,1.25,
    1,1,1,1,
    0.6,0.6,0.6,
    1.25,1.25,1.25,
    1,1,1,1,
    0,0,0,
    0,0,0,
    0,0,0,
    0,
    8]

    xmax_list=[60,
    3,1,7,
    3,3,3,3,
    3,3,3,3,
    1.2,1.2,1.2,
    6,6,6,
    1000,1000,1000,1000,
    25,25,25,
    25,25,25,
    25,25,25,
    1.5,
    18]

    step_list = [2,
    0.25,0.05,0.2,
    0.05,0.05,0.05,0.05,
    0.1,0.1,0.1,0.1,
    0.05,0.05,0.05,
    0.25,0.25,0.25,
    100,100,100,100,
    1,1,1,
    1,1,1,
    1,1,1,
    0.05,
    0.5]


    # Create figure
    plt.ioff()
    #plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(label_list[var_dict[var]],fontsize=20)
    ax.set_ylabel("Fraction of SLAMS",fontsize=20)
    ax.set_xlim(xmin_list[var_dict[var]],xmax_list[var_dict[var]])
    ax.set_ylim(0,1)
    weights = np.ones(var_list.shape)/float(var_list.size) # Normalise by total number of jets

    # Logarithmic scale for plasma beta
    if var in ["beta_max","beta_avg","beta_med","b_vmax"]:
        bins = np.arange(0,3.25,0.25)
        bins = 10**bins
        plt.xscale("log")
        ax.set_xlim(1,xmax_list[var_dict[var]])
        hist = ax.hist(var_list,weights=weights,bins=bins)
    else:
        bins = np.arange(xmin_list[var_dict[var]],xmax_list[var_dict[var]]+step_list[var_dict[var]],step_list[var_dict[var]])
        if var == "death_distance":
            ax.set_xlim(8,xmax_list[var_dict[var]])
            bins = np.arange(8,xmax_list[var_dict[var]]+step_list[var_dict[var]],step_list[var_dict[var]])
        hist = ax.hist(var_list,bins=bins,weights=weights)

    ax.axvline(np.median(var_list), linestyle="dashed", color="black", linewidth=2)

    plt.title(",".join(runids),fontsize=20)
    plt.tight_layout()

    # Create output directory
    if not os.path.exists("SLAMS/figures/histograms/"+"_".join(runids)+"/"):
        try:
            os.makedirs("SLAMS/figures/histograms/"+"_".join(runids)+"/")
        except OSError:
            pass

    # Save figure
    fig.savefig("SLAMS/figures/histograms/"+"_".join(runids)+"/"+var+"_"+str(time_thresh)+".png")
    print("SLAMS/figures/histograms/"+"_".join(runids)+"/"+var+"_"+str(time_thresh)+".png")

    plt.close(fig)

    return None

class PropReader:
    # Class for reading jet property files

    def __init__(self,ID,runid,start=580,fname=None):

        self.ID = ID
        self.runid = runid
        self.start = start
        self.sw_pars = ja.sw_par_dict(runid)
        self.sw_pars[0] /= 1.0e+6
        self.sw_pars[1] /= 1.0e+3
        self.sw_pars[2] /= 1.0e-9
        self.sw_pars[3] /= 1.0e-9

        if type(fname) is not str:
            self.fname = str(start)+"."+ID+".props"
        else:
            self.fname = fname

        try:
            self.props = pd.read_csv("SLAMS/slams/"+runid+"/"+self.fname).as_matrix()
        except IOError:
            raise IOError("File not found!")

        var_list = ["time","x_mean","y_mean","z_mean","A","Nr_cells","r_mean","theta_mean","phi_mean","size_rad","size_tan","x_vmax","y_vmax","z_vmax","n_avg","n_med","n_max","v_avg","v_med","v_max","B_avg","B_med","B_max","T_avg","T_med","T_max","TPar_avg","TPar_med","TPar_max","TPerp_avg","TPerp_med","TPerp_max","beta_avg","beta_med","beta_max","x_min","rho_vmax","b_vmax","pd_avg","pd_med","pd_max"]
        n_list = list(xrange(len(var_list)))
        self.var_dict = dict(zip(var_list,n_list))

    def read(self,name):
        if name not in self.var_dict:
            print("Variable not found!")
            return None
        else:
            return self.props[:,self.var_dict[name]]

    def amax_index(self):
        return self.read("A").argmax()

    def read_at_amax(self,name):
        return self.read(name)[self.amax_index()]

def slams_hist_script():

    runids = ["ABA","ABC","AEA","AEC"]
    #runids = ["BFD"]

    var_list = ["duration",
    "size_rad","size_tan","size_ratio",
    "pdyn_vmax","pd_avg","pd_med","pd_max",
    "n_max","n_avg","n_med","rho_vmax",
    "v_max","v_avg","v_med",
    "B_max","B_avg","B_med",
    "beta_max","beta_avg","beta_med","b_vmax",
    "T_avg","T_med","T_max",
    "TPar_avg","TPar_med","TPar_max",
    "TPerp_avg","TPerp_med","TPerp_max",
    "A","death_distance"]

    for var in var_list:
        slams_all_hist(runids,var,time_thresh=10)

    return None

def slams_hist_script_vs(runids):

    var_list = ["duration",
    "size_rad","size_tan","size_ratio",
    "pdyn_vmax","pd_avg","pd_med","pd_max",
    "n_max","n_avg","n_med","rho_vmax",
    "v_max","v_avg","v_med",
    "B_max","B_avg","B_med",
    "beta_max","beta_avg","beta_med","b_vmax",
    "T_avg","T_med","T_max",
    "TPar_avg","TPar_med","TPar_max",
    "TPerp_avg","TPerp_med","TPerp_max",
    "A","death_distance"]

    for var in var_list:
        slams_vs_hist(runids,var,time_thresh=10)

    return None

def hist_script_script():

    slams_hist_script()
    slams_hist_script_vs(["ABA","AEA"])
    slams_hist_script_vs(["ABC","AEC"])

    return None