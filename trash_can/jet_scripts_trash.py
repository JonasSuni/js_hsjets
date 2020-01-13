def draw_all_cont():
    # Draw contours for all criteria
    # NOT FUNCTIONAL
    #raise NotImplementedError("DEPRECATED")

    pt.plot.plot_colormap(filename=vlasdir+"/2D/ABA/bulk/bulk.0000611.vlsv",outputdir="Contours/ALLCONT_",usesci=0,draw=1,lin=1,boxre=[4,18,-12,12],colormap="parula",cbtitle="nPa",scale=1,expression=pc.expr_pdyn,external=ext_crit,var="rho",vmin=0,vmax=1.5,wmark=1,pass_vars=["rho","v","CellID"])

def ext_crit(ax,XmeshXY,YmeshXY,extmaps):
    # NOT FUNCTIONAL
    #raise NotImplementedError("DEPRECATED")

    rho = extmaps["rho"].flatten()
    vx = extmaps["v"][:,:,0].flatten()
    vy = extmaps["v"][:,:,1].flatten()
    vz = extmaps["v"][:,:,2].flatten()
    vmag = np.linalg.norm([vx,vy,vz],axis=0)
    cellids = extmaps["CellID"].flatten()
    XmeshXY = XmeshXY.flatten()
    YmeshXY = YmeshXY.flatten()
    shp = extmaps["rho"].shape

    pdyn = m_p*rho*(vmag**2)
    pdyn_x = m_p*rho*(vx**2)

    pdyn = pdyn[cellids.argsort()]
    pdyn_x = pdyn_x[cellids.argsort()]
    rho = rho[cellids.argsort()]
    XmeshXY = XmeshXY[cellids.argsort()]
    YmeshXY = YmeshXY[cellids.argsort()]

    fullcells = pt.vlsvfile.VlsvReader(vlasdir+"/2D/ABA/bulk/bulk.0000611.vlsv").read_variable("CellID")
    fullcells.sort()

    trho = np.loadtxt(wrkdir_DNR+"tavg/ABA/611_rho.tavg")[np.in1d(fullcells,cellids)]
    tpdyn = np.loadtxt(wrkdir_DNR+"tavg/ABA/611_pdyn.tavg")[np.in1d(fullcells,cellids)]

    rho_sw = 1000000
    v_sw = 750000
    pdyn_sw = m_p*rho_sw*(v_sw**2)

    pdyn = scipy.ndimage.zoom(np.reshape(pdyn,shp),3)
    pdyn_x = scipy.ndimage.zoom(np.reshape(pdyn_x,shp),3)
    rho = scipy.ndimage.zoom(np.reshape(rho,shp),3)
    XmeshXY = scipy.ndimage.zoom(np.reshape(XmeshXY,shp),3)
    YmeshXY = scipy.ndimage.zoom(np.reshape(YmeshXY,shp),3)
    trho = scipy.ndimage.zoom(np.reshape(trho,shp),3)
    tpdyn = scipy.ndimage.zoom(np.reshape(tpdyn,shp),3)

    jetp = np.ma.masked_greater(pdyn_x,0.25*pdyn_sw)
    #jetp.mask[nrho < level_sw] = False
    jetp.fill_value = 0
    jetp[jetp.mask == False] = 1

    jetah = np.ma.masked_greater(pdyn,2*tpdyn)
    jetah.fill_value = 0
    jetah[jetah.mask == False] = 1

    # make karlsson mask
    jetk = np.ma.masked_greater(rho,1.5*trho)
    jetk.fill_value = 0
    jetk[jetk.mask == False] = 1

    # draw contours
    #contour_plaschke = ax.contour(XmeshXY,YmeshXY,jetp.filled(),[0.5],linewidths=0.8, colors="black",label="Plaschke")

    contour_archer = ax.contour(XmeshXY,YmeshXY,jetah.filled(),[0.5],linewidths=0.8, colors="black",label="ArcherHorbury")

    #contour_karlsson = ax.contour(XmeshXY,YmeshXY,jetk.filled(),[0.5],linewidths=0.8, colors="magenta",label="Karlsson")

    return None

def lineout_plot(runid,filenumber,p1,p2,var):
    # DEPRECATED, new version incoming at some point
    #raise NotImplementedError("DEPRECATED, new version incoming at some point")

    # find correct file based on file number and run id
    if runid in ["AEC"]:
        bulkpath = vlasdir+"/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = vlasdir+"/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = vlasdir+"/2D/"+runid+"/bulk/"

    var_dict = {"rho":[1e+6,"$\\rho~[cm^{-3}]$",1],"v":[1e+3,"$v~[km/s]$",750]}

    bulkname = "bulk."+str(filenumber).zfill(7)+".vlsv"

    lin = pt.calculations.lineout(pt.vlsvfile.VlsvReader(bulkpath+bulkname),np.array(p1)*r_e,np.array(p2)*r_e,var,interpolation_order=1,points=100)

    var_arr = lin[2]
    if len(var_arr.shape) == 2:
        var_arr = np.linalg.norm(var_arr,axis=-1)
    r_arr = np.linalg.norm(lin[1],axis=-1)/r_e

    if var in var_dict:
        var_arr /= var_dict[var][0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(var_arr)
    #ax.plot(r_arr,var_arr)
    #ax.set_xlabel("$R~[R_e]$",labelpad=10,fontsize=20)
    ax.tick_params(labelsize=20)
    if var in var_dict:
        ax.set_ylabel(var_dict[var][1],labelpad=10,fontsize=20)
    #ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True,prune="lower"))

    plt.tight_layout()

    fig.show()

    fig.savefig("Contours/"+"lineout_"+runid+"_"+str(filenumber)+"_"+var+".png")

def jet_mult_time_series(runid,start,jetid,thresh = 0.0,transient="jet"):
    # Creates multivariable time series for specified jet

    # Check transient type
    if transient == "jet":
        outputdir = "jet_sizes"
    elif transient == "slamsjet":
        outputdir = wrkdir_DNR+"working/SLAMSJETS/time_series"
    elif transient == "slams":
        outputdir = "SLAMS/time_series"

    # Create outputdir if it doesn't already exist
    if not os.path.exists(outputdir+"/"+runid):
        try:
            os.makedirs(outputdir+"/"+runid)
        except OSError:
            pass

    # Open properties file, read variable data
    props = jio.PropReader(jetid,runid,start,transient=transient)
    var_list = ["time","A","n_max","v_max","pd_max","r_mean"]
    time_arr,area_arr,n_arr,v_arr,pd_arr,r_arr = [props.read(var)/ja.sw_normalisation(runid,var) for var in var_list]

    # Threshold condition
    if np.max(area_arr) < thresh or time_arr.size < 10:
        print("Jet smaller than threshold, exiting!")
        return None

    # Draw figure
    plt.ioff()

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax2.set_xlabel("Time [s]",fontsize=20)
    ax.set_ylabel("Fraction of maximum",fontsize=20)
    ax2.set_ylabel("Fraction of maximum",fontsize=20)
    ax.grid()
    ax2.grid()
    plt.title("Run: {}, ID: {}".format(runid,jetid))
    ax2.plot(time_arr,area_arr/np.max(area_arr),label="A\_max = {:.3g}".format(np.max(area_arr)))
    ax.plot(time_arr,n_arr/np.max(n_arr),label="n\_max = {:.3g}".format(np.max(n_arr)))
    ax.plot(time_arr,v_arr/np.max(v_arr),label="v\_max = {:.3g}".format(np.max(v_arr)))
    ax.plot(time_arr,pd_arr/np.max(pd_arr),label="pd\_max = {:.3g}".format(np.max(pd_arr)))
    ax2.plot(time_arr,r_arr/np.max(r_arr),label="r\_max = {:.3g}".format(np.max(r_arr)))
    #ax2.plot(time_arr,ja.bow_shock_r(runid,time_arr)/np.max(r_arr),label="Bow shock")

    ax.legend(loc="lower right")
    ax2.legend(loc="lower right")

    plt.tight_layout()

    # Save figure
    fig.savefig("{}/{}/{}_mult_time_series.png".format(outputdir,runid,jetid))
    print("{}/{}/{}_mult_time_series.png".format(outputdir,runid,jetid))

    plt.close(fig)

    return None

def jet_time_series(runid,start,jetid,var,thresh = 0.0,transient="jet"):
    # Creates timeseries of specified variable for specified jet

    # Check transient type
    if transient == "jet":
        outputdir = "jet_sizes"
    elif transient == "slamsjet":
        outputdir = wrkdir_DNR+"working/SLAMSJETS/time_series"
    elif transient == "slams":
        outputdir = "SLAMS/time_series"

    # Create outputdir if it doesn't already exist
    if not os.path.exists(outputdir+"/"+runid):
        try:
            os.makedirs(outputdir+"/"+runid)
        except OSError:
            pass

    # Open props file, read time, area and variable data
    props = jio.PropReader(jetid,runid,start,transient=transient)
    time_arr = props.read("time")
    area_arr = props.read("A")
    var_arr = props.read(var)/ja.sw_normalisation(runid,var)

    # Threshold condition
    if np.max(area_arr) < thresh:
        print("Jet smaller than threshold, exiting!")
        return None

    # Draw figure
    plt.ioff()

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time [s]",fontsize=20)
    ax.set_ylabel(var_pars_list(var)[0],fontsize=20)
    plt.grid()
    plt.title("Run: {}, ID: {}".format(runid,jetid))
    ax.plot(time_arr,var_arr,color="black")

    plt.tight_layout()

    # Save figure
    fig.savefig("{}/{}/{}_time_series_{}.png".format(outputdir,runid,jetid,var))
    print("{}/{}/{}_time_series_{}.png".format(outputdir,runid,jetid,var))

    plt.close(fig)

    return None

def jts_make(runid,start,startid,stopid,thresh = 0.0,transient="jet"):
    # Script for creating time series for multiple jets

    for n in range(startid,stopid+1):
        try:
            jet_mult_time_series(runid,start,str(n).zfill(5),thresh=thresh,transient=transient)
        except IOError:
            print("Could not create time series!")

    return None

def SEA_make(runid,var,centering="pd_avg",thresh=5):
    # Creates Superposed Epoch Analysis of jets in specified run, centering specified var around maximum of
    # specified centering variable

    #jetids = dict(zip(["ABA","ABC","AEA","AEC"],[[2,29,79,120,123,129],[6,12,45,55,60,97,111,141,146,156,162,179,196,213,223,235,259,271],[57,62,80,167,182,210,252,282,302,401,408,465,496],[2,3,8,72,78,109,117,127,130]]))[runid]

    # Range of jetids to attempt
    jetids = np.arange(1,2500,1)

    # Define epoch time array, +- 1 minute from center
    epoch_arr = np.arange(-60.0,60.1,0.5)
    SEA_arr = np.zeros_like(epoch_arr) # Initialise superposed epoch array

    for n in jetids:

        # Try reading jet
        try:
            props = jio.PropReader(str(n).zfill(5),runid,580)
        except:
            continue

        # Read time and centering
        time_arr = props.read("time")
        cent_arr = props.read(centering)/ja.sw_normalisation(runid,centering)

        # Threshold condition
        if time_arr.size < thresh:
            continue

        # Read variable data
        var_arr = props.read(var)/ja.sw_normalisation(runid,var)

        # Try scaling to fractional increase
        try:
            var_arr /= sheath_pars_list(var)[1]
            var_arr -= 1
        except:
            pass

        # Interpolate variable data to fit epoch time, and stack it with SEA array
        res_arr = np.interp(epoch_arr,time_arr-time_arr[np.argmax(cent_arr)],var_arr,left=0.0,right=0.0)
        SEA_arr = np.vstack((SEA_arr,res_arr))

    # Remove the row of zeros from stack
    SEA_arr = SEA_arr[1:]

    # Calculate mean and STD of the stack
    SEA_arr_mean = np.mean(SEA_arr,axis=0)
    SEA_arr_std = np.std(SEA_arr,ddof=1,axis=0)

    # Draw figure
    plt.ioff()

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Epoch time [s]",fontsize=20)

    try:
        ax.set_ylabel("Fractional increase {}".format(sheath_pars_list(var)[0]),fontsize=20)
    except:
        ax.set_ylabel("Averaged {}".format(var_pars_list(var)[0]),fontsize=20)

    plt.grid()
    plt.title("Run: {}, Epoch centering: {}".format(runid,centering.replace("_",r"\_")))
    ax.plot(epoch_arr,SEA_arr_mean,color="black")
    ax.fill_between(epoch_arr,SEA_arr_mean-SEA_arr_std,SEA_arr_mean+SEA_arr_std,alpha=0.3)

    plt.tight_layout()

    # Save figure
    if not os.path.exists("Figures/SEA/"+runid+"/"):
        try:
            os.makedirs("Figures/SEA/"+runid+"/")
        except OSError:
            pass

    fig.savefig("Figures/SEA/{}/SEA_{}.png".format(runid,var))
    print("Figures/SEA/{}/SEA_{}.png".format(runid,var))

    plt.close(fig)

    return None

def SEA_script(centering="pd_avg",thresh=5):
    # Script for making several SEA graphs for different runs

    runids = ["ABA","ABC","AEA","AEC"]
    var = ["n_max","v_max","pd_max","n_avg","n_med","v_avg","v_med","pd_avg","pd_med","pdyn_vmax"]

    for runid in runids:
        for v in var:
            SEA_make(runid,v,centering=centering,thresh=thresh)

    return None


def jet_lifetime_plots(var,amax=True):
    # Creates scatter plot of jet lifetime versus variable value either at time of maximum area or global
    # maximum for all ecliptical runs.

    # List of runids
    runids = ["ABA","ABC","AEA","AEC"]

    # Get all filenames in folder
    filenames_list = []
    for runid in runids:
        filenames_list.append(os.listdir(wrkdir_DNR+"working/jets/"+runid))

    # Filter for property files
    file_list_list = []
    for filenames in filenames_list:
        file_list_list.append([filename for filename in filenames if ".props" in filename])

    # Dictionaries for false positive cutoff, marker shape and colour
    run_cutoff_dict = dict(zip(["ABA","ABC","AEA","AEC"],[10,8,10,8]))
    run_marker_dict = dict(zip(["ABA","ABC","AEA","AEC"],["x","o","^","d"]))
    run_color_dict = dict(zip(["ABA","ABC","AEA","AEC"],["black","red","blue","green"]))

    # Initialise lists of coordinates
    x_list_list = [[],[],[],[]]
    y_list_list = [[],[],[],[]]

    for n in range(len(runids)):
        for fname in file_list_list[n]:
            props = jio.PropReader("",runids[n],fname=fname)

            # Condition
            if props.read("time")[-1]-props.read("time")[0] + 0.5 > 10 and max(props.read("r_mean")) > run_cutoff_dict[runids[n]]:
                    x_list_list[n].append(props.read("time")[-1]-props.read("time")[0])
                    if amax:
                        y_list_list[n].append(props.read_at_amax(var)/ja.sw_normalisation(runids[n],var))
                    else:
                        y_list_list[n].append(np.max(props.read(var))/ja.sw_normalisation(runids[n],var))

    # Draw figure
    plt.ioff()

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Lifetime [s]",fontsize=20)
    ax.set_ylabel(var_pars_list(var)[0],fontsize=20)
    plt.grid()

    lines = []
    labs = []

    for n in range(len(runids)):
        line1, = ax.plot(x_list_list[n],y_list_list[n],run_marker_dict[runids[n]],markeredgecolor=run_color_dict[runids[n]],markersize=5,markerfacecolor="None",markeredgewidth=2)
        lines.append(line1)
        labs.append(runids[n])

    plt.title(",".join(runids)+"\nN = "+str(sum([len(l) for l in x_list_list])),fontsize=20)
    plt.legend(lines,labs,numpoints=1)
    plt.tight_layout()

    # Fit line to data and draw it
    x_list_full = []
    y_list_full = []

    for n in range(len(x_list_list)):
        x_list_full+=x_list_list[n]
        y_list_full+=y_list_list[n]

    p = np.polyfit(x_list_full,y_list_full,deg=1)
    x_arr = np.arange(np.min(x_list_full),np.max(x_list_full),1)
    y_arr = np.polyval(p,x_arr)

    ax.plot(x_arr,y_arr,linestyle="dashed")

    # TO DO: Make annotation look nice DONE
    ax.annotate("y = {:5.3f}x + {:5.3f}".format(p[0],p[1]),xy=(0.1,0.9),xycoords="axes fraction")

    # Save figure
    if not os.path.exists("Figures/paper/misc/scatter/"+"_".join(runids)+"/"):
        try:
            os.makedirs("Figures/paper/misc/scatter/"+"_".join(runids)+"/")
        except OSError:
            pass

    if amax:
        fig.savefig("Figures/paper/misc/scatter/{}/{}_{}_amax.png".format("_".join(runids),"lifetime",var))
        print("Figures/paper/misc/scatter/{}/{}_{}_amax.png".format("_".join(runids),"lifetime",var))
    else:
        fig.savefig("Figures/paper/misc/scatter/{}/{}_{}.png".format("_".join(runids),"lifetime",var))
        print("Figures/paper/misc/scatter/{}/{}_{}.png".format("_".join(runids),"lifetime",var))

    plt.close(fig)

    return None
