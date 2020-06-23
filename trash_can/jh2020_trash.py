def jh20_slams_movie(start,stop,var="Pdyn",vmax=15e-9):

    outputdir = wrkdir_DNR+"jh20_slams_movie/{}/".format(var)
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    bulkpath = jx.find_bulkpath("ABC")
    for itr in range(start,stop+1):
        filepath = bulkpath+"bulk.{}.vlsv".format(str(itr).zfill(7))

        colmap = "parula"

        pt.plot.plot_colormap(filename=filepath,outputfile=outputdir+"{}.png".format(str(itr).zfill(5)),boxre=[6,18,-6,6],var=var,usesci=0,lin=1,vmin=0,vmax=15e-9,colormap=colmap,external=jh20_slams_ext,pass_vars=["rho","v","CellID","Pdyn","RhoNonBackstream","PTensorNonBackstreamDiagonal","Mms","B","X","Y"])

def jh20_slams_ext(ax, XmeshXY,YmeshXY, pass_maps):

    cellids = pass_maps["CellID"]
    B = pass_maps["B"]
    X = pass_maps["X"]
    Y = pass_maps["Y"]
    rho = pass_maps["rho"]
    pdyn = pass_maps["Pdyn"]
    pr_PTDNBS = pass_maps["PTensorNonBackstreamDiagonal"]
    pr_rhonbs = pass_maps["RhoNonBackstream"]

    T_sw = 0.5e+6
    epsilon = 1.e-10
    kb = 1.38065e-23

    pr_pressurenbs = (1.0/3.0) * (pr_PTDNBS.sum(-1))
    pr_TNBS = pr_pressurenbs/ ((pr_rhonbs + epsilon) * kb)

    B_sw = 5.0e-9
    pd_sw = 3.3e6*600e3*600e3*m_p

    Bmag = np.linalg.norm(B,axis=-1)

    slams = np.ma.masked_greater_equal(Bmag,3.0*B_sw)
    #slams.mask[pr_TNBS >= 2.0*T_sw] = False
    #slams.mask[pdyn<=0.5*pd_sw] = False
    slams_mask = slams.mask.astype(int)

    slams_cont = ax.contour(XmeshXY,YmeshXY,slams_mask,[0.5],linewidths=0.8,colors=jx.orange)

def get_indent_depth(runid,crit="ew_pd"):

    x_res = 227000/r_e

    sj_ids,slams_ids=find_slams_of_jet(runid)
    indents = []
    depths = []

    for n in range(sj_ids.size):
        sj_props = jio.PropReader(str(sj_ids[n]).zfill(5),runid,transient="slamsjet")
        x_sj = sj_props.read("x_mean")
        y_sj = sj_props.read("y_mean")
        t_sj = sj_props.read("time")
        sj_dist = jx.bs_rd(runid,t_sj,x_sj,y_sj)
        sj_dist_min = np.min(sj_dist)

        slams_props = jio.PropReader(str(slams_ids[n]).zfill(5),runid,transient="slams")
        is_upstream_slams = slams_props.read("is_upstream")
        if np.all(is_upstream_slams==0.0):
            continue
        t_slams = slams_props.read("time")
        last_time = t_slams[is_upstream_slams>0][-1]


        if crit == "ew_pd":
            bow_shock_value = slams_props.read_at_time("ew_pd_enh",last_time)/ja.sw_normalisation(runid,"pd_avg")
        elif crit == "nonloc":
            bs_ch = slams_props.read_at_time("xbs_ch",last_time)
            bs_rho = slams_props.read_at_time("xbs_rho",last_time)
            bs_mms = slams_props.read_at_time("xbs_mms",last_time)
            bow_shock_value = np.linalg.norm([bs_ch-bs_rho,bs_rho-bs_mms,bs_mms-bs_ch])
        else:
            slams_cells = slams_props.get_cells()
            last_cells = np.array(slams_cells)[is_upstream_slams>0][-1]
            cell_pos = np.array([jx.get_cell_coordinates(runid,cellid)/r_e for cellid in last_cells])
            cell_x = cell_pos[:,0]
            cell_y = cell_pos[:,1]
            cell_t_arr = np.ones_like(cell_x)*(t_slams[is_upstream_slams>0][-1])
            slams_bs_dist = jx.bs_rd(runid,cell_t_arr,cell_x,cell_y)
            upstream_dist_min = np.min(slams_bs_dist)
            bow_shock_value = upstream_dist_min-x_res

        depths.append(sj_dist_min)
        indents.append(bow_shock_value)

    return [np.array(depths),np.array(indents)]

def jh2020_fig4(crit="ew_pd"):

    runids = ["ABA","ABC","AEA","AEC"]
    marker_list = ["x","o","^","v"]

    fig,ax = plt.subplots(1,1,figsize=(10,10))
    for runid in runids:
        depths,indents = get_indent_depth(runid,crit=crit)
        ax.plot(depths,indents,marker_list[runids.index(runid)],label=runid)

    ax.set_xlabel("$\mathrm{Last~X-X_{bs}~[R_e]}$",fontsize=20,labelpad=10)
    #ax.set_ylabel("$\mathrm{Indentation~[R_e]}$",fontsize=20,labelpad=10)
    if crit == "ew_pd":
        ax.set_ylabel("$\mathrm{Mean~earthward~P_{dyn}~[P_{dyn,sw}]}$",fontsize=20,labelpad=10)
    elif crit == "nonloc":
        ax.set_ylabel("$\mathrm{Bow~shock~nonlocality~[R_e]}$",fontsize=20,labelpad=10)
    else:
        ax.set_ylabel("$\mathrm{Bow~shock~indentation~[R_e]}$",fontsize=20,labelpad=10)
    ax.legend(frameon=False,numpoints=1,markerscale=2)
    ax.tick_params(labelsize=20)
    ax.axvline(0,linestyle="dashed",linewidth=0.6,color="black")
    #ax.axhline(0,linestyle="dashed",linewidth=0.6,color="black")
    #ax.plot([-3.0,3.0],[-3.0,3.0],linestyle="dashed",linewidth=0.6,color="black")
    ax.set_xlim(-2.5,0.5)
    #ax.set_ylim(-0.3,0.6)

    if not os.path.exists(homedir+"Figures/jh2020"):
        try:
            os.makedirs(homedir+"Figures/jh2020")
        except OSError:
            pass

    fig.savefig(homedir+"Figures/jh2020/fig4_{}.png".format(crit))
    plt.close(fig)

def separate_jets_new(runid,allow_relatives=True):
    # Separate events into slamsjets, jets and slams

    runids = ["ABA","ABC","AEA","AEC"]

    sj_ids = []
    jet_ids = []
    slams_ids = []

    for n1 in range(6000):

        try:
            props = jio.PropReader(str(n1).zfill(5),runid,transient="slamsjet")
        except:
            continue

        if props.read("is_slams").any() and props.read("is_jet").any():
            if allow_relatives:
                sj_ids.append(n1)
                slams_ids.append(n1)
                jet_ids.append(n1)
            else:
                non_jet_time = props.read("time")[props.read("is_jet")==1][0]-0.5 # last time when event is not jet
                non_slams_time = props.read("time")[props.read("is_slams")==1][-1]+0.5 # first time when event is not slams
                try:
                    bs_arrival = props.read("time")[props.read("at_bow_shock")==1][0]
                except:
                    bs_arrival = props.read("time")[props.read("at_bow_shock")==1]
                try:
                    bs_departure = props.read("time")[props.read("at_bow_shock")==1][-1]
                except:
                    bs_departure = props.read("time")[props.read("at_bow_shock")==1]
                if "splinter" in props.meta:
                    splinter_time = props.read("time")[props.read("is_splinter")==1][0] # time of first splintering
                    extra_splin_times = np.array(props.get_splin_times()) # times of additional splinterings, if any
                    if splinter_time > bs_departure or (extra_splin_times > bs_departure).any():
                        continue
                    else:
                        if np.logical_and(props.read("is_splinter"),props.read("is_jet")).any():
                            sj_ids.append(n1)
                            jet_ids.append(n1)
                            slams_ids.append(n1)
                        else:
                            slams_ids.append(n1)
                if "merger" in props.meta:
                    merger_time = props.read("time")[props.read("is_merger")==1][0] # time of first merging
                    if merger_time < bs_arrival:
                        continue
                    else:
                        sj_ids.append(n1)
                        jet_ids.append(n1)
                        slams_ids.append(n1)
        elif props.read("is_jet").any():
            if props.read("at_bow_shock")[0] != 1:
                continue
            else:
                if allow_relatives:
                    jet_ids.append(n1)
                else:
                    if "splinter" in props.meta:
                        splinter_time = props.read("time")[props.read("is_splinter")==1][0] # time of first splintering
                        extra_splin_times = np.array(props.get_splin_times()) # times of additional splinterings, if any
                        try:
                            bs_departure = props.read("time")[props.read("at_bow_shock")==1][-1]
                        except:
                            bs_departure = props.read("time")[props.read("at_bow_shock")==1]
                        if splinter_time > bs_departure or (extra_splin_times > bs_departure).any():
                            continue
                        else:
                            jet_ids.append(n1)
                    else:
                        jet_ids.append(n1)

        elif props.read("is_slams").any():
            if props.read("at_bow_shock")[-1] != 1:
                continue
            else:
                if allow_relatives:
                    slams_ids.append(n1)
                else:
                    if "merger" in props.meta:
                        merger_time = props.read("time")[props.read("is_merger")==1][0] # time of first merging
                        try:
                            bs_arrival = props.read("time")[props.read("at_bow_shock")==1][0]
                        except:
                            bs_arrival = props.read("time")[props.read("at_bow_shock")==1]
                        if merger_time < bs_arrival:
                            continue
                        else:
                            slams_ids.append(n1)
                    else:
                        slams_ids.append(n1)

    return [np.unique(sj_ids),np.unique(jet_ids),np.unique(slams_ids)]

def separate_jets(runid,allow_splinters=True):
    # Separate events into slamsjets, non-slams jets, and non-jet slams

    runids = ["ABA","ABC","AEA","AEC"]
    run_cutoff_dict = dict(zip(runids,[10,8,10,8])) # DEPRECATED

    sj_jet_ids = []
    non_sj_ids = []
    pure_slams_ids = []

    for n1 in range(6000):

        try:
            props = jio.PropReader(str(n1).zfill(5),runid,transient="slamsjet")
        except:
            continue

        # is the event a slamsjet?
        if np.logical_and(props.read("is_slams")==1,props.read("is_jet")==1).any():
            sj_bool = True # event is categorised as slamsjet by default
            non_jet_time = props.read("time")[props.read("is_jet")==1][0]-0.5 # last time when event is not jet
            non_slams_time = props.read("time")[props.read("is_slams")==1][-1]+0.5 # first time when event is not slams

            # is the event a splinter?
            if not allow_splinters and "splinter" in props.meta:
                splinter_time = props.read("time")[props.read("is_splinter")==1][0] # time of first splintering
                extra_splin_times = np.array(props.get_splin_times()) # times of additional splinterings, if any
                if splinter_time >= non_slams_time:
                    sj_bool = False
                elif (extra_splin_times >= non_slams_time).any():
                    sj_bool = False

            # is the event a merger?
            if not allow_splinters and "merger" in props.meta:
                merger_time = props.read("time")[props.read("is_merger")==1][0] # time of first merging
                if merger_time <= non_jet_time:
                    sj_bool = False

            if sj_bool:
                sj_jet_ids.append(n1)
            else:
                continue

        # is the event a non-slams jet?
        elif (props.read("is_jet")==1).any():
            if not allow_splinters and "splinter" in props.meta: # discard splinters unconditionally
                continue
            elif props.read("at_bow_shock")[0] != 1: # is event not at bow shock?
                continue
            else:
                non_sj_ids.append(n1)

        # is the event a non-jet slams?
        elif (props.read("is_slams")==1).any():
            if not allow_splinters and "merger" in props.meta:
                continue
            # else:
            #     pure_slams_ids.append(n1)
            else:
                pure_slams_ids.append(n1)

    return [np.array(sj_jet_ids),np.array(non_sj_ids),np.array(pure_slams_ids)]

def separate_jets_old(runid):

    runids = ["ABA","ABC","AEA","AEC"]
    run_cutoff_dict = dict(zip(runids,[10,8,10,8]))

    sj_jet_ids = []
    non_sj_ids = []

    for n1 in range(3000):
        try:
            props = jio.PropReader(str(n1).zfill(5),runid,transient="jet")
        except:
            continue

        # if "splinter" in props.meta:
        #     continue

        if props.read("sep_from_bs")[0] > 0.5:
            continue

        jet_first_cells = props.get_cells()[0]
        jet_first_time = props.read("time")[0]

        for n2 in range(3000):
            if n2 == 2999:
                if "splinter" not in props.meta:
                    non_sj_ids.append(n1)
                break

            try:
                props_sj = jio.PropReader(str(n2).zfill(5),runid,transient="slamsjet")
            except:
                continue

            sj_cells = props_sj.get_cells()
            sj_times = props_sj.read("time")
            try:
                matched_cells = sj_cells[np.where(sj_times==jet_first_time)[0][0]]
            except:
                continue

            if np.intersect1d(jet_first_cells,matched_cells).size > 0.05*len(jet_first_cells):
                sj_jet_ids.append(n1)
                break


    return [np.array(sj_jet_ids),np.array(non_sj_ids)]

def find_slams_of_jet(runid):

    sj_ids=[]
    slams_ids=[]

    for n1 in range(3000):
        try:
            props_sj = jio.PropReader(str(n1).zfill(5),runid,transient="slamsjet")
        except:
            continue

        sj_first_cells = props_sj.get_cells()[0]
        for n2 in range(3000):
            try:
                props_slams = jio.PropReader(str(n2).zfill(5),runid,transient="slams")
            except:
                continue
            slams_first_cells = props_slams.get_cells()[0]
            if np.intersect1d(slams_first_cells,sj_first_cells).size > 0.25*len(slams_first_cells):
                sj_ids.append(n1)
                slams_ids.append(n2)
                break

    return [np.array(sj_ids),np.array(slams_ids)]

def jh2020_fig2(xlim=[200.,399.5]):

    # time_arr = np.arange(580./2,1179./2+1./2,0.5)
    # time_list = [time_arr,np.array([time_arr,time_arr,time_arr,time_arr]).T,np.array([time_arr,time_arr,time_arr,time_arr]).T,time_arr,np.array([time_arr,time_arr]).T,time_arr]
    norm_list = [1.e6,1.e3,1.e3,1.e3,1.e3,1.e-9,1.e-9,1.e-9,1.e-9,1.e-9,1.e6,1.e6,1.]
    color_list = ["black", jx.medium_blue, jx.dark_blue, jx.orange]

    # data_in = np.loadtxt("taito_wrkdir/timeseries/ABC/1814507/580_1179").T
    # data_out = np.loadtxt("taito_wrkdir/timeseries/ABC/1814525/580_1179").T
    data_in = np.loadtxt("taito_wrkdir/timeseries/ABC/1814506/400_799").T
    data_out = np.loadtxt("taito_wrkdir/timeseries/ABC/1794536/400_799").T

    time_arr = data_in[0]
    time_list = [time_arr,np.array([time_arr,time_arr,time_arr,time_arr]).T,np.array([time_arr,time_arr,time_arr,time_arr]).T,time_arr,np.array([time_arr,time_arr]).T,time_arr]

    data_in = np.array([data_in[n+1]/norm_list[n] for n in range(len(data_in)-1)])
    data_out = np.array([data_out[n+1]/norm_list[n] for n in range(len(data_out)-1)])

    label_list = ["$\mathrm{\\rho~[cm^{-3}]}$","$\mathrm{v~[km/s]}$","$\mathrm{B~[nT]}$","$\mathrm{P_{dyn}~[nPa]}$","$\mathrm{T~[MK]}$","$\mathrm{\\beta}$"]

    fig,ax_list = plt.subplots(6,2,figsize=(15,15),sharex=True,sharey="row")

    #annot_list_list = [[""],["vx","vy","vz","v"],["Bx","By","Bz","B"],[""],["TPar","TPerp"],[""]]
    annot_list_list = [[""],["v","vx","vy","vz"],["B","Bx","By","Bz"],[""],["TPar","TPerp"],[""]]
    #re_arr_arr = np.array([3,0,1,2])
    re_arr_arr = np.array([0,1,2,3])

    for col in range(2):

        data = [data_in,data_out][col]
        xtitle = ["Inside bow shock","Outside bow shock"][col]
        data_list = [data[0],data[1:5][re_arr_arr].T,data[5:9][re_arr_arr].T,data[9],data[10:12].T,data[12]]
        for row in range(6):
            ann_list = annot_list_list[row]
            var = data_list[row]
            time = time_list[row]
            ax = ax_list[row][col]
            ax.tick_params(labelsize=15)
            ax.axvline(338.5,linestyle="dashed",linewidth=0.8)
            if len(var.T) == 4:
                ax.axhline(0,linestyle="dashed",linewidth=0.8)
            ax.plot(time,var)
            if col == 0:
                ax.set_ylabel(label_list[row],fontsize=15,labelpad=10)
                ax.axvspan(340,356,color="red",alpha=0.3,ec="none")
            if col == 1:
                ax.axvspan(325,335.5,color="red",alpha=0.3,ec="none")
            if row == 0:
                ax.set_title(xtitle,fontsize=15)
            if row == 5:
                ax.set_xlabel("Simulation time [s]",fontsize=15,labelpad=10)
            ax.set_xlim(xlim[0],xlim[1])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6,prune="lower"))
            for m in range(len(ann_list)):
                ax.annotate(ann_list[m],xy=(0.8+m*0.2/len(ann_list),0.05),xycoords="axes fraction",color=color_list[m])

    if not os.path.exists(homedir+"Figures/jh2020"):
        try:
            os.makedirs(homedir+"Figures/jh2020")
        except OSError:
            pass

    fig.savefig(homedir+"Figures/jh2020/fig2.png")
    plt.close(fig)

def jh2020_fig3():

    # epoch_arr,SEA_mean_ABA,SEA_std_ABA = jh2020_SEA("ABA")
    # epoch_arr,SEA_mean_ABC,SEA_std_ABC = jh2020_SEA("ABC")
    # epoch_arr,SEA_mean_AEA,SEA_std_AEA = jh2020_SEA("AEA")
    # epoch_arr,SEA_mean_AEC,SEA_std_AEC = jh2020_SEA("AEC")

    hist_ABA,bin_edges = jh2020_hist("ABA")
    hist_ABC,bin_edges = jh2020_hist("ABC")
    hist_AEA,bin_edges = jh2020_hist("AEA")
    hist_AEC,bin_edges = jh2020_hist("AEC")

    bins = bin_edges[:-1]

    fig,ax = plt.subplots(1,1,figsize=(10,7))

    ax.set_xlabel("$\mathrm{X-X_{bs}~[R_e]}$",labelpad=10,fontsize=20)
    # ax.set_ylabel("$\mathrm{P_{dyn,mean}~[P_{dyn,SW}]}$",labelpad=10,fontsize=20)
    ax.set_ylabel("Normalised count",labelpad=10,fontsize=20)
    #ax.set_xlim(-2.0,2.0)
    ax.axvline(0,linestyle="dashed",linewidth="0.5")
    ax.tick_params(labelsize=20)

    # ax.plot(epoch_arr,SEA_mean_ABA,label="ABA")
    # ax.plot(epoch_arr,SEA_mean_ABC,label="ABC")
    # ax.plot(epoch_arr,SEA_mean_AEA,label="AEA")
    # ax.plot(epoch_arr,SEA_mean_AEC,label="AEC")

    ax.step(bins,hist_ABA,where="post",label="ABA")
    ax.step(bins,hist_ABC,where="post",label="ABC")
    ax.step(bins,hist_AEA,where="post",label="AEA")
    ax.step(bins,hist_AEC,where="post",label="AEC")


    ax.legend(frameon=False,numpoints=1,markerscale=3)

    if not os.path.exists(homedir+"Figures/jh2020"):
        try:
            os.makedirs(homedir+"Figures/jh2020")
        except OSError:
            pass

    fig.savefig(homedir+"Figures/jh2020/fig3.png")
    plt.close(fig)

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

    bs_dist_arr = jx.bs_dist(runid,time_arr,x_arr,y_arr)

    pd_arr = pd_arr[np.argsort(bs_dist_arr)]
    bs_dist_arr.sort()

    return (bs_dist_arr,pd_arr)

def jh2020_SEA(runid,transient="slamsjet"):

    pd_sw = jx.sw_par_dict(runid)[3]/1.0e-9

    epoch_arr = np.arange(-2.0,2.005,0.01)
    SEA_arr = np.zeros_like(epoch_arr)
    SEA_mean = np.zeros_like(epoch_arr)
    SEA_std = np.zeros_like(epoch_arr)

    for n in range(3000):
        jetid = str(n).zfill(5)
        try:
            props = jio.PropReader(jetid,runid,transient=transient)
        except:
            continue

        bs_dist,pd_arr = get_transient_xseries(runid,jetid,transient=transient)
        pd_arr = pd_arr[np.argsort(bs_dist)]/pd_sw
        bs_dist.sort()

        pd_epoch = np.interp(epoch_arr,bs_dist,pd_arr,left=np.nan,right=np.nan)
        SEA_arr = np.vstack((SEA_arr,pd_epoch))

    SEA_arr = SEA_arr[1:]
    SEA_mean = np.nanmean(SEA_arr,axis=0)
    SEA_std = np.nanstd(SEA_arr,axis=0,ddof=1)

    return (epoch_arr,SEA_mean,SEA_std)

def jh2020_hist(runid,transient="slamsjet"):

    hist_arr = np.array([])

    sj_counter = 0
    for n in range(3000):
        jetid = str(n).zfill(5)
        try:
            props = jio.PropReader(jetid,runid,transient=transient)
        except:
            continue

        bs_dist,pd_arr = get_transient_xseries(runid,jetid,transient=transient)
        bs_dist.sort()

        hist_arr = np.append(hist_arr,bs_dist)
        sj_counter += 1

    weight_arr = np.ones_like(hist_arr)/sj_counter
    hist,bin_edges = np.histogram(hist_arr,bins=20,range=(-1.5,1.5),weights=weight_arr)

    return (hist,bin_edges)
