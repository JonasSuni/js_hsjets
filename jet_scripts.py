import plot_contours as pc
import pytools as pt
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jet_analyser as ja
import jet_contours as jc
import jetfile_make as jfm

from matplotlib import rc

parula = pc.make_parula()

m_p = 1.672621898e-27
r_e = 6.371e+6

###PROP MAKER FILES HERE###

def prop_file_maker(run,start,stop,halftimewidth):
    # create properties files, with custom jet criteria for bulk files in 
    # range start,stop (inclusive)

    timerange = xrange(start,stop+1)

    for n in timerange:
        props = ja.jet_script_cust(n,run,halftimewidth,boxre=[6,16,-6,6],min_size=50,max_size=3000,neighborhood_reach=[1,1],freeform_file_id="")

    return None

def prop_file_maker_AH(run,start,stop,halftimewidth):
    # create properties files, with AH jet criteria, for bulk files in
    # range start,stop (inclusive)

    timerange = xrange(start,stop+1)

    for n in timerange:
        props = ja.jet_script(n,run,halftimewidth,criterion="AH",boxre=[8,16,-6,6],min_size=100,max_size=3000,neighborhood_reach=[1,1],freeform_file_id="")

    return None





###FIGURE MAKERS HERE###

def magp_ratio(runid):

    filenames = os.listdir("Props/"+runid)

    mag_p_bool = np.array([])

    for filename in filenames:

        props = pd.read_csv("Props/"+runid+"/"+filename).as_matrix()

        mag_p_bool = np.append(mag_p_bool,props[:,25])

    magp_ratio = float(mag_p_bool[mag_p_bool>0].size)/float(mag_p_bool.size)

    return magp_ratio

def hist_xy(runid,var1,var2,figname,normed_b=True,weight_b=True,bins=15):
    # create 2D histogram of the specified variables

    rc('text', usetex=False)

    # list filenames of files in folder
    filenames = os.listdir("Props/"+runid)

    # initialise variables
    x = np.array([])
    y = np.array([])
    nr_cells = np.array([])

    # create dictionary for axis labels
    label_list = pd.read_csv("Props/"+runid+"/"+filenames[0]).columns.tolist()
    label_length = len(label_list)
    label_dict = dict(zip(xrange(label_length),label_list))

    for filename in filenames:

        # open properties file
        props = pd.read_csv("Props/"+runid+"/"+filename).as_matrix()

        # append the values of the properties to the variables
        x = np.append(x,props[:,var1])
        y = np.append(y,props[:,var2])
        nr_cells = np.append(nr_cells,props[:,22])

    if not weight_b:
        nr_cells *= 0
        nr_cells += 1.0

    # create figure
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(label_dict[var1])
    ax.set_ylabel(label_dict[var2])

    # draw histogram
    xy_hist = ax.hist2d(x,y,bins=bins,normed=normed_b,weights=nr_cells)
    plt.colorbar(xy_hist[3], ax=ax)

    # save figure
    plt.savefig("Figures/"+figname+".png")

    rc('text', usetex=True)

def plot_xy(runid,var1,var2,figname):
    # plot the two specified variables against each other

    rc('text', usetex=False)

    # list filenames of files in folder
    filenames = os.listdir("Props/"+runid)

    # initialise variables
    x = np.array([])
    y = np.array([])

    # create dictionary for axis labels
    label_list = pd.read_csv("Props/"+runid+"/"+filenames[0]).columns.tolist()
    label_length = len(label_list)
    label_dict = dict(zip(xrange(label_length),label_list))

    for filename in filenames:

        # open properties file
        props = pd.read_csv("Props/"+runid+"/"+filename).as_matrix()

        # append the values of the properties to the variables
        x = np.append(x,props[:,var1])
        y = np.append(y,props[:,var2])

    # create figure
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(label_dict[var1])
    ax.set_ylabel(label_dict[var2])

    # draw plot
    xy_plot = ax.plot(x,y,"x",color="black")

    # save figure
    plt.savefig("Figures/"+figname+".png")

    rc('text', usetex=True)

def var_hist_mult(runid,var1,figname,normed_b=True,weight_b=True):
    # create histogram of specified variable

    rc('text', usetex=False)

    # list filenames of files in folder
    filenames = os.listdir("Props/"+runid)

    # initialise variables
    hist_var = np.array([])
    nr_cells = np.array([])

    # create dictionary for axis labels
    label_list = pd.read_csv("Props/"+runid+"/"+filenames[0]).columns.tolist()
    label_length = len(label_list)
    label_dict = dict(zip(xrange(label_length),label_list))

    for filename in filenames:

        # open properties file
        props = pd.read_csv("Props/"+runid+"/"+filename).as_matrix()

        # append the values of the properties to the variables
        hist_var = np.append(hist_var,props[:,var1])
        nr_cells = np.append(nr_cells,props[:,22])

    if not weight_b:
        nr_cells *= 0
        nr_cells += 1.0

    # create figure
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(label_dict[var1])
    ax.set_ylabel("Probability density")

    # draw histogram
    var_h = ax.hist(hist_var,bins=15,weights=nr_cells,normed=normed_b)

    # save figure
    plt.savefig("Figures/"+figname+".png")

    rc('text', usetex=True)

def y_hist_mult(runid,figname,normed_b=True,weight_b=True):

    filenames = os.listdir("Props/"+runid)

    y = np.array([])
    nr_cells = np.array([])

    for filename in filenames:

        props = pd.read_csv("Props/"+runid+"/"+filename).as_matrix()

        y = np.append(y,props[:,19])
        nr_cells = np.append(nr_cells,props[:,22])

  
    if not weight_b:
        nr_cells *= 0
        nr_cells += 1.0

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("$Y_{vmax}$ $[R_e]$")
    ax.set_ylabel("Probability density")

    y_h = ax.hist(y,bins=list(xrange(-6,7)),weights=nr_cells,normed=normed_b)

    fig.show()

    plt.savefig("Figures/"+figname+".png")

def phi_hist_mult(runid,figname,normed_b=True,weight_b=True):

    filenames = os.listdir("Props/"+runid)

    phi = np.array([])
    nr_cells = np.array([])

    for filename in filenames:

        props = pd.read_csv("Props/"+runid+"/"+filename).as_matrix()

        phi = np.append(phi,props[:,23])
        nr_cells = np.append(nr_cells,props[:,22])

    if not weight_b:
        nr_cells *= 0
        nr_cells += 1.0

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("$Angle [deg]$")
    ax.set_ylabel("Probability density")

    phi_h = ax.hist(phi,bins=list(xrange(-40,41,5)),weights=nr_cells,normed=normed_b)

    fig.show()

    plt.savefig("Figures/"+figname+".png")





###CONTOUR MAKER HERE###

def contour_gen(run,start,stop):

    for n in xrange(start,stop+1):

        
        jfm.pahkmake(n,run,180)

        pt.plot.plot_colormap(filename="VLSV/temp_all.vlsv",var="spdyn",colormap=parula,outputdir="Contours/"+run+"/"+str(n)+"_",boxre=[6,16,-6,6],vmin=0,vmax=1.5,cbtitle="nPa",usesci=0,lin=1,external=jc.jc_cust_new,pass_vars=["npdynx","nrho","tapdyn"])

    return None





###VIRTUAL SPACECRAFT MAKER HERE###

def make_wave_figs(outputfolder,start,stop):

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    f_path = "/proj/vlasov/2D/ABA/bulk/"

    for n in xrange(start,stop+1):

        f_name="bulk."+str(n).zfill(7)+".vlsv"

        pt.plot.plot_colormap(filename=f_path+f_name,outputdir=outputfolder+"/"+str(n)+"_",colormap=parula,vmin=0,vmax=1.5,usesci=0,lin=1,cbtitle="nPa",boxre=[6,18,-10,2],expression=pc.expr_pdyn,external=sc_pos_marker_SLAMS,pass_vars=["rho","v"])

    return None

def make_jet_figs(outputfolder,start,stop):

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    f_path = "/proj/vlasov/2D/ABA/bulk/"

    for n in xrange(start,stop+1):

        f_name="bulk."+str(n).zfill(7)+".vlsv"

        pt.plot.plot_colormap(filename=f_path+f_name,outputdir=outputfolder+"/"+str(n)+"_",colormap=parula,vmin=0,vmax=1.5,usesci=0,lin=1,cbtitle="nPa",boxre=[6,18,-10,2],expression=pc.expr_pdyn,external=sc_pos_marker_JET,pass_vars=["rho","v"])

    return None

def sc_pos_marker_SLAMS(ax,XmeshXY,YmeshXY,extmaps,ext_pars):

    pos_mark = ax.plot(12,-4.4,marker="o",color="black",markersize=2)

def sc_pos_marker_JET(ax,XmeshXY,YmeshXY,extmaps,ext_pars):

    rho = extmaps[0]

    msk = np.loadtxt("Masks/ABA/611.mask")

    msk = np.reshape(msk,rho.shape)

    jet = np.ma.masked_greater_equal(msk,1.0)
    jet.fill_value = 0
    jet[jet.mask == False] = 1

    contour_fromfile = ax.contour(XmeshXY,YmeshXY,jet.filled(),[0.5],linewidths=0.75,colors="black")

    pos_mark1 = ax.plot(9.17,-3.69,marker="o",color="green",markersize=3)
    pos_mark2 = ax.plot(9.98,-4.4,marker="o",color="red",markersize=3)
    pos_mark3 = ax.plot(10.81,-5.15,marker="o",color="cyan",markersize=3)

def get_pos_index(posre,runid,file_number):

    vlsvreader = pt.vlsvfile.VlsvReader("/proj/vlasov/2D/"+runid+"/bulk/bulk."+str(file_number).zfill(7)+".vlsv")

    X = vlsvreader.read_variable("X")
    Y = vlsvreader.read_variable("Y")
    cellids = vlsvreader.read_variable("CellID")

    x_i = np.where(abs(X-posre[0]*r_e)<120000)[0]
    y_i = np.where(abs(Y-posre[1]*r_e)<120000)[0]

    pos_index = np.intersect1d(x_i,y_i)

    pos_id = cellids[pos_index[0]]

    return pos_id

def jet_spacecrafts(start,stop,figname="",font_size=20):

    vlsvreader = pt.vlsvfile.VlsvReader("/proj/vlasov/2D/ABA/bulk/bulk.0000611.vlsv")

    pos1 = [9.17,-3.69]
    pos2 = [9.98,-4.4]
    pos3 = [10.81,-5.15]

    X = vlsvreader.read_variable("X")
    Y = vlsvreader.read_variable("Y")
    cellids = vlsvreader.read_variable("CellID")

    x1_i = np.where(abs(X-pos1[0]*r_e)<120000)[0]
    y1_i = np.where(abs(Y-pos1[1]*r_e)<120000)[0]

    x2_i = np.where(abs(X-pos2[0]*r_e)<120000)[0]
    y2_i = np.where(abs(Y-pos2[1]*r_e)<120000)[0]

    x3_i = np.where(abs(X-pos3[0]*r_e)<120000)[0]
    y3_i = np.where(abs(Y-pos3[1]*r_e)<120000)[0]

    pos1_index = np.intersect1d(x1_i,y1_i)
    pos2_index = np.intersect1d(x2_i,y2_i)
    pos3_index = np.intersect1d(x3_i,y3_i)

    pos1_id = cellids[pos1_index[0]]
    pos2_id = cellids[pos2_index[0]]
    pos3_id = cellids[pos3_index[0]]

    f_path = "/proj/vlasov/2D/ABA/bulk/"

    B_imf = np.array([-np.cos(np.deg2rad(30))*5.0e-9,np.sin(np.deg2rad(30))*5.0e-9,0])

    Bx_arr = np.array([])
    By_arr = np.array([])
    Bz_arr = np.array([])
    Bmag_arr = np.array([])
    vx_arr = np.array([])
    vy_arr = np.array([])
    vz_arr = np.array([])
    vmag_arr = np.array([])
    rho_arr = np.array([])
    pdyn_arr = np.array([])

    open("jetsc/jetsc_1.csv","w").close()
    open("jetsc/jetsc_2.csv","w").close()
    open("jetsc/jetsc_3.csv","w").close()

    sc1_file = open("jetsc/jetsc_1.csv","a")
    sc2_file = open("jetsc/jetsc_2.csv","a")
    sc3_file = open("jetsc/jetsc_3.csv","a")

    sc1_file.write("t [s],Bx [nT],By [nT],Bz [nT],|B| [nT],vx [km/s],vy [km/s],vz [km/s],|v| [km/s],rho [cm^-3],Pdyn [nPa]")
    sc2_file.write("t [s],Bx [nT],By [nT],Bz [nT],|B| [nT],vx [km/s],vy [km/s],vz [km/s],|v| [km/s],rho [cm^-3],Pdyn [nPa]")
    sc3_file.write("t [s],Bx [nT],By [nT],Bz [nT],|B| [nT],vx [km/s],vy [km/s],vz [km/s],|v| [km/s],rho [cm^-3],Pdyn [nPa]")

    for n in xrange(start,stop+1):

        sc1_file.write("\n")
        sc2_file.write("\n")
        sc3_file.write("\n")

        f_name="bulk."+str(n).zfill(7)+".vlsv"

        f = pt.vlsvfile.VlsvReader(f_path+f_name)

        B = f.read_variable("B",cellids=[pos1_id,pos2_id,pos3_id])
        Bx = B[:,0]
        By = B[:,1]
        Bz = B[:,2]
        Bmag = np.linalg.norm(B,axis=-1)

        v = f.read_variable("v",cellids=[pos1_id,pos2_id,pos3_id])
        vx = v[:,0]
        vy = v[:,1]
        vz = v[:,2]
        vmag = np.linalg.norm(v,axis=-1)

        rho = f.read_variable("rho",cellids=[pos1_id,pos2_id,pos3_id])

        pdyn = m_p*rho*(vmag**2)

        time = float(n)/2

        Bx_arr = np.append(Bx_arr,Bx)
        By_arr = np.append(By_arr,By)
        Bz_arr = np.append(Bz_arr,Bz)
        Bmag_arr = np.append(Bmag_arr,Bmag)
        vx_arr = np.append(vx_arr,vx)
        vy_arr = np.append(vy_arr,vy)
        vz_arr = np.append(vz_arr,vz)
        vmag_arr = np.append(vmag_arr,vmag)
        rho_arr = np.append(rho_arr,rho)
        pdyn_arr = np.append(pdyn_arr,pdyn)

        sc1_arr = [time,Bx[0]/1.0e-9,By[0]/1.0e-9,Bz[0]/1.0e-9,Bmag[0]/1.0e-9,vx[0]/1.0e+3,vy[0]/1.0e+3,vz[0]/1.0e+3,vmag[0]/1.0e+3,rho[0]/1.0e+6,pdyn[0]/1.0e-9]
        sc2_arr = [time,Bx[1]/1.0e-9,By[1]/1.0e-9,Bz[1]/1.0e-9,Bmag[1]/1.0e-9,vx[1]/1.0e+3,vy[1]/1.0e+3,vz[1]/1.0e+3,vmag[1]/1.0e+3,rho[1]/1.0e+6,pdyn[1]/1.0e-9]
        sc3_arr = [time,Bx[2]/1.0e-9,By[2]/1.0e-9,Bz[2]/1.0e-9,Bmag[2]/1.0e-9,vx[2]/1.0e+3,vy[2]/1.0e+3,vz[2]/1.0e+3,vmag[2]/1.0e+3,rho[2]/1.0e+6,pdyn[2]/1.0e-9]

        sc1_file.write(",".join(map(str,sc1_arr)))
        sc2_file.write(",".join(map(str,sc2_arr)))
        sc3_file.write(",".join(map(str,sc3_arr)))

    time_arr = np.array(xrange(start,stop+1)).astype(float)/2

    Bx_arr /= 1.0e-9
    By_arr /= 1.0e-9
    Bz_arr /= 1.0e-9
    Bmag_arr /= 1.0e-9
    vx_arr /= 1.0e+3
    vy_arr /= 1.0e+3
    vz_arr /= 1.0e+3
    vmag_arr /= 1.0e+3
    rho_arr /= 1.0e+6
    pdyn_arr /= 1.0e-9

    Bx_arr = np.reshape(Bx_arr,(len(xrange(start,stop+1)),3))
    By_arr = np.reshape(By_arr,(len(xrange(start,stop+1)),3))
    Bz_arr = np.reshape(Bz_arr,(len(xrange(start,stop+1)),3))
    Bmag_arr = np.reshape(Bmag_arr,(len(xrange(start,stop+1)),3))
    vx_arr = np.reshape(vx_arr,(len(xrange(start,stop+1)),3))
    vy_arr = np.reshape(vy_arr,(len(xrange(start,stop+1)),3))
    vz_arr = np.reshape(vz_arr,(len(xrange(start,stop+1)),3))
    vmag_arr = np.reshape(vmag_arr,(len(xrange(start,stop+1)),3))
    rho_arr = np.reshape(rho_arr,(len(xrange(start,stop+1)),3))
    pdyn_arr = np.reshape(pdyn_arr,(len(xrange(start,stop+1)),3))

    plt.ion()
    fig = plt.figure(figsize=(30,15))

    Bx_ax = fig.add_subplot(521)
    By_ax = fig.add_subplot(523)
    Bz_ax = fig.add_subplot(525)
    Bmag_ax = fig.add_subplot(527)
    vx_ax = fig.add_subplot(522)
    vy_ax = fig.add_subplot(524)
    vz_ax = fig.add_subplot(526)
    vmag_ax = fig.add_subplot(528)
    rho_ax = fig.add_subplot(529)
    pdyn_ax = fig.add_subplot(5,2,10)

    Bx_ax.grid()
    By_ax.grid()
    Bz_ax.grid()
    Bmag_ax.grid()
    vx_ax.grid()
    vy_ax.grid()
    vz_ax.grid()
    vmag_ax.grid()
    rho_ax.grid()
    pdyn_ax.grid()

    #vx_ax.yaxis.tick_right()
    #vx_ax.yaxis.set_ticks_position('both')
    #vx_ax.yaxis.set_label_position("right")
    #vy_ax.yaxis.tick_right()
    #vy_ax.yaxis.set_ticks_position('both')
    #vy_ax.yaxis.set_label_position("right")
    #vz_ax.yaxis.tick_right()
    #vz_ax.yaxis.set_ticks_position('both')
    #vz_ax.yaxis.set_label_position("right")
    #vmag_ax.yaxis.tick_right()
    #vmag_ax.yaxis.set_ticks_position('both')
    #vmag_ax.yaxis.set_label_position("right")
    #pdyn_ax.yaxis.tick_right()
    #pdyn_ax.yaxis.set_ticks_position('both')
    #pdyn_ax.yaxis.set_label_position("right")

    Bx_ax.set_xlim(290,320)
    By_ax.set_xlim(290,320)
    Bz_ax.set_xlim(290,320)
    Bmag_ax.set_xlim(290,320)
    vx_ax.set_xlim(290,320)
    vy_ax.set_xlim(290,320)
    vz_ax.set_xlim(290,320)
    vmag_ax.set_xlim(290,320)
    rho_ax.set_xlim(290,320)
    pdyn_ax.set_xlim(290,320)

    Bx_ax.set_ylim(-10,10)
    By_ax.set_ylim(-15,30)
    Bz_ax.set_ylim(-20,20)
    Bmag_ax.set_ylim(10,30)
    vx_ax.set_ylim(-400,0)
    vy_ax.set_ylim(-400,200)
    vz_ax.set_ylim(-400,200)
    vmag_ax.set_ylim(0,500)
    rho_ax.set_ylim(2,5)
    pdyn_ax.set_ylim(0,1.5)

    Bx_ax.set_yticks([-10,-5,0,5,10])
    By_ax.set_yticks([-15,0,15,30])
    Bz_ax.set_yticks([-20,-10,0,10,20])
    Bmag_ax.set_yticks([10,20,30])
    vx_ax.set_yticks([-400,-300,-200,-100,0])
    vy_ax.set_yticks([-400,-200,0,200])
    vz_ax.set_yticks([-400,-200,0,200])
    vmag_ax.set_yticks([0,100,200,300,400,500])
    rho_ax.set_yticks([2,3,4,5])
    pdyn_ax.set_yticks([0,0.5,1,1.5])

    Bx_ax.set_xticks([295,300,305,310,315,320])
    By_ax.set_xticks([295,300,305,310,315,320])
    Bz_ax.set_xticks([295,300,305,310,315,320])
    Bmag_ax.set_xticks([295,300,305,310,315,320])
    vx_ax.set_xticks([295,300,305,310,315,320])
    vy_ax.set_xticks([295,300,305,310,315,320])
    vz_ax.set_xticks([295,300,305,310,315,320])
    vmag_ax.set_xticks([295,300,305,310,315,320])
    rho_ax.set_xticks([295,300,305,310,315,320])
    pdyn_ax.set_xticks([295,300,305,310,315,320])

    Bx_ax.set_xticklabels([])
    By_ax.set_xticklabels([])
    Bz_ax.set_xticklabels([])
    Bmag_ax.set_xticklabels([])
    vx_ax.set_xticklabels([])
    vy_ax.set_xticklabels([])
    vz_ax.set_xticklabels([])
    vmag_ax.set_xticklabels([])


    Bx_ax.set_ylabel("$B_x$ [nT]",fontsize=font_size)
    By_ax.set_ylabel("$B_y$ [nT]",fontsize=font_size)
    Bz_ax.set_ylabel("$B_z$ [nT]",fontsize=font_size)
    Bmag_ax.set_ylabel("$|B|$ [nT]",fontsize=font_size)
    vx_ax.set_ylabel("$v_x$ [km/s]",fontsize=font_size)
    vy_ax.set_ylabel("$v_y$ [km/s]",fontsize=font_size)
    vz_ax.set_ylabel("$v_z$ [km/s]",fontsize=font_size)
    vmag_ax.set_ylabel("$|v|$ [km/s]",fontsize=font_size)
    rho_ax.set_ylabel("$\\rho$ [cm$^{-3}$]",fontsize=font_size)
    pdyn_ax.set_ylabel("$P_{dyn}$ [nPa]",fontsize=font_size)
    pdyn_ax.set_xlabel("Time [s]",fontsize=font_size)
    rho_ax.set_xlabel("Time [s]",fontsize=font_size)

    Bx_ax.tick_params(labelsize=16)
    By_ax.tick_params(labelsize=16)
    Bz_ax.tick_params(labelsize=16)
    Bmag_ax.tick_params(labelsize=16)
    vx_ax.tick_params(labelsize=16)
    vy_ax.tick_params(labelsize=16)
    vz_ax.tick_params(labelsize=16)
    vmag_ax.tick_params(labelsize=16)
    rho_ax.tick_params(labelsize=16)
    vmag_ax.tick_params(labelsize=16)
    pdyn_ax.tick_params(labelsize=16)

    sp1_color="green"

    Bx_ax.plot(time_arr,Bx_arr[:,0],color=sp1_color,linewidth=2)
    By_ax.plot(time_arr,By_arr[:,0],color=sp1_color,linewidth=2)
    Bz_ax.plot(time_arr,Bz_arr[:,0],color=sp1_color,linewidth=2)
    Bmag_ax.plot(time_arr,Bmag_arr[:,0],color=sp1_color,linewidth=2)
    vx_ax.plot(time_arr,vx_arr[:,0],color=sp1_color,linewidth=2)
    vy_ax.plot(time_arr,vy_arr[:,0],color=sp1_color,linewidth=2)
    vz_ax.plot(time_arr,vz_arr[:,0],color=sp1_color,linewidth=2)
    vmag_ax.plot(time_arr,vmag_arr[:,0],color=sp1_color,linewidth=2)
    rho_ax.plot(time_arr,rho_arr[:,0],color=sp1_color,linewidth=2)
    vmag_ax.plot(time_arr,vmag_arr[:,0],color=sp1_color,linewidth=2)
    pdyn_ax.plot(time_arr,pdyn_arr[:,0],color=sp1_color,linewidth=2)

    Bx_ax.plot(time_arr,Bx_arr[:,1],color="red",linewidth=2)
    By_ax.plot(time_arr,By_arr[:,1],color="red",linewidth=2)
    Bz_ax.plot(time_arr,Bz_arr[:,1],color="red",linewidth=2)
    Bmag_ax.plot(time_arr,Bmag_arr[:,1],color="red",linewidth=2)
    vx_ax.plot(time_arr,vx_arr[:,1],color="red",linewidth=2)
    vy_ax.plot(time_arr,vy_arr[:,1],color="red",linewidth=2)
    vz_ax.plot(time_arr,vz_arr[:,1],color="red",linewidth=2)
    vmag_ax.plot(time_arr,vmag_arr[:,1],color="red",linewidth=2)
    rho_ax.plot(time_arr,rho_arr[:,1],color="red",linewidth=2)
    vmag_ax.plot(time_arr,vmag_arr[:,1],color="red",linewidth=2)
    pdyn_ax.plot(time_arr,pdyn_arr[:,1],color="red",linewidth=2)

    Bx_ax.plot(time_arr,Bx_arr[:,2],color="cyan",linewidth=2)
    By_ax.plot(time_arr,By_arr[:,2],color="cyan",linewidth=2)
    Bz_ax.plot(time_arr,Bz_arr[:,2],color="cyan",linewidth=2)
    Bmag_ax.plot(time_arr,Bmag_arr[:,2],color="cyan",linewidth=2)
    vx_ax.plot(time_arr,vx_arr[:,2],color="cyan",linewidth=2)
    vy_ax.plot(time_arr,vy_arr[:,2],color="cyan",linewidth=2)
    vz_ax.plot(time_arr,vz_arr[:,2],color="cyan",linewidth=2)
    vmag_ax.plot(time_arr,vmag_arr[:,2],color="cyan",linewidth=2)
    rho_ax.plot(time_arr,rho_arr[:,2],color="cyan",linewidth=2)
    vmag_ax.plot(time_arr,vmag_arr[:,2],color="cyan",linewidth=2)
    pdyn_ax.plot(time_arr,pdyn_arr[:,2],color="cyan",linewidth=2)

    Bx_ax.axvline(305.5,linestyle="dashed",color="black",linewidth=2)
    By_ax.axvline(305.5,linestyle="dashed",color="black",linewidth=2)
    Bz_ax.axvline(305.5,linestyle="dashed",color="black",linewidth=2)
    Bmag_ax.axvline(305.5,linestyle="dashed",color="black",linewidth=2)
    vx_ax.axvline(305.5,linestyle="dashed",color="black",linewidth=2)
    vy_ax.axvline(305.5,linestyle="dashed",color="black",linewidth=2)
    vz_ax.axvline(305.5,linestyle="dashed",color="black",linewidth=2)
    vmag_ax.axvline(305.5,linestyle="dashed",color="black",linewidth=2)
    rho_ax.axvline(305.5,linestyle="dashed",color="black",linewidth=2)
    vmag_ax.axvline(305.5,linestyle="dashed",color="black",linewidth=2)
    pdyn_ax.axvline(305.5,linestyle="dashed",color="black",linewidth=2)

    plt.tight_layout()

    fig.show()

    plt.savefig("jetsc/scrafts_"+str(start)+"_"+str(stop)+"_"+figname+".png")
    print("jetsc/scrafts_"+str(start)+"_"+str(stop)+"_"+figname+".png")

    sc1_file.close()
    sc2_file.close()
    sc3_file.close()

    return None

def slams_spacecraft(start,stop,pos=[12,-4.4],font_size=16,fig_size=(16,16),figname="",cols2=False,alt_labels=False):

    vlsvreader = pt.vlsvfile.VlsvReader("/proj/vlasov/2D/ABA/bulk/bulk.0000611.vlsv")

    X = vlsvreader.read_variable("X")
    Y = vlsvreader.read_variable("Y")
    cellids = vlsvreader.read_variable("CellID")

    x_i = np.where(abs(X-pos[0]*r_e)<120000)[0]
    y_i = np.where(abs(Y-pos[1]*r_e)<120000)[0]

    pos_index = np.intersect1d(x_i,y_i)

    pos_id = cellids[pos_index[0]]
    f_path = "/proj/vlasov/2D/ABA/bulk/"

    B_imf = np.array([-np.cos(np.deg2rad(30))*5.0e-9,np.sin(np.deg2rad(30))*5.0e-9,0])

    phi_arr = np.array([])
    Bx_arr = np.array([])
    By_arr = np.array([])
    Bz_arr = np.array([])
    Bmag_arr = np.array([])
    rho_arr = np.array([])
    vmag_arr = np.array([])
    pdyn_arr = np.array([])

    for n in xrange(start,stop+1):

        f_name="bulk."+str(n).zfill(7)+".vlsv"

        f = pt.vlsvfile.VlsvReader(f_path+f_name)

        B = f.read_variable("B",cellids=pos_id)
        
        phi = np.rad2deg(np.arccos(np.dot(B,B_imf)/(np.linalg.norm(B)*np.linalg.norm(B_imf))))

        Bx = B[0]
        By = B[1]
        Bz = B[2]
        Bmag = np.linalg.norm(B,axis=-1)

        rho = f.read_variable("rho",cellids=pos_id)
        vmag = f.read_variable("v",cellids=pos_id,operator="magnitude")
        pdyn = m_p*rho*(vmag**2)

        phi_arr = np.append(phi_arr,phi)
        Bx_arr = np.append(Bx_arr,Bx)
        By_arr = np.append(By_arr,By)
        Bz_arr = np.append(Bz_arr,Bz)
        Bmag_arr = np.append(Bmag_arr,Bmag)
        rho_arr = np.append(rho_arr,rho)
        vmag_arr = np.append(vmag_arr,vmag)
        pdyn_arr = np.append(pdyn_arr,pdyn)

    time_arr = np.array(xrange(start,stop+1)).astype(float)/2

    Bx_arr /= 1.0e-9
    By_arr /= 1.0e-9
    Bz_arr /= 1.0e-9
    Bmag_arr /= 1.0e-9
    rho_arr /= 1.0e+6
    vmag_arr /= 1.0e+3
    pdyn_arr /= 1.0e-9

    plt.ion()
    fig = plt.figure(figsize=fig_size)

    # 1 COLUMNS VERSION

    if not cols2:

        #phi_ax = fig.add_subplot(811)
        Bx_ax = fig.add_subplot(711)
        By_ax = fig.add_subplot(712)
        Bz_ax = fig.add_subplot(713)
        Bmag_ax = fig.add_subplot(714)
        rho_ax = fig.add_subplot(715)
        vmag_ax = fig.add_subplot(716)
        pdyn_ax = fig.add_subplot(717)

    # ALTERNATING LABELS

    if not cols2 and alt_labels:

        By_ax.yaxis.tick_right()
        By_ax.yaxis.set_ticks_position('both')
        By_ax.yaxis.set_label_position("right")
        Bmag_ax.yaxis.tick_right()
        Bmag_ax.yaxis.set_ticks_position('both')
        Bmag_ax.yaxis.set_label_position("right")
        vmag_ax.yaxis.tick_right()
        vmag_ax.yaxis.set_ticks_position('both')
        vmag_ax.yaxis.set_label_position("right")

    # 2 COLUMN VERSION

    if cols2:

        Bx_ax = fig.add_subplot(421)
        By_ax = fig.add_subplot(423)
        Bz_ax = fig.add_subplot(425)
        Bmag_ax = fig.add_subplot(427)
        rho_ax = fig.add_subplot(422)
        vmag_ax = fig.add_subplot(424)
        pdyn_ax = fig.add_subplot(426)

    #phi_ax.set_xlim(260,310)
    Bx_ax.set_xlim(260,310)
    By_ax.set_xlim(260,310)
    Bz_ax.set_xlim(260,310)
    Bmag_ax.set_xlim(260,310)
    rho_ax.set_xlim(260,310)
    vmag_ax.set_xlim(260,310)
    pdyn_ax.set_xlim(260,310)

    #phi_ax.set_ylim(10,50)
    Bx_ax.set_ylim(-10,0)
    By_ax.set_ylim(0,10)
    Bz_ax.set_ylim(-5,5)
    Bmag_ax.set_ylim(4,10)
    rho_ax.set_ylim(0.5,2)
    vmag_ax.set_ylim(600,800)
    pdyn_ax.set_ylim(0.5,2)

    #phi_ax.set_ylabel("$\\phi$ [deg]",fontsize=font_size)
    Bx_ax.set_ylabel("$B_x$ [nT]",fontsize=font_size)
    By_ax.set_ylabel("$B_y$ [nT]",fontsize=font_size)
    Bz_ax.set_ylabel("$B_z$ [nT]",fontsize=font_size)
    Bmag_ax.set_ylabel("$|B|$ [nT]",fontsize=font_size)
    rho_ax.set_ylabel("$\\rho$ [cm$^{-3}$]",fontsize=font_size)
    vmag_ax.set_ylabel("$|v|$ [km/s]",fontsize=font_size)
    pdyn_ax.set_ylabel("$P_{dyn}$ [nPa]",fontsize=font_size)
    pdyn_ax.set_xlabel("Time [s]",fontsize=font_size)

    #phi_ax.set_yticks([10,20,30,40,50])
    Bx_ax.set_yticks([-10,-7.5,-5,-2.5,0])
    By_ax.set_yticks([0,2.5,5,7.5,10])
    Bz_ax.set_yticks([-5,-2.5,0,2.5,5])
    Bmag_ax.set_yticks([4,6,8,10])
    rho_ax.set_yticks([0.5,1,1.5,2])
    vmag_ax.set_yticks([600,650,700,750,800])
    pdyn_ax.set_yticks([0.5,1,1.5,2])

    #phi_ax.set_xticks([])
    Bx_ax.set_xticks([])
    By_ax.set_xticks([])
    Bz_ax.set_xticks([])
    Bmag_ax.set_xticks([])
    rho_ax.set_xticks([])
    vmag_ax.set_xticks([])
    pdyn_ax.set_xticks([270,280,290,300,310])

    #phi_ax.tick_params(labelsize=16)
    Bx_ax.tick_params(labelsize=16)
    By_ax.tick_params(labelsize=16)
    Bz_ax.tick_params(labelsize=16)
    Bmag_ax.tick_params(labelsize=16)
    rho_ax.tick_params(labelsize=16)
    vmag_ax.tick_params(labelsize=16)
    pdyn_ax.tick_params(labelsize=16)

    #phi_ax.plot(time_arr,phi_arr,color="black",linewidth=2)
    Bx_ax.plot(time_arr,Bx_arr,color="black",linewidth=2)
    By_ax.plot(time_arr,By_arr,color="black",linewidth=2)
    Bz_ax.plot(time_arr,Bz_arr,color="black",linewidth=2)
    Bmag_ax.plot(time_arr,Bmag_arr,color="black",linewidth=2)
    rho_ax.plot(time_arr,rho_arr,color="black",linewidth=2)
    vmag_ax.plot(time_arr,vmag_arr,color="black",linewidth=2)
    pdyn_ax.plot(time_arr,pdyn_arr,color="black",linewidth=2)

    #phi_ax.axvline(280,linestyle="dashed",color="black",linewidth=2)
    Bx_ax.axvline(280,linestyle="dashed",color="black",linewidth=2)
    By_ax.axvline(280,linestyle="dashed",color="black",linewidth=2)
    Bz_ax.axvline(280,linestyle="dashed",color="black",linewidth=2)
    Bz_ax.axhline(0,linestyle="dotted",color="black",linewidth=2)
    Bmag_ax.axvline(280,linestyle="dashed",color="black",linewidth=2)
    rho_ax.axvline(280,linestyle="dashed",color="black",linewidth=2)
    vmag_ax.axvline(280,linestyle="dashed",color="black",linewidth=2)
    pdyn_ax.axvline(280,linestyle="dashed",color="black",linewidth=2)

    plt.tight_layout()

    fig.show()

    plt.savefig("Figures/SLAMS_"+str(start)+"_"+str(stop)+"_"+figname+".png")
    print("Figures/SLAMS_"+str(start)+"_"+str(stop)+"_"+figname+".png")

def wave_spacecraft(start,stop,step,pos=[12,-4.4],font_size=16,fig_size=(16,16)):

    x_def = 60521928.9248/r_e
    y_def = -26995643.4721/r_e

    vlsvreader = pt.vlsvfile.VlsvReader("/proj/vlasov/2D/ABA/bulk/bulk.0000611.vlsv")

    X = vlsvreader.read_variable("X")
    Y = vlsvreader.read_variable("Y")
    cellids = vlsvreader.read_variable("CellID")

    x_i = np.where(abs(X-pos[0]*r_e)<120000)[0]
    y_i = np.where(abs(Y-pos[1]*r_e)<120000)[0]

    pos_index = np.intersect1d(x_i,y_i)

    pos_id = cellids[pos_index[0]]
    f_path = "/proj/vlasov/2D/ABA/bulk/"

    B_arr = np.array([])
    v_arr = np.array([])
    rho_arr = np.array([])
    pdyn_arr = np.array([])

    for n in xrange(start,stop+1,step):

        f_name="bulk."+str(n).zfill(7)+".vlsv"

        f = pt.vlsvfile.VlsvReader(f_path+f_name)

        B = f.read_variable("B",cellids=pos_id,operator="magnitude")
        v = f.read_variable("v",cellids=pos_id,operator="magnitude")
        rho = f.read_variable("rho",cellids=pos_id)

        pdyn = m_p*rho*(v**2)

        B_arr = np.append(B_arr,B)
        v_arr = np.append(v_arr,v)
        rho_arr = np.append(rho_arr,rho)
        pdyn_arr = np.append(pdyn_arr,pdyn)

    time_arr = np.array(xrange(start,stop+1,step)).astype(float)/2

    B_arr /= 1.0e-9
    v_arr /= 1.0e+3
    rho_arr /= 1.0e+6
    pdyn_arr /= 1.0e-9

    plt.ion()
    plt.tick_params(labelsize=16)
    fig = plt.figure(figsize=fig_size)

    B_ax = fig.add_subplot(411)
    v_ax = fig.add_subplot(412)
    rho_ax = fig.add_subplot(413)
    pdyn_ax = fig.add_subplot(414)

    B_ax.set_xlim(200,500)
    v_ax.set_xlim(200,500)
    rho_ax.set_xlim(200,500)
    pdyn_ax.set_xlim(200,500)

    B_ax.set_ylim(0,30)
    v_ax.set_ylim(0,400)
    rho_ax.set_ylim(2,7)
    pdyn_ax.set_ylim(0,1)

    #B_ax.set_ylabel("B [nT]",fontsize=font_size)
    #v_ax.set_ylabel("v [km/s]",fontsize=font_size)
    #rho_ax.set_ylabel("$\\rho$ [cm$^{-3}$]",fontsize=font_size)
    #pdyn_ax.set_ylabel("$P_{dyn}$ [nPa]",fontsize=font_size)
    pdyn_ax.set_xlabel("Time [s]",fontsize=font_size)

    magsh_Bticks = [0,10,20,30]
    magsh_vticks = [0,100,200,300,400]
    magsh_rhoticks = [2,3,4,5,6,7]
    magsh_pdynticks = [0,0.5,1]

    B_ax.set_yticks([0,10,20,30])
    v_ax.set_yticks([0,100,200,300,400])
    rho_ax.set_yticks([2,3,4,5,6,7])
    pdyn_ax.set_yticks([0,0.5,1])
    B_ax.set_xticks([])
    v_ax.set_xticks([])
    rho_ax.set_xticks([])
    pdyn_ax.set_xticks([250,300,350,400,450,500])

    B_ax.plot(time_arr,B_arr)
    v_ax.plot(time_arr,v_arr)
    rho_ax.plot(time_arr,rho_arr)
    pdyn_ax.plot(time_arr,pdyn_arr)

    B_ax.axvline(305.5,linestyle="dashed")
    v_ax.axvline(305.5,linestyle="dashed")
    rho_ax.axvline(305.5,linestyle="dashed")
    pdyn_ax.axvline(305.5,linestyle="dashed")

    plt.tight_layout()

    fig.show()

    plt.savefig("Figures/Minna1b_"+str(start)+"_"+str(stop)+".png")





###MULTI FILE SCRIPTS HERE###

def presentation_script(run_id,fig_name):

    '''props are 
    0: n_avg [cm^-3],   1: n_med [cm^-3],   2: n_max [cm^-3],
    3: v_avg [km/s],    4: v_med [km/s],    5: v_max [km/s],
    6: B_avg [nT],      7: B_med [nT],      8: B_max [nT],
    9: T_avg [MK],      10: T_med [MK],     11: T_max [MK],
    12: Tpar_avg [MK],  13: Tpar_med [MK],  14: Tpar_max [MK],
    15: Tperp_avg [MK], 16: Tperp_med [MK], 17: Tperp_max [MK],
    18: X_vmax [R_e],   19: Y_vmax [R_e],   20: Z_vmax [R_e],
    21: A [km^2],       22: Nr_cells,       23: phi [deg],
    24: r_d [R_e],      25: mag_p_bool,     26: size_x [R_e],
    27: size_y [R_e],   28: MMS,            29: MA'''

    hist_xy(run_id,18,19,fig_name+run_id+"_x_y",normed_b=False,weight_b=True,bins=[np.linspace(8,12,17),np.linspace(-4,4,17)])

    hist_xy(run_id,18,5,fig_name+run_id+"_x_vmax",normed_b=True,weight_b=True,bins=[np.linspace(8,12,17),np.linspace(100,900,17)])
    #hist_xy(run_id,18,12,fig_name+run_id+"_x_Tpar_avg",normed_b=True,weight_b=True)
    #hist_xy(run_id,18,15,fig_name+run_id+"_x_Tperp_avg",normed_b=True,weight_b=True)

    hist_xy(run_id,19,5,fig_name+run_id+"_y_vmax",normed_b=True,weight_b=True,bins=[np.linspace(-4,4,17),np.linspace(100,900,17)])
    #hist_xy(run_id,19,12,fig_name+run_id+"_y_Tpar_avg",normed_b=True,weight_b=True)
    #hist_xy(run_id,19,15,fig_name+run_id+"_y_Tperp_avg",normed_b=True,weight_b=True)

    hist_xy(run_id,18,22,fig_name+run_id+"_x_nrcells",normed_b=True,weight_b=True,bins=[np.linspace(8,12,17),np.linspace(50,1950,17)])
    hist_xy(run_id,19,22,fig_name+run_id+"_y_nrcells",normed_b=True,weight_b=True,bins=[np.linspace(-4,4,17),np.linspace(50,1950,17)])
    #hist_xy(run_id,22,12,fig_name+run_id+"_nrcells_Tpar_avg",normed_b=True,weight_b=True)
    #hist_xy(run_id,22,15,fig_name+run_id+"_nrcells_Tperp_avg",normed_b=True,weight_b=True)
    hist_xy(run_id,22,5,fig_name+run_id+"_nrcells_vmax",normed_b=True,weight_b=True,bins=[np.linspace(50,1950,17),np.linspace(100,900,17)])

    var_hist_mult(run_id,18,fig_name+run_id+"_x_hist",normed_b=True,weight_b=True)
    var_hist_mult(run_id,19,fig_name+run_id+"_y_hist",normed_b=True,weight_b=True)
    var_hist_mult(run_id,5,fig_name+run_id+"_vmax_hist",normed_b=True,weight_b=True)
    #var_hist_mult(run_id,12,fig_name+run_id+"_Tpar_avg_hist",normed_b=True,weight_b=True)
    #var_hist_mult(run_id,15,fig_name+run_id+"_Tperp_avg_hist",normed_b=True,weight_b=True)

    print("Magp_ratio is "+str(magp_ratio(run_id)))

    plt.close("all")

def fromfile_cont_movie(outputfolder,runid,start,stop):

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    if runid == "ABA":

        v_max = 5.0

    elif runid == "ABC":

        v_max = 15.0

    parula = pc.make_parula()

    for n in xrange(start,stop+1):

        pt.plot.plot_colormap(filename="/proj/vlasov/2D/"+runid+"/bulk/bulk."+str(n).zfill(7)+".vlsv",outputdir="Contours/"+outputfolder+"/"+runid+"_"+str(n)+"_",usesci=0,lin=1,vmin=0.8,vmax=v_max,colormap=parula,boxre=[4,16,-6,6],cbtitle="",expression=pc.expr_srho,external=jc.jc_fromfile,pass_vars=["rho","CellID"],ext_pars=[runid,n,180])

def make_figs(outputfolder,box_re=[8,16,-6,6],plaschkemax=1,rhomax=6,rhomax5=6,rhomin=0,pdynmax=1.5):

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    file_name = "VLSV/temp_all.vlsv"

    pt.plot.plot_colormap(filename=file_name,var="npdynx",colormap=parula,outputdir=outputfolder+"/Fig2_",usesci=0,lin=1,boxre=box_re,vmax=plaschkemax,vmin=0,cbtitle="",title="$\\rho v_x^2/\\rho_{sw} v_{sw}^2$",external=jc.jc_plaschke,pass_vars=["npdynx","nrho"])

    pt.plot.plot_colormap(filename=file_name,var="tapdyn",colormap=parula,outputdir=outputfolder+"/Fig3a_",usesci=0,lin=1,boxre=box_re,vmax=4,vmin=0,cbtitle="",title="$\\rho v^2/<\\rho v^2>_{3min}$",external=jc.jc_archerhorbury,pass_vars=["tapdyn"])

    pt.plot.plot_colormap(filename=file_name,var="spdyn",colormap=parula,outputdir=outputfolder+"/Fig3b_",usesci=0,lin=1,boxre=box_re,vmax=pdynmax,vmin=0,cbtitle="nPa",title="$\\rho v^2$")

    pt.plot.plot_colormap(filename=file_name,var="tpdynavg",colormap=parula,outputdir=outputfolder+"/Fig3c_",usesci=0,lin=1,boxre=box_re,vmax=pdynmax,vmin=0,cbtitle="nPa",title="$<\\rho v^2>_{3min}$")

    pt.plot.plot_colormap(filename=file_name,var="tarho",colormap=parula,outputdir=outputfolder+"/Fig4a_",usesci=0,lin=1,boxre=box_re,vmax=2,vmin=0,cbtitle="",title="$\\rho/<\\rho>_{3min}$",external=jc.jc_karlsson,pass_vars=["tarho"])

    pt.plot.plot_colormap(filename=file_name,var="srho",colormap=parula,outputdir=outputfolder+"/Fig4b_",usesci=0,lin=1,boxre=box_re,vmax=rhomax,vmin=rhomin,cbtitle="cm$^{-3}$",title="$\\rho$")

    pt.plot.plot_colormap(filename=file_name,var="trhoavg",colormap=parula,outputdir=outputfolder+"/Fig4c_",usesci=0,lin=1,boxre=box_re,vmax=rhomax,vmin=rhomin,cbtitle="cm$^{-3}$",title="$<\\rho>_{3min}$")

    pt.plot.plot_colormap(filename=file_name,var="srho",colormap=parula,outputdir=outputfolder+"/Fig5_",usesci=0,lin=1,boxre=box_re,vmax=rhomax5,vmin=rhomin,cbtitle="cm$^{-3}$",title="$\\rho$",external=jc.jc_all,pass_vars=["npdynx","nrho","tapdyn","tarho"])

def minna_figs(outputfolder,box_re=[8,16,-6,6],plaschkemax=1,rhomax=6,rhomax5=5,rhomin=0.8,pdynmax=1.5):

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    file_name = "VLSV/temp_all.vlsv"

    pt.plot.plot_colormap(filename=file_name,var="npdynx",colormap=parula,outputdir=outputfolder+"/Fig2_",usesci=0,lin=1,boxre=box_re,vmax=plaschkemax,vmin=0,cbtitle="",title="",external=jc.jc_plaschke,pass_vars=["npdynx","nrho"])

    pt.plot.plot_colormap(filename=file_name,var="tapdyn",colormap=parula,outputdir=outputfolder+"/Fig3a_",usesci=0,lin=1,boxre=box_re,vmax=4,vmin=0,cbtitle="",title="",external=jc.jc_archerhorbury,pass_vars=["tapdyn"])

    pt.plot.plot_colormap(filename=file_name,var="spdyn",colormap=parula,outputdir=outputfolder+"/Fig3b_",usesci=0,lin=1,boxre=box_re,vmax=pdynmax,vmin=0,cbtitle="",title="")

    pt.plot.plot_colormap(filename=file_name,var="tpdynavg",colormap=parula,outputdir=outputfolder+"/Fig3c_",usesci=0,lin=1,boxre=box_re,vmax=pdynmax,vmin=0,cbtitle="",title="")

    pt.plot.plot_colormap(filename=file_name,var="tarho",colormap=parula,outputdir=outputfolder+"/Fig4a_",usesci=0,lin=1,boxre=box_re,vmax=2,vmin=0,cbtitle="",title="",external=jc.jc_karlsson,pass_vars=["tarho"])

    pt.plot.plot_colormap(filename=file_name,var="srho",colormap=parula,outputdir=outputfolder+"/Fig4b_",usesci=0,lin=1,boxre=box_re,vmax=rhomax,vmin=0,cbtitle="",title="")

    pt.plot.plot_colormap(filename=file_name,var="trhoavg",colormap=parula,outputdir=outputfolder+"/Fig4c_",usesci=0,lin=1,boxre=box_re,vmax=rhomax,vmin=0,cbtitle="",title="")

    pt.plot.plot_colormap(filename=file_name,var="srho",colormap=parula,outputdir=outputfolder+"/Fig5_",usesci=0,lin=1,boxre=box_re,vmax=rhomax5,vmin=rhomin,cbtitle="",title="",external=jc.jc_all,pass_vars=["npdynx","nrho","tapdyn","tarho"])