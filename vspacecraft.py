import plot_contours as pc
import pytools as pt
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jet_analyser as ja
import jet_contours as jc
import jetfile_make as jfm
import jet_scripts as js

from matplotlib import rc

parula = pc.make_parula()

m_p = 1.672621898e-27
r_e = 6.371e+6

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

def wave_spacecraft(start,stop,step,pos=[12,-4.4],font_size=20,fig_size=(16,16)):

    x_def = 60521928.9248/r_e
    y_def = -26995643.4721/r_e

    pos = [x_def,y_def]

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

    B_ax.set_xticks([250,300,350,400,450,500])
    v_ax.set_xticks([250,300,350,400,450,500])
    rho_ax.set_xticks([250,300,350,400,450,500])
    pdyn_ax.set_xticks([250,300,350,400,450,500])

    B_ax.set_xticklabels([])
    v_ax.set_xticklabels([])
    rho_ax.set_xticklabels([])

    B_ax.plot(time_arr,B_arr,color="black",linewidth=2)
    v_ax.plot(time_arr,v_arr,color="black",linewidth=2)
    rho_ax.plot(time_arr,rho_arr,color="black",linewidth=2)
    pdyn_ax.plot(time_arr,pdyn_arr,color="black",linewidth=2)

    B_ax.axvline(305.5,linestyle="dashed",color="black",linewidth=2)
    v_ax.axvline(305.5,linestyle="dashed",color="black",linewidth=2)
    rho_ax.axvline(305.5,linestyle="dashed",color="black",linewidth=2)
    pdyn_ax.axvline(305.5,linestyle="dashed",color="black",linewidth=2)

    B_ax.tick_params(labelsize=16)
    v_ax.tick_params(labelsize=16)
    rho_ax.tick_params(labelsize=16)
    pdyn_ax.tick_params(labelsize=16)

    plt.tight_layout()

    fig.show()

    plt.savefig("Figures/Minna1b_"+str(start)+"_"+str(stop)+".png")