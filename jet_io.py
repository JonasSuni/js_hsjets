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

m_p = 1.672621898e-27
r_e = 6.371e+6

class Jet:

    def __init__(self,ID,runid,birthday):

        self.ID = ID
        self.runid = runid
        self.birthday = birthday
        self.cellids = []
        self.times = [birthday]

        print("Created jet with ID "+self.ID)

    def return_cellid_string(self):

        return "\n".join([",".join(map(str,l)) for l in self.cellids])

    def return_time_string(self):

        return "\n".join(map(str,self.times))

def jet_maker(runid,start,stop):

    outputdir = "/homeappl/home/sunijona/events/"+runid+"/"

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    for file_nr in xrange(start,stop+1):

        # find correct file based on file number and run id
        if runid in ["AEC","AEF","BEA","BEB"]:
            bulkpath = "/proj/vlasov/2D/"+runid+"/"
        else:
            bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

        if runid == "AED":
            bulkname = "bulk.old."+str(file_nr).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(file_nr).zfill(7)+".vlsv"

        # open vlsv file for reading
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

        msk = ja.make_cust_mask(file_nr,runid,180,[8,16,-6,6])

        print(len(msk))

        jets = ja.sort_jets(vlsvobj,msk,100,5000,[1,1])

        open(outputdir+str(file_nr)+".events","w").close()

        fileobj = open(outputdir+str(file_nr)+".events","a")

        for jet in jets:

            fileobj.write(",".join(map(str,jet))+"\n")

        fileobj.close()

    return None

def timefile_write(runid,filenr,key,time):

    tf = open("/homeappl/home/sunijona/jets/"+runid+"/"+str(filenr)+"."+key+".times","a")
    tf.write(str(time)+"\n")
    tf.close()

    return None

def timefile_read(runid,filenr,key):

    tf = open("/homeappl/home/sunijona/jets/"+runid+"/"+str(filenr)+"."+key+".times","r")
    contents = tf.read().split("\n")[:-1]
    tf.close()

    return map(float,contents)

def jetfile_write(runid,filenr,key,jet):

    jf = open("/homeappl/home/sunijona/jets/"+runid+"/"+str(filenr)+"."+key+".jet","a")
    jf.write(",".join(map(str,jet))+"\n")
    jf.close()

    return None

def jetfile_read(runid,filenr,key):

    outputlist = []

    jf = open("/homeappl/home/sunijona/jets/"+runid+"/"+str(filenr)+"."+key+".jet","r")
    contents = jf.read()
    lines = contents.split("\n")[:-1]

    for line in lines:

        outputlist.append(map(int,line.split(",")))

    return outputlist

def eventfile_read(runid,filenr):

    outputlist = []

    ef = open("/homeappl/home/sunijona/events/"+runid+"/"+str(filenr)+".events","r")
    contents = ef.read()
    lines = contents.split("\n")[:-1]

    for line in lines:

        outputlist.append(map(int,line.split(",")))

    return outputlist

def propfile_write(runid,filenr,key,props):

    open("/homeappl/home/sunijona/jets/"+runid+"/"+str(filenr)+"."+key+".props","w").close()
    pf = open("/homeappl/home/sunijona/jets/"+runid+"/"+str(filenr)+"."+key+".props","a")
    pf.write("time [s],x_mean [R_e],y_mean [R_e],A [R_e^2],Nr_cells,phi [deg],r_d [R_e],size_rad [R_e],size_tan [R_e]"+"\n")
    pf.write("\n".join([",".join(map(str,line)) for line in props]))
    pf.close()
    print("Wrote to /homeappl/home/sunijona/jets/"+runid+"/"+str(filenr)+"."+key+".props")

def jio_figmake(runid,start,jetid,figname):

    props = calc_jet_properties(runid,start,jetid)

    jetsize_fig(runid,start,jetid,figname=figname,props_arr=props)

def figmake_script(runid,start,ids):

    for ID in ids:
        jio_figmake(runid,start,ID,figname=ID)

def jetsize_fig(runid,start,jetid,figsize=(10,12),figname="sizefig",props_arr=None):
    # script for creating time series of jet linear sizes and area

    if props_arr == None:
        linsizes = pd.read_csv("/homeappl/home/sunijona/jets/"+runid+"/"+str(start)+"."+jetid+".props").as_matrix()
    else:
        linsizes = props_arr

    time_arr = linsizes[:,0]
    area_arr = linsizes[:,4]
    rad_size_arr = linsizes[:,8]
    tan_size_arr = linsizes[:,9]
    r_arr = linsizes[:,7]

    tmin,tmax=min(time_arr),max(time_arr)
    Amin,Amax=min(area_arr),max(area_arr)
    rsmin,rsmax=min(rad_size_arr),max(rad_size_arr)
    psmin,psmax=min(tan_size_arr),max(tan_size_arr)
    rmin,rmax=min(r_arr),max(r_arr)

    plt.ion()
    fig = plt.figure(figsize=figsize)

    area_ax = fig.add_subplot(411)
    rad_size_ax = fig.add_subplot(412)
    tan_size_ax = fig.add_subplot(413)
    r_ax = fig.add_subplot(414)

    area_ax.grid()
    rad_size_ax.grid()
    tan_size_ax.grid()
    r_ax.grid()

    area_ax.set_xlim(tmin,tmax)
    rad_size_ax.set_xlim(tmin,tmax)
    tan_size_ax.set_xlim(tmin,tmax)
    r_ax.set_xlim(tmin,tmax)

    area_ax.set_ylim(Amin,Amax)
    rad_size_ax.set_ylim(rsmin,rsmax)
    tan_size_ax.set_ylim(psmin,psmax)
    r_ax.set_ylim(rmin,rmax)

    #area_ax.set_yticks([0.5,1,1.5,2,2.5])
    #rad_size_ax.set_yticks([0.8,1.2,1.6,2,2.4,2.8])
    #tan_size_ax.set_yticks([0.6,1,1.4])

    #area_ax.set_xticks([295,300,305,310,315,320])
    #rad_size_ax.set_xticks([295,300,305,310,315,320])
    #tan_size_ax.set_xticks([295,300,305,310,315,320])

    area_ax.set_xticklabels([])
    rad_size_ax.set_xticklabels([])
    tan_size_ax.set_xticklabels([])

    area_ax.set_ylabel("Area [R$_{e}^{2}$]",fontsize=20)
    rad_size_ax.set_ylabel("Radial size [R$_{e}$]",fontsize=20)
    tan_size_ax.set_ylabel("Tangential size [R$_{e}$]",fontsize=20)
    r_ax.set_ylabel("Radial distance [R$_{e}$]",fontsize=20)
    r_ax.set_xlabel("Time [s]",fontsize=20)

    area_ax.tick_params(labelsize=16)
    rad_size_ax.tick_params(labelsize=16)
    tan_size_ax.tick_params(labelsize=16)
    r_ax.tick_params(labelsize=16)

    area_ax.plot(time_arr,area_arr,color="black",linewidth=2)
    rad_size_ax.plot(time_arr,rad_size_arr,color="black",linewidth=2)
    tan_size_ax.plot(time_arr,tan_size_arr,color="black",linewidth=2)
    r_ax.plot(time_arr,r_arr,color="black",linewidth=2)

    plt.tight_layout()

    fig.show()

    plt.savefig("jet_sizes/"+figname+".png")
    print("jet_sizes/"+figname+".png")

    return None

def calc_jet_properties(runid,start,jetid):

    jet_list = jetfile_read(runid,start,jetid)
    time_list = timefile_read(runid,start,jetid)

    dt = (np.pad(np.array(time_list),(0,1),"constant")-np.pad(np.array(time_list),(1,0),"constant"))[1:-1]

    if max(dt) > 5:
        print("Jet not sufficiently continuous, exiting.")
        return 1

    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    nr_list = [int(t*2) for t in time_list]

    prop_arr = np.array([])

    for n in xrange(len(nr_list)):

        if runid == "AED":
            bulkname = "bulk.old."+str(nr_list[n]).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(nr_list[n]).zfill(7)+".vlsv"

        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

        # read variables
        if vlsvobj.check_variable("X"):
            X = vlsvobj.read_variable("X",cellids=jet_list[n])
            Y = vlsvobj.read_variable("Y",cellids=jet_list[n])
            Z = vlsvobj.read_variable("Z",cellids=jet_list[n])
        else:
            sorigid = vlsvobj.read_variable("CellID")
            sorigid = sorigid[sorigid.argsort()]

            X,Y,Z = ja.ci2vars_nofile(ja.xyz_reconstruct(vlsvobj),sorigid,jet_list[n])

        if n == 0 and vlsvobj.check_variable("DX"):
            dA = vlsvobj.read_variable("DX")[0]*vlsvobj.read_variable("DY")[0]
        elif n == 0 and not vlsvobj.check_variable("DX"):
            dA = ja.get_cell_area(vlsvobj)

        # calculate geometric center of jet
        x_mean = np.mean([max(X),min(X)])/r_e
        y_mean = np.mean([max(Y),min(Y)])/r_e
        z_mean = np.mean([max(Z),min(Z)])/r_e

        # calculate jet size
        A = dA*len(jet_list[n])/(r_e**2)
        Nr_cells = len(jet_list[n])

        # geometric center of jet in polar coordinates
        phi = np.rad2deg(np.arctan((y_mean+z_mean)/x_mean))
        r_d = np.linalg.norm([x_mean,y_mean,z_mean])

        # r-coordinates corresponding to all (x,y)-points in jet
        r = np.linalg.norm(np.array([X,Y,Z]),axis=0)/r_e

        # calculate linear sizes of jet
        size_rad = max(r)-min(r)
        size_tan = A/size_rad

        time = time_list[n]

        # "time [s],x_mean [R_e],y_mean [R_e],z_mean [R_e],A [R_e^2],Nr_cells,phi [deg],r_d [R_e],size_rad [R_e],size_tan [R_e]"
        temp_arr = [time,x_mean,y_mean,z_mean,A,Nr_cells,phi,r_d,size_rad,size_tan]

        prop_arr = np.append(prop_arr,np.array(temp_arr))

    prop_arr = np.reshape(prop_arr,(len(nr_list),len(temp_arr)))

    propfile_write(runid,start,jetid,prop_arr)

    return prop_arr

def track_jets(runid,start,stop,threshold=0.6):

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    if runid == "AED":
        bulkname = "bulk.old."+str(start).zfill(7)+".vlsv"
    else:
        bulkname = "bulk."+str(start).zfill(7)+".vlsv"

    if not os.path.exists("/homeappl/home/sunijona/jets/"+runid):
        os.makedirs("/homeappl/home/sunijona/jets/"+runid)

    vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

    events_old = eventfile_read(runid,start)
    events = eventfile_read(runid,start+1)

    # remove non-bow shock events from events_old

    bs_events = []

    for old_event in events_old:

        X,Y = ja.ci2vars(vlsvobj,["X","Y"],old_event)

        r = np.linalg.norm([X,Y],axis=0)

        if max(r)/r_e > 10:

            bs_events.append(old_event)

    jet_dict = dict()

    jetobj_list = []

    counter = 1

    for event in events:

        for bs_event in bs_events:

            if np.intersect1d(bs_event,event).size > threshold*len(event):

                curr_id = str(counter).zfill(5)

                jetobj_list.append(Jet(curr_id,runid,float(start)/2))

                jetobj_list[-1].cellids.append(bs_event)
                jetobj_list[-1].cellids.append(event)
                jetobj_list[-1].times.append(float(start+1)/2)

                counter += 1

                break

    for n in xrange(start+2,stop+1):

        flags = []

        events = eventfile_read(runid,n)

        for event in events:

            for jetobj in jetobj_list:

                if np.intersect1d(jetobj.cellids[-1],event).size > threshold*len(event):

                    if jetobj.ID in flags:

                        jetobj_new = copy.deepcopy(jetobj)
                        jetobj_new.ID = str(counter).zfill(5)
                        print("Created jet with ID "+jetobj_new.ID)
                        jetobj_new.cellids = jetobj_new.cellids[:-1]
                        jetobj_new.cellids.append(event)
                        jetobj_new.times = jetobj_new.times[:-1]
                        jetobj_new.times.append(float(n)/2)

                        jetobj_list.append(jetobj_new)

                        counter += 1

                        break

                    else:

                        jetobj.cellids.append(event)
                        jetobj.times.append(float(n)/2)

                        flags.append(jetobj.ID)

                        break

        #for jetobj in jetobj_list:

        #    if jetobj.ID not in flags:

        #        jetobj_list.remove(jetobj)

    for jetobj in jetobj_list:

        jetfile = open("/homeappl/home/sunijona/jets/"+jetobj.runid+"/"+str(start)+"."+jetobj.ID+".jet","w")
        timefile = open("/homeappl/home/sunijona/jets/"+jetobj.runid+"/"+str(start)+"."+jetobj.ID+".times","w")

        jetfile.write(jetobj.return_cellid_string())
        timefile.write(jetobj.return_time_string())

        jetfile.close()
        timefile.close()

    return None