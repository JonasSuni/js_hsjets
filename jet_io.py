import numpy as numpy
import pytools as pt
import scipy
import pandas as pandas
import jet_scripts as js
import jet_analyser as ja
import jetfile_make as jfm
import os

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

        jets = ja.sort_jets(vlsvobj,msk,100,3000,[1,1])

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

        outputlist.append(map(int,line))

    return outputlist

def eventfile_read(runid,filenr):

    outputlist = []

    ef = open("/homeappl/home/sunijona/events/"+runid+"/"+str(filenr)+".events","r")
    contents = ef.read()
    lines = contents.split("\n")[:-1]

    for line in lines:

        outputlist.append(map(int,line))

    return outputlist

def track_jets(runid,start,stop):

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

    events_old = eventfile_read(runid,start) #PH events in file start
    events = eventfile_read(runid,start+1) #PH events in file start+1

    # remove non-bow shock events from events_old

    bs_events = []

    for old_event in events_old:

        X,Y = ci2vars(vlsvobj,["X","Y"],old_event)

        r = np.linalg.norm([X,Y],axis=0)

        if max(r)/r_e > 11:

            bs_events.append(old_event)

    jet_dict = dict()

    counter = 1

    for event in events:

        for bs_event in bs_events:

            if np.intersect1d(bs_event,event).size > 0.6*len(event):

                curr_id = str(counter).zfill(5)

                jet_dict[curr_id] = event

                jetfile_write(runid,start,curr_id,bs_event) # write bs_event to file
                jetfile_write(runid,start,curr_id,event)# write event to file
                timefile_write(runid,start,curr_id,float(start)/2)# write times to file
                timefile_write(runid,start,curr_id,float(start+1)/2)

                counter += 1

                break

    for n in xrange(start+2,stop+1):

        events = eventfile_read(runid,n) #PH events in file n

        for k in jet_dict.keys():

            key_del = True

            for event in events:

                if np.intersect1d(jet_dict[k],event) > 0.6*len(event):

                    jet_dict[k] = event

                    jetfile_write(runid,start,k,event) #write event to file
                    timefile_write(runid,start,k,float(n)/2)#write time to file

                    key_del = False

                    break

            if key_del:

                jet_dict.pop(k)

    return None