import pytools as pt
import sys, os, socket
import numpy as np
import pandas as pd
import scipy.ndimage

r_e = 6.371e+6

def jc_plaschke(ax,XmeshXY,YmeshXY,extmaps):
    # extmaps consists of [npdynx,nrho,p_crit]

    npdynx = extmaps[0]
    nrho = extmaps[1]

    npdynx = scipy.ndimage.zoom(npdynx, 3)
    nrho = scipy.ndimage.zoom(nrho, 3)
    XmeshXY = scipy.ndimage.zoom(XmeshXY, 3)
    YmeshXY = scipy.ndimage.zoom(YmeshXY, 3)

    # colours to use
    color_plaschke = '#000000'

    # thresholds
    level_plaschke = 0.25
    level_sw = 2

    # mask plaschke but not solar wind
    jet = np.ma.masked_greater(npdynx,level_plaschke)
    #jet.mask[nrho < level_sw] = False
    jet.fill_value = 0
    jet[jet.mask == False] = 1

    jet2 = np.ma.masked_greater(npdynx,0.5)
    jet2.fill_value = 0
    jet2[jet2.mask == False] = 1

    # draw contours

    contour_plaschke = ax.contour(XmeshXY,YmeshXY,jet.filled(),[0.5],linewidths=1.0, colors=color_plaschke)

    contour_plaschke2 = ax.contour(XmeshXY,YmeshXY,jet2.filled(),[0.5],linewidths=1.0, colors="white")

    return None


def jc_archerhorbury(ax,XmeshXY,YmeshXY,extmaps):
    # extmaps consists of [tapdyn]

    tapdyn = extmaps[0]

    tapdyn = scipy.ndimage.zoom(tapdyn, 3)
    XmeshXY = scipy.ndimage.zoom(XmeshXY, 3)
    YmeshXY = scipy.ndimage.zoom(YmeshXY, 3)

    # colours to use
    color_archerhorbury = '#000000'

    # thresholds
    level_archerhorbury = 2
    
    # mask archer
    jet = np.ma.masked_greater(tapdyn,level_archerhorbury)
    jet.fill_value = 0
    jet[jet.mask == False] = 1

    # draw contours

    contour_archer = ax.contour(XmeshXY,YmeshXY,jet.filled(),[0.5],linewidths=1.0, colors=color_archerhorbury)

    return None    


def jc_karlsson(ax,XmeshXY,YmeshXY,extmaps):
    # extmaps consists of [tarho]

    tarho = extmaps[0]
    
    tarho = scipy.ndimage.zoom(tarho, 3)
    XmeshXY = scipy.ndimage.zoom(XmeshXY, 3)
    YmeshXY = scipy.ndimage.zoom(YmeshXY, 3)

    # colours to use
    color_karlsson = '#000000'

    # thresholds
    level_karlsson = 1.5

    # mask karlsson
    jet = np.ma.masked_greater(tarho,level_karlsson)
    jet.fill_value = 0
    jet[jet.mask == False] = 1

    # draw contours

    contour_karlsson = ax.contour(XmeshXY,YmeshXY,jet.filled(),[0.5],linewidths=1.0, colors=color_karlsson)


def jc_all(ax,XmeshXY,YmeshXY,extmaps):
    # extmaps consists of [npdynx,nrho,tapdyn,tarho]

    npdynx,nrho,tapdyn,tarho = extmaps[0],extmaps[1],extmaps[2],extmaps[3]

    npdynx = scipy.ndimage.zoom(npdynx, 3)
    nrho = scipy.ndimage.zoom(nrho, 3)
    tapdyn = scipy.ndimage.zoom(tapdyn, 3)
    tarho = scipy.ndimage.zoom(tarho, 3)
    XmeshXY = scipy.ndimage.zoom(XmeshXY, 3)
    YmeshXY = scipy.ndimage.zoom(YmeshXY, 3)

    # thresholds
    level_plaschke = 0.25
    level_sw = 2
    level_archerhorbury = 2
    level_karlsson = 1.5

    # make plaschke mask
    jetp = np.ma.masked_greater(npdynx,level_plaschke)
    #jetp.mask[nrho < level_sw] = False
    jetp.fill_value = 0
    jetp[jetp.mask == False] = 1

    # make archer&horbury mask
    jetah = np.ma.masked_greater(tapdyn,level_archerhorbury)
    jetah.fill_value = 0
    jetah[jetah.mask == False] = 1

    # make karlsson mask
    jetk = np.ma.masked_greater(tarho,level_karlsson)
    jetk.fill_value = 0
    jetk[jetk.mask == False] = 1

    # draw contours
    contour_plaschke = ax.contour(XmeshXY,YmeshXY,jetp.filled(),[0.5],linewidths=1.0, colors="black",label="Plaschke 0.25")

    contour_archer = ax.contour(XmeshXY,YmeshXY,jetah.filled(),[0.5],linewidths=1.0, colors="blue",label="ArcherHorbury")

    contour_karlsson = ax.contour(XmeshXY,YmeshXY,jetk.filled(),[0.5],linewidths=1.0, colors="magenta",label="Karlsson")

    return None


def jc_all_cust(ax,XmeshXY,YmeshXY,extmaps):
    # extmaps consists of [npdynx,nrho,tapdyn,tarho,identifiers]

    npdynx,nrho,tapdyn,tarho = extmaps[0],extmaps[1],extmaps[2],extmaps[3]

    npdynx = scipy.ndimage.zoom(npdynx, 3)
    nrho = scipy.ndimage.zoom(nrho, 3)
    tapdyn = scipy.ndimage.zoom(tapdyn, 3)
    tarho = scipy.ndimage.zoom(tarho, 3)
    XmeshXY = scipy.ndimage.zoom(XmeshXY, 3)
    YmeshXY = scipy.ndimage.zoom(YmeshXY, 3)

    identifiers = extmaps[4]
    identifiers = identifiers[0]
    ids = []

    for value in identifiers:
        for n in value:
            ids.append(n)

    runid = "".join(map(chr,ids[0:3]))
    file_nr_1 = ids[3]
    halftimewidth_1 = ids[4]

    props = pd.read_csv("Props/"+runid+"/props_"+runid+"_"+str(file_nr_1)+"_"+str(halftimewidth_1)+".csv").as_matrix()

    x = props[:,12]
    y = props[:,13]

    # thresholds
    level_plaschke = 0.25
    level_sw = 3.0
    level_archerhorbury = 2
    level_karlsson = 1.5

    # make plaschke mask
    jetp = np.ma.masked_greater(npdynx,level_plaschke)
    jetp.mask[nrho < level_sw] = False
    jetp.fill_value = 0
    jetp[jetp.mask == False] = 1

    # make archer&horbury mask
    jetah = np.ma.masked_greater(tapdyn,level_archerhorbury)
    jetah.fill_value = 0
    jetah[jetah.mask == False] = 1

    # make karlsson mask
    jetk = np.ma.masked_greater(tarho,level_karlsson)
    jetk.fill_value = 0
    jetk[jetk.mask == False] = 1

    # make custom mask
    jet_cust = jetah
    jet_cust.mask = np.logical_or(jet_cust.mask,jetp.mask)

    # draw contours
    contour_cust = ax.contour(XmeshXY,YmeshXY,jet_cust.filled(),[0.5],linewidths=1.0, colors="black",label="CUSTOM")

    ax.plot(x,y,"x",color="red")


    return None

def jc_cust_new(ax,XmeshXY,YmeshXY,extmaps):
    # extmaps consists of [npdynx,nrho,tapdyn,tarho,tpdynavg,trhoavg,srho,CellID]

    npdynx,nrho,tapdyn,tarho,tpdynavg,trhoavg,srho,cellids = extmaps[0],extmaps[1],extmaps[2],extmaps[3],extmaps[4],extmaps[5],extmaps[6],extmaps[7]

    srho_ref = trhoavg[np.where(cellids==1339391)][0]

    npdynx = scipy.ndimage.zoom(npdynx, 3)
    nrho = scipy.ndimage.zoom(nrho, 3)
    tapdyn = scipy.ndimage.zoom(tapdyn, 3)
    tarho = scipy.ndimage.zoom(tarho, 3)
    tpdynavg = scipy.ndimage.zoom(tpdynavg, 3)
    trhoavg = scipy.ndimage.zoom(trhoavg, 3)
    srho = scipy.ndimage.zoom(srho, 3)
    XmeshXY = scipy.ndimage.zoom(XmeshXY, 3)
    YmeshXY = scipy.ndimage.zoom(YmeshXY, 3)

    jet = np.ma.masked_greater(npdynx,0.25)
    jet.mask[srho < 3*srho_ref] = False
    jet.mask[tapdyn > 2] = True
    jet.fill_value = 0
    jet[jet.mask == False] = 1

    contour_new = ax.contour(XmeshXY,YmeshXY,jet.filled(),[0.5],linewidths=1.0, colors="black")

    return None