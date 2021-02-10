import os
import numpy as np
import matplotlib.pyplot as plt

wrkdir_DNR = os.environ["WRK"]+"/"
homedir = os.environ["HOME"]+"/"

medium_blue = '#006DDB'
crimson = '#920000'
violet = '#B66DFF'
dark_blue = '#490092'
orange = '#db6d00'
green = '#24ff24'

def plot_neighbours(xmin=-10,xmax=10,ymin=-10,ymax=10):

    xflat = np.arange(xmin,xmax+1)
    yflat = np.arange(ymin,ymax+1)

    X,Y = np.meshgrid(xflat,yflat)
    cellids = np.arange(X.size)+1
    cell_mesh = np.reshape(cellids,X.shape)

    cell_jet = (np.sqrt((X+2)**2+(Y-4)**2)+np.sqrt((X-2)**2+(Y-2)**2)<8).astype(int).flatten()
    mask_cells = cellids[cell_jet==1]
    cell_marked = np.zeros_like(cell_mesh).flatten()

    jet_cells = np.array([])

    ann_locs = [(0.1,0.8),(0.1,0.8),(0.1,0.8),(0.1,0.8),(0.1,0.8),(0.1,0.8),(0.1,0.8),(0.1,0.8)]
    ann_labs = ["a)","b)","c)","d)","e)","f)","g)","h)"]

    fig,ax_list = plt.subplots(2,3,figsize=(10,10))

    ax_list[0][0].pcolormesh(xflat,yflat,np.reshape(np.in1d(cellids,jet_cells),cell_mesh.shape),zorder=0,shading="auto",vmin=0,vmax=1,cmap="Reds")
    ax_list[0][0].contour(X,Y,np.reshape(cell_jet,cell_mesh.shape),[0.5],colors="green",zorder=2,linewidths=1.2)
    ax_list[0][0].get_xaxis().set_ticks([])
    ax_list[0][0].get_yaxis().set_ticks([])
    ax_list[0][0].set_xlabel("Step 0",fontsize=24,labelpad=10)

    jet_cells = np.array(mask_cells[0])

    ax_list[0][1].pcolormesh(xflat,yflat,np.reshape(np.in1d(cellids,jet_cells),cell_mesh.shape),zorder=0,shading="auto",vmin=0,vmax=1,cmap="Reds")
    ax_list[0][1].contour(X,Y,np.reshape(cell_jet,cell_mesh.shape),[0.5],colors="green",zorder=2,linewidths=1.2)
    ax_list[0][1].get_xaxis().set_ticks([])
    ax_list[0][1].get_yaxis().set_ticks([])
    ax_list[0][1].set_xlabel("Step 1",fontsize=24,labelpad=10)

    jet_cells = get_nbrs(jet_cells,xflat.size,yflat.size,[2,2])

    ax_list[0][2].pcolormesh(xflat,yflat,np.reshape(np.in1d(cellids,jet_cells),cell_mesh.shape),zorder=0,shading="auto",vmin=0,vmax=1,cmap="Reds")
    ax_list[0][2].contour(X,Y,np.reshape(cell_jet,cell_mesh.shape),[0.5],colors="green",zorder=2,linewidths=1.2)
    ax_list[0][2].get_xaxis().set_ticks([])
    ax_list[0][2].get_yaxis().set_ticks([])
    ax_list[0][2].set_xlabel("Step 2",fontsize=24,labelpad=10)

    jet_cells = np.intersect1d(jet_cells,mask_cells)

    ax_list[1][0].pcolormesh(xflat,yflat,np.reshape(np.in1d(cellids,jet_cells),cell_mesh.shape),zorder=0,shading="auto",vmin=0,vmax=1,cmap="Reds")
    ax_list[1][0].contour(X,Y,np.reshape(cell_jet,cell_mesh.shape),[0.5],colors="green",zorder=2,linewidths=1.2)
    ax_list[1][0].get_xaxis().set_ticks([])
    ax_list[1][0].get_yaxis().set_ticks([])
    ax_list[1][0].set_xlabel("Step 2.5",fontsize=24,labelpad=10)

    jet_cells = get_nbrs(jet_cells,xflat.size,yflat.size,[2,2])

    ax_list[1][1].pcolormesh(xflat,yflat,np.reshape(np.in1d(cellids,jet_cells),cell_mesh.shape),zorder=0,shading="auto",vmin=0,vmax=1,cmap="Reds")
    ax_list[1][1].contour(X,Y,np.reshape(cell_jet,cell_mesh.shape),[0.5],colors="green",zorder=2,linewidths=1.2)
    ax_list[1][1].get_xaxis().set_ticks([])
    ax_list[1][1].get_yaxis().set_ticks([])
    ax_list[1][1].set_xlabel("Step 3",fontsize=24,labelpad=10)

    jet_cells = np.intersect1d(jet_cells,mask_cells)

    ax_list[1][2].pcolormesh(xflat,yflat,np.reshape(np.in1d(cellids,jet_cells),cell_mesh.shape),zorder=0,shading="auto",vmin=0,vmax=1,cmap="Reds")
    ax_list[1][2].contour(X,Y,np.reshape(cell_jet,cell_mesh.shape),[0.5],colors="green",zorder=2,linewidths=1.2)
    ax_list[1][2].get_xaxis().set_ticks([])
    ax_list[1][2].get_yaxis().set_ticks([])
    ax_list[1][2].set_xlabel("Step 3.5",fontsize=24,labelpad=10)

    for idx,ax in enumerate(fig.get_axes()):
        for a in np.arange(xmin+0.5,xmax,1):
            ax.axvline(a,linewidth=0.4,zorder=1)
        for b in np.arange(ymin+0.5,ymax,1):
            ax.axhline(b,linewidth=0.4,zorder=1)
        ax.set_aspect("equal",adjustable="box")
        ax.annotate(ann_labs[idx],ann_locs[idx],xycoords="axes fraction",fontsize=20)
        #ax.set_xlabel("X",fontsize=20)
        #ax.set_ylabel("Y",fontsize=20)

    plt.tight_layout()

    fig.savefig(wrkdir_DNR+"Figures/thesis/celltest.png")
    plt.close(fig)

    return None

def get_nbrs(cells,x_size,y_size,reach=[1,1]):

    cells = np.array(cells,ndmin=1)
    out_cells = np.array(cells,ndmin=1)

    for a in range(-reach[0],reach[0]+1):
        for b in range(-reach[1],reach[1]+1):
            new_cells = cells+a
            new_cells = new_cells[(new_cells-1)//x_size==(cells-1)//x_size]
            new_cells = new_cells[np.logical_and((new_cells>0),(new_cells<=x_size*y_size))]
            new_cells = new_cells+x_size*b
            new_cells = new_cells[np.logical_and((new_cells>0),(new_cells<=x_size*y_size))]
            out_cells = np.append(out_cells,new_cells)

    return np.unique(out_cells).astype(int)
