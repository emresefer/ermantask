import scipy as sp
import scipy.cluster
import numpy as np
import math
import xlrd
import sys
import os
import argparse
from xlrd.sheet import ctype_text
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def transData(data):
    """transform categorical vars to numeric vars
    Args:
       data:
    Returns:
       moddata:
       cat2assign:
    """
    def findCategories(data):
        """identify categories mapping 
        """
        categindices = set()
        for rowdata in data:
            for colind,item in enumerate(rowdata):
                if type(item) != float and type(item) != int:
                   categindices.add(colind)
        cat2values = {catind:set() for catind in categindices}
        for rowdata in data:
            for catind in categindices:
                cat2values[catind].add(rowdata[catind])
        cat2assign = {catind:{} for catind in categindices}
        for catind in cat2values.keys():
            for cind,catval in enumerate(cat2values[catind]):
                cat2assign[catind][catval] = cind
        return cat2assign 
                
    cat2assign = findCategories(data)      
    moddata = []
    for rowdata in data:
        newrow = [cat2assign[cind][item] if cat2assign.has_key(cind) else item for cind,item in enumerate(rowdata)]
        moddata.append(list(newrow))
    return moddata,cat2assign


def readFile(fname):
    """read xls input file
    """ 
    xl_workbook = xlrd.open_workbook(fname)
    sheet_names = xl_workbook.sheet_names()
    xl_sheet = xl_workbook.sheet_by_name(sheet_names[0])    
    row = xl_sheet.row(0)
    fields = [cell_obj.value for idx, cell_obj in enumerate(row)]

    ind2userid = {}    
    data = []
    num_cols = xl_sheet.ncols   # Number of columns
    for row_idx in range(1, xl_sheet.nrows):    # Iterate through rows
        rowdata = []
        ind2userid[row_idx-1] = xl_sheet.cell(row_idx, 0).value
        for col_idx in xrange(1,num_cols):  # Iterate through columns
            cell_obj = xl_sheet.cell(row_idx, col_idx)  # Get cell object by row, col
            rowdata.append(cell_obj.value)
        data.append(list(rowdata))  
    return data,ind2userid,fields


def findClust(moddata,clustnum):
    """finds clusters
    Args:
       moddata:
       clustnum:
    Returns:
       labeldict:
       centroid:
    """
    centroid,distort = scipy.cluster.vq.kmeans(moddata, clustnum, iter=100)
    labeldict = {}
    for tind,tyval in enumerate(moddata):
        mindist,minlabel = 10000000000.0, None
        for cind,curcent in enumerate(centroid):
            curdist = math.sqrt(sum([myval**2 for myval in curcent - tyval]))
            if curdist < mindist:
               mindist = curdist
               minlabel = cind
        labeldict[tind] = minlabel
    return labeldict,centroid,distort


def writeCluster(fpath,fields,itlist,ind2userid):
    """write cluster to fpath
    Args:
       fpath:
       fields,itlist,ind2userid:
    """
    with open(fpath,"w") as outfile:
       outfile.write("\t".join(fields)+"\n")
       for item in itlist:
           outrow = [ind2userid[item]]
           outrow.extend(moddata[item])
           outfile.write("\t".join([str(item) for item in outrow])+"\n")

           
def plotClustError(outxvals,outyvals,plotpath):
    """plot cluster error
    Args:
       outxvals,outyvals:
       plotpath:
    Returns:   
    """
    plt.clf()
    plt.rc('font', family='serif', size=30)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(13,10)
    FSIZE = 30
    YFSIZE = 30
    LEGENDSIZE = 30 
    MARKERSIZE = 25
    DPI = 300
    ymax,ymin = max(outyvals), min(outyvals)
    plt.ylim(ymin*0.8,ymax*1.05)
    locpos = 1
    plt.xlabel("Cluster number",fontsize=FSIZE)
    plt.ylabel("Error",fontsize=YFSIZE)
    plt.xlim(min(outxvals)-0.5,max(outxvals)+0.5)
     
    plt.plot(outxvals,outyvals,marker="s",markersize=MARKERSIZE,linestyle='None',color="r")
    ax = plt.axes()        
    ax.xaxis.grid()        
    plt.legend(loc=locpos,prop={'size':LEGENDSIZE})
    plt.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.13)    
    plt.savefig(plotpath, dpi=DPI)  
           

def getParser():
    """get command line parser
    """
    parser = argparse.ArgumentParser(description='Analyze clussters')
    parser.add_argument("-o", "--clustoutdir", type=str, default="clusters", help="output directory")
    parser.add_argument("-i", "--fname", type=str, default="Segmentation Machine Learning.xlsx", help="input file in xlsx format")
    parser.add_argument("-c", "--clustnum", type=int, default=8, help="number of clusters. Default is run for 8 clusters")
    return parser


parser = getParser()
args = parser.parse_args()
clustoutdir = args.clustoutdir
if not os.path.exists(clustoutdir):
   os.makedirs(clustoutdir) 
fname = args.fname
clustnum = args.clustnum
data,ind2userid,fields = readFile(fname)
moddata,cat2assign = transData(data)
rowsize = len(moddata)
if clustnum != -1:
   labeldict,centroid,distort = findClust(np.array(moddata),clustnum)
   clusts = [[] for tind in xrange(clustnum)]
   for itemind in labeldict.keys():
       clusts[labeldict[itemind]].append(itemind)
   for cind,itlist in enumerate(clusts):
       fpath = "{0}/{1}.cluster".format(clustoutdir,cind)
       writeCluster(fpath,fields,itlist,ind2userid)
else:
   xvals,yvals = [],[] 
   for clustsize in xrange(3,min(int(round(2*math.sqrt(rowsize))),rowsize-5)+1):
       labeldict,centroid,distort = findClust(np.array(moddata),clustsize)
       xvals.append(clustsize)
       yvals.append(distort)
       clusts = [[] for tind in xrange(clustsize)]
       for itemind in labeldict.keys():
           clusts[labeldict[itemind]].append(itemind)
       locdir = "{0}/{1}clusters".format(clustoutdir,clustsize)
       if not os.path.exists(locdir):
          os.makedirs(locdir)     
       for cind,itlist in enumerate(clusts):
           fpath = "{0}/{1}.cluster".format(locdir,cind)
           writeCluster(fpath,fields,itlist,ind2userid)
   plotpath = "clustererror.png"        
   plotClustError(xvals,yvals,plotpath)     
        
