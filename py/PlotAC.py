import numpy as np
import pylab as plt

basefolder = "/homecentral/srao/Documents/code/mypybox"
import code, sys, os
import matplotlib as mpl
mpl.use('Agg')
sys.path.append(basefolder)
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from Print2Pdf import Print2Pdf

def ProcessFigure(figHdl, filepath, IF_SAVE, IF_XTICK_INT = True, figFormat = 'eps', paperSize = [4, 3], titleSize = 10, axPosition = [0.25, 0.25, .65, .65], tickFontsize = 10, labelFontsize = 12, nDecimalsX = 1, nDecimalsY = 1):
    # FixAxisLimits(figHdl)
    # FixAxisLimits(plt.gcf(), IF_XTICK_INT, nDecimalsX, nDecimalsY)
    Print2Pdf(plt.gcf(), filepath, paperSize, figFormat=figFormat, labelFontsize = labelFontsize, tickFontsize=tickFontsize, titleSize = titleSize, IF_ADJUST_POSITION = True, axPosition = axPosition)
    plt.show()

def FixAxisLimits(fig, IF_XTICK_INT = True, nDecimalsX = 1, nDecimalsY = 1):
    ax = fig.axes[0]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.' + '%s'%(int(nDecimalsX)) + 'f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.' + '%s'%(int(nDecimalsY)) + 'f'))
    xmiddle = 0.5 * (xmin + xmax)
    xticks = [xmin, xmiddle, xmax]
    if IF_XTICK_INT:
	if xmiddle != int(xmiddle):
	    xticks = [xmin, xmax]
	ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))	
    ax.set_xticks(xticks)
    ax.set_yticks([ymin, 0.5 *(ymin + ymax), ymax])
    plt.draw()



def PlotAc(pList, kList, tStop = 500, tau_syn = 3, N = 20000):
    fg0, ax0 = plt.subplots()
    fg1, ax1 = plt.subplots()
    for K in kList:
	print 'K = ', K
	for p in pList:
	    filebase = '/homecentral/srao/cuda/data/pub/bidir/i2i/tau3/p%s/K%s/'%(p, int(K))
	    fr = np.loadtxt(filebase + 'firingrates_xi0.8_theta0_0.80_3.0_cntrst100.0_100000_tr0.csv')
	    fre = fr[:N]
	    fri = fr[N:]
	    tag = 'NI2E4' + '_K%s'%(K)
	    filetag = 'bidir' + tag + '_tau%s_p%s'%(tau_syn, p)    
	    saveFolder = './data/'
	    savedFilename = 'long_tau_vs_ac_mat_tr1_' + filetag + '.npy'
	    ac = np.load(saveFolder + savedFilename)
	    ax0.plot(np.squeeze(ac[:tStop, 0, 0]) * fre.mean() / np.mean(fre**2), linewidth = 0.5)
	    ax1.plot(np.squeeze(ac[:tStop, 1, 0]) * fri.mean() / np.mean(fri**2), linewidth = 0.5)
    plt.figure(fg0.number)
    plt.ylabel('AC(Hz)')
    ########## PRINT FIGURE ##########
    paperSize = [4.0, 3.0]
    axPosition=[.26, .28, .65, .65]
    filename = './figs/ac_vs_K_E_p8'
    ProcessFigure(fg0, filename, 1, paperSize = paperSize, axPosition = axPosition, titleSize=10, nDecimalsX=0, nDecimalsY=0, figFormat='svg', labelFontsize = 10, tickFontsize = 8)
    filename = './figs/ac_vs_K_I_p8'

    plt.figure(fg1.number)
    plt.xlabel('Time lag(ms)')
    ProcessFigure(fg1, filename, 1, paperSize = paperSize, axPosition = axPosition, titleSize=10, nDecimalsX=0, nDecimalsY=0, figFormat='svg', labelFontsize = 10, tickFontsize = 8)
    
    plt.show()
    
    
def PlotFrDistr(pList, kList, tStop = 500, tau_syn = 3, N = 20000):
    fg0, ax0 = plt.subplots()
    fg1, ax1 = plt.subplots()
    for K in kList:
	print 'K = ', K
	for p in pList:
	    filebase = '/homecentral/srao/cuda/data/pub/bidir/i2i/tau3/p%s/K%s/'%(p, int(K))
	    fr = np.loadtxt(filebase + 'firingrates_xi0.8_theta0_0.80_3.0_cntrst100.0_100000_tr0.csv')
	    fre = fr[:N]
	    fri = fr[N:]
	    ax0.hist(np.log10(fre[fre>0]), 16, histtype = 'step')
	    ax1.hist(np.log10(fri[fri>0]), 16, histtype = 'step')
    plt.show()

def PlotMeanRates(pList, kList, tStop = 500, tau_syn = 3, N = 20000):
    fre = []
    fri = []
    for K in kList:
	print 'K = ', K
	for p in pList:
	    filebase = '/homecentral/srao/cuda/data/pub/bidir/i2i/tau3/p%s/K%s/'%(p, int(K))
	    fr = np.loadtxt(filebase + 'firingrates_xi0.8_theta0_0.80_3.0_cntrst100.0_100000_tr0.csv')
	    fre.append(fr[:N].mean())
	    fri.append(fr[N:].mean())

    print fre
    plt.plot(kList, fre, 'k.-', label = 'E', linewidth = 0.4)
    plt.plot(kList, fri, 'r.-', label = 'I', linewidth = 0.4)

    plt.ylim([0, 16])
    plt.xlim([kList[0] - 100, kList[-1] + 100])
    plt.gca().set_xticks(kList)
    plt.gca().set_yticks([0, 8, 16])    
    plt.legend(loc = 0, frameon = False, numpoints = 1, prop = {'size': 6})
    plt.xlabel('K')
    plt.ylabel('Mean rate(Hz)')
    

    paperSize = [2.0, 1.5]
    axPosition=[.26, .25, .65, .65]

    filename = './figs/fr_vs_K_p8'
    ProcessFigure(plt.gcf(), filename, 1, paperSize = paperSize, axPosition = axPosition, titleSize=10, nDecimalsX=0, nDecimalsY=0, figFormat='svg', labelFontsize = 10, tickFontsize = 8)

    plt.show()
