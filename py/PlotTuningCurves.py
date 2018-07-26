import numpy as np
import pylab as plt
import sys
basefolder = "/homecentral/srao/Documents/code/mypybox"
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
import SetAxisProperties as AxPropSet
from Print2Pdf import Print2Pdf

def GetAvgTunningCurve(y):
    # returns avg tuning curve over the population
    nNeurons, nTheta = y.shape
    out = np.zeros((nTheta, ))
    for kNeuron in range(nNeurons):
        tc = y[kNeuron, :]
        maxIdx = np.argmax(tc)
        if maxIdx == 0:
            tc = np.roll(tc, 1)
        maxIdx = np.argmax(tc)
        tc = np.roll(tc, maxIdx * -1 + 4)
        out = out + tc / nNeurons
    return out
    
def PltTuningCurve(ax, tc, color):
    maxIdx = np.argmax(tc)
    if maxIdx == 0:
        tc = np.roll(tc, 1)
    maxIdx = np.argmax(tc)
    tc = np.roll(tc, maxIdx * -1 + 4)
    tc = np.concatenate((tc, [tc[0]]))
    theta = np.linspace(0, 180, tc.size) - 90    
    ax.plot(theta, tc / np.max(tc), 'o-', color = color, linewidth = 0.2, markeredgecolor = color, markersize = 2.0, mfc = 'none', markeredgewidth = .3)

#---------------------------------------------------------------#  

if __name__ == "__main__":
    alpha = [0, 2, 5, 8]
    tau = 3
    IF_GROUP_CELL_TYPE = True
    simDuration = 100000
    plotNCells = 0
    NE = 20000
    bidirType = sys.argv[1]
    paperSize = [2.0, 1.5]
    axPosition = [.26, .28, .65, .65]
    figFormat = 'eps'
    for k, kAlpha in enumerate(alpha):
        dbName = 'bidir%srho5_xi0.8_kff800_p%s'%(bidirType.upper(), kAlpha)
        figBasefolder = '/homecentral/srao/Documents/code/tmp/'
        figFolder = figBasefolder + 'figs/publication_figures/ac/%s/tuningCurves/p%st%s/'%(bidirType, kAlpha, tau)
        if kAlpha == 0:
            dataFolder = '/homecentral/srao/cuda/data/pub/bidir/p%s/'%(kAlpha)
            dbName = 'bidirI2Irho5_xi0.8_kff800_p0'
        else:
            if tau == 3:
                dataFolder = '/homecentral/srao/cuda/data/pub/bidir/%s/p%s/'%(bidirType, kAlpha)
            else:
                dataFolder = '/homecentral/srao/cuda/data/pub/bidir/%s/tau%s/p%s/'%(bidirType, tau, kAlpha)
        print dataFolder
        frFilename = 'firingrates_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_%s_tr0.csv'%(kAlpha, tau, simDuration)
        fr = np.loadtxt(dataFolder + frFilename)
        print 'loading tc file --> ', frFilename
        freIdx = np.argsort(fr[:NE])
        friIdx = np.argsort(fr[NE:])
        tcFileName = '/homecentral/srao/db/data/tuningCurves_%s.npy'%(dbName)
        print 'loading tc file --> ', tcFileName
        print freIdx[-1], friIdx[-1]
        tc = np.load(tcFileName)
        tce = tc[:NE, :]
        tci = tc[NE:, :]
        avgTcE = GetAvgTunningCurve(tce)
        avgTcI = GetAvgTunningCurve(tci)
        if not IF_GROUP_CELL_TYPE:
            fg, ax = plt.subplots()
            PltTuningCurve(ax, avgTcE, 'k')
            PltTuningCurve(ax, avgTcI, 'r')
            AxPropSet.SetProperties(ax, [-90, 90], [0, 1], 'Stimulus(deg)', '')
            ax.set_ylim([0, 1.2])
            ax.set_xlim([-96, 96])        
            figname = 'pop_avg_tc_E_I_p%s_tau%s'%(kAlpha, tau)
            print 'saving in folder --> ', figFolder + figname
            Print2Pdf(fg, figFolder + figname,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
            plt.close('all')
        else:
            if k == 0:
                colors = ['b', 'g', 'r','c']
                fgE, axE = plt.subplots()
                fgI, axI = plt.subplots()            
            PltTuningCurve(axE, avgTcE, colors[k])
            PltTuningCurve(axI, avgTcI, colors[k])
            AxPropSet.SetProperties(axE, [-90, 90], [0, 1], 'Stimulus(deg)', '')
            axE.set_ylim([0, 1.2])
            axE.set_xlim([-96, 96])        
            AxPropSet.SetProperties(axI, [-90, 90], [0, 1], 'Stimulus(deg)', '')            
            axI.set_ylim([0, 1.2])
            axI.set_xlim([-96, 96])
    if IF_GROUP_CELL_TYPE:
        fignameE = 'pop_avg_tc_E_summary'
        fignameI = 'pop_avg_tc_I_summary'
        figFolder = figBasefolder + 'figs/publication_figures/ac/%s/tuningCurves/'%(bidirType)        
        print 'saving in folder --> ', figFolder
        Print2Pdf(fgE, figFolder + fignameE,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)            
        Print2Pdf(fgI, figFolder + fignameI,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
    
    #     for i in range(plotNCells * 2 + 1):
    #         if i > 0:
    #             fg, ax = plt.subplots()
    #             PltTuningCurve(ax, tce[freIdx[-1 * i], :], 'k')
    #             # figname = 'tc_E_%s'%(freIdx[-1 * i])
    #             # AxPropSet.SetProperties(ax, [-90, 90], [0, 1], '', '')
    #             # if i == plotNCells *2 - 1:
    #             #     AxPropSet.SetProperties(ax, [-90, 90], [0, 1], 'Stimulus(deg)', '')
    #             # ax.set_ylim([0, 1.2])
    #             # ax.set_xlim([-94, 95])                            
    #             # Print2Pdf(fg, figFolder + figname,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
    #             # plt.close('all')
    #             # fg, ax = plt.subplots()
    #             PltTuningCurve(ax, tci[friIdx[-1 * i], :], 'r')
    #             AxPropSet.SetProperties(ax, [-90, 90], [0, 1], 'Stimulus(deg)', '')
    #             # if i == plotNCells *2 - 1:
    #             #     AxPropSet.SetProperties(ax, [-90, 90], [0, 1], 'Stimulus(deg)', '')
    #             ax.set_ylim([0, 1.2])
    #             ax.set_xlim([-96, 96])
    # #            figname = 'tc_I_%s'%(friIdx[-1 * i])
    #             figname = 'tc_E_I_%s_%s'%(freIdx[-1 * i], friIdx[-1 * i])
    #             Print2Pdf(fg, figFolder + figname,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
    #             plt.close('all')
    # #            print freIdx[-1 * i], friIdx[-1 * i]
