basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf

plt.ioff()
fg0, ax0 = plt.subplots()
fg1, ax1 = plt.subplots()

def PlotFrDistr(fr, ne, frOrCv, nBins = 50):
    fre = fr[:ne]
    fri = fr[ne:]
    if frOrCv == 'fr':
        ax0.hist(np.log10(fre[fre>0]), nBins, normed = 1, histtype = 'step', linewidth = 0.5)
        ax1.hist(np.log10(fri[fri>0]), nBins, normed = 1, histtype = 'step', linewidth = 0.5)
        #         ax1.hist(np.log10(fre[fre>0]), nBins, normed = 1, histtype = 'step', linewidth = 0.5, color = 'k')
        # ax1.hist(np.log10(fri[fri>0]), nBins, normed = 1, histtype = 'step', linewidth = 0.5, color = 'r')
    elif frOrCv == 'cv':
        # ax0.hist(fre[~np.isnan(fre)], nBins, normed = 1, histtype = 'step', linewidth = 0.5)
        # ax1.hist(fri[~np.isnan(fri)], nBins, normed = 1, histtype = 'step', linewidth = 0.5)
        ax1.hist(fre[~np.isnan(fre)], nBins, normed = 1, histtype = 'step', linewidth = 0.5, color = 'k')            
        ax1.hist(fri[~np.isnan(fri)], nBins, normed = 1, histtype = 'step', linewidth = 0.5, color = 'r')    

if __name__ == "__main__":
    NE = 20000
    frOrCv = sys.argv[1]
    bidirType = sys.argv[2]
    if bidirType == "i2i":
        filetag = "bidirI2I"
    elif bidirType == "e2e":
        filetag = "bidirE2E"
    elif bidirType == "e2i":
        filetag = "bidirE2I"
    databaseFolder = "/homecentral/srao/db/data/"
    figFolder = './figs/publication_figures/frdistr/'
    if frOrCv == 'fr':
        fignameE = 'fr_distr_E_'
        fignameI = 'fr_distr_I_'
      #  tc = np.load(databaseFolder + 'tuningCurves_cntrlrho5_xi0.8_kff800.npy')
        tc = np.load(databaseFolder + 'tuningCurves_bidirI2Irho5_xi0.8_kff800_p0.npy')        
        if bidirType == "e2i":
            tc2 = np.load(databaseFolder + 'tuningCurves_bidirE2Irho5_xi0.8_kff800_p2.npy')
            tc5 = np.load(databaseFolder + 'tuningCurves_bidirE2Irho5_xi0.8_kff800_p5.npy')
            tc8 = np.load(databaseFolder + 'tuningCurves_bidirE2Irho5_xi0.8_kff800_p8.npy')
            figFolder = figFolder + 'e2i/'
        elif bidirType == "i2i":
            tc2 = np.load(databaseFolder + 'tuningCurves_bidirI2Irho5_xi0.8_kff800_p2.npy')
            tc5 = np.load(databaseFolder + 'tuningCurves_bidirI2Irho5_xi0.8_kff800_p5.npy')
            tc8 = np.load(databaseFolder + 'tuningCurves_bidirI2Irho5_xi0.8_kff800_p8.npy')
            figFolder = figFolder + 'i2i/'
        elif bidirType == "e2e":
            tc2 = np.load(databaseFolder + 'tuningCurves_bidirE2Erho5_xi0.8_kff800_p2.npy')
            tc5 = np.load(databaseFolder + 'tuningCurves_bidirE2Erho5_xi0.8_kff800_p5.npy')
            tc8 = np.load(databaseFolder + 'tuningCurves_bidirE2Erho5_xi0.8_kff800_p8.npy')
            figFolder = figFolder + 'e2e/'
        tc = tc.mean(1)
        tc2 = tc2.mean(1)
        tc5 = tc5.mean(1)
        tc8 = tc8.mean(1)
        xlabelText = 'Log firing rates'
    elif frOrCv == 'cv':
        fignameE = 'circVar_distr_E_'
        fignameI = 'circVar_distr_I_'
#        tc = np.load(databaseFolder + 'Selectivity_cntrlrho5_xi0.8.npy')
        tc = np.load(databaseFolder + 'Selectivity_bidirI2Irho5_xi0.8_kff800_p0.npy')
        if bidirType == "e2i":
            tc2 = np.load(databaseFolder + 'Selectivity_bidirE2Erho5_xi0.8_kff800_p2.npy')
            tc5 = np.load(databaseFolder + 'Selectivity_bidirE2Erho5_xi0.8_kff800_p5.npy')
            tc8 = np.load(databaseFolder + 'Selectivity_bidirE2Erho5_xi0.8_kff800_p8.npy')
            figFolder = figFolder + 'e2i/'
        elif bidirType == "i2i":
            tc2 = np.load(databaseFolder + 'Selectivity_bidirI2Irho5_xi0.8_kff800_p2.npy')
            tc5 = np.load(databaseFolder + 'Selectivity_bidirI2Irho5_xi0.8_kff800_p5.npy')
            tc8 = np.load(databaseFolder + 'Selectivity_bidirI2Irho5_xi0.8_kff800_p8.npy')
            figFolder = figFolder + 'i2i/'
        elif bidirType == "e2e":           
            tc2 = np.load(databaseFolder + 'Selectivity_bidirE2Erho5_xi0.8_kff800_p2.npy')
            tc5 = np.load(databaseFolder + 'Selectivity_bidirE2Erho5_xi0.8_kff800_p5.npy')
            tc8 = np.load(databaseFolder + 'Selectivity_bidirE2Erho5_xi0.8_kff800_p8.npy')
            figFolder = figFolder + 'e2e/'
        xlabelText = 'Circular variance'
    PlotFrDistr(tc, NE, frOrCv)
    # PlotFrDistr(tc2, NE, frOrCv)
    # PlotFrDistr(tc5, NE, frOrCv)
    # PlotFrDistr(tc8, NE, frOrCv)
    ax0.set_xlabel(xlabelText)
    ax1.set_xlabel(xlabelText)
    ax0.set_ylabel("Probability")
    ax1.set_ylabel("Probability")
    if frOrCv == 'fr':
        ax0.set_xticks([-3, 0, 3])
        ax1.set_xticks([-3, 0, 3])
        ax0.set_yticks([0, 0.5, 1.0])
        ax1.set_yticks([0, 0.6, 1.2])
    elif frOrCv == 'cv':
        ax0.set_xlim(0, 1.0)
        ax1.set_xlim(0, 1.0)                
        ax0.set_xticks([0, 0.5, 1.0])
        ax1.set_xticks([0, 0.5, 1.0])
        ax0.set_yticks([0, 1.0, 2.0])
        ax1.set_yticks([0, 1.0, 2.0])
        if bidirType == 'e2e':
            ax1.set_yticks([0, 2.5, 5.0])
        if bidirType == 'e2i':
            ax1.set_yticks([0, 2.5, 5.0])
        if bidirType == 'i2i':
            ax0.set_yticks([0, 1.5, 3.0])
            ax1.set_yticks([0, 2.5, 5.0])
    figFormat = 'eps'
    paperSize = [2.0, 1.5]
    axPosition = [0.27, .27, .65, .65]
    Print2Pdf(fg0,  figFolder + fignameE + filetag,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
    Print2Pdf(fg1,  figFolder + fignameI + filetag,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = [0.27, .27, .65, .65])
    # plt.ion()
    # plt.show()


    
        
    
    

