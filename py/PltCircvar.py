basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
import Keyboard as kb

pltLineWidth = 0.5

def PlotCircVars(circVariance, NE, NI, axE, axI, legendLabel, nBins = 50, IF_PLOT = True):
    cvE = circVariance[:NE]
    cvI = circVariance[NE:]
    cvE = cvE[np.logical_not(np.isnan(cvE))]
    cvI = cvI[np.logical_not(np.isnan(cvI))]
    cveCnt, cvEbins = np.histogram(cvE, nBins)
    cviCnt, cvIbins = np.histogram(cvI, nBins)
    binSizeE = np.diff(cvEbins).mean()
    binSizeI = np.diff(cvIbins).mean()
#    axE.plot(cvEbins[:-1], cveCnt / float(cveCnt.sum()), '.-', label = legendLabel + r'($\mu$: %.4s)'%(cvE.mean()))
#    axI.plot(cvIbins[:-1], cviCnt / float(cviCnt.sum()), '.-', label = legendLabel + r'($\mu$: %.4s)'%(cvI.mean()))
#    axE.plot(cvEbins[:-1], cveCnt / (float(cveCnt.sum()) * binSizeE), '-', label = legendLabel)
 #   axI.plot(cvIbins[:-1], cviCnt / (float(cviCnt.sum()) * binSizeI), '-', label = legendLabel)
    if IF_PLOT:
        axE.hist(cvE, nBins, normed = 1, histtype = 'step', label = legendLabel, linewidth = pltLineWidth)
        axI.hist(cvI, nBins, normed = 1, histtype = 'step', label = legendLabel, linewidth = pltLineWidth)
        print 'mean circ var: ', cvE.mean(), cvI.mean()
    return np.array([cvE.mean(), cvI.mean()])

def plotfr(filename, axE, axI, legendLabel, filetype = 'tuning', NE = 20000, NI = 20000, nBins = 100):
    if filetype == 'tuning':
        tc = np.load('/homecentral/srao/db/data/tuningCurves_' + filename + '.npy')
        tce = tc[:NE, :].mean(1)
        tci = tc[NE:, :].mean(1)
        cntE, binsE = np.histogram(tce, nBins)
        cntI, binsI = np.histogram(tci, nBins)
        binSizeE = np.diff(binsE).mean()
        binSizeI = np.diff(binsI).mean()        
#        axE.plot(binsE[:-1], cntE / (1.0 * cntE.sum() * binSizeE), '.-', label = legendLabel + '(%.4sHz)'%(tce.mean()))
 #       axI.plot(binsI[:-1], cntI / (1.0 * cntI.sum() * binSizeI), '.-', label = legendLabel + '(%.5sHz)'%(tci.mean()))
        axE.hist(tce, nBins, normed = 1, histtype = 'step', label = legendLabel, linewidth = pltLineWidth)
        axI.hist(tci, nBins, normed = 1, histtype = 'step', label = legendLabel, linewidth = pltLineWidth)        
    return np.array([tce.mean(), tci.mean()])

def plotlogfr(filename, axE, axI, legendLabel, filetype = 'tuning', NE = 20000, NI = 20000, nBins = 100, IF_PLOT = True):
    if filetype == 'tuning':
        tc = np.load('/homecentral/srao/db/data/tuningCurves_' + filename + '.npy')
        tce = tc[:NE, :].mean(1)
        tci = tc[NE:, :].mean(1)
        tce = np.log10(tce)
        tci = np.log10(tci)
        tce = tce[~np.logical_or(np.isnan(tce), np.isinf(tce))]
        tci = tci[~np.logical_or(np.isnan(tci), np.isinf(tci))]
        cntE, binsE = np.histogram(tce, nBins)
        cntI, binsI = np.histogram(tci, nBins)
        binSizeE = np.diff(binsE).mean()
        binSizeI = np.diff(binsI).mean()                
#        axE.plot(binsE[:-1], cntE / (1.0 * cntE.sum() * binSizeE), '.-', label = legendLabel) # + '(%.4sHz)'%(tce.mean()))
#        axI.plot(binsI[:-1], cntI / (1.0 * cntI.sum() * binSizeI), '.-', label = legendLabel) # + '(%.5sHz)'%(tci.mean()))
        if IF_PLOT:
            axE.hist(tce, nBins, normed = 1, histtype = 'step', label = legendLabel, linewidth = pltLineWidth)
            axI.hist(tci, nBins, normed = 1, histtype = 'step', label = legendLabel, linewidth = pltLineWidth)        
            axI.set_xlabel('Log firing rates')
            axI.set_ylabel('Count')
#            axI.set_ylabel('Normalized Count')            
#            axI.set_title('I population')

if __name__ == '__main__':
    NE = 20000
    NI = 20000
    figE, axE = plt.subplots()
    figI, axI = plt.subplots()
    figSaveFormat = sys.argv[1]
    bidirTypeName = sys.argv[2] #'i2i'
    axinsetXTicks = [0.0, 0.2, 0.5, 0.8]
    legendLabels = ['control', 'p = 0.2', 'p = 0.5', 'p = 0.8']    
    filetag = 'rho5_xi0.8'
    axinsetXlabel = 'p'
    axinsetCordinatesCV = [.7, .69, .13*1.6, .13]
    axinsetCordinatesCV = [.50, .50, .13*1.6, .13]    
    if bidirTypeName == 'i2i':
        bidirType = '_bidirI2I'
        dbnames = ['cntrl', 'bidirI2I_p2p2', 'bidirI2I_p5p5', 'bidirI2I_p8p8']
        figFolder = '/homecentral/srao/cuda/data/poster/figs/bidir/i2i/'
    elif bidirTypeName == 'e2e':
        bidirType = '_bidirE2E'
        dbnames = ['cntrl', 'bidirE2E_p2', 'bidirE2E_p5', 'bidirE2E_p8']
        figFolder = '/homecentral/srao/cuda/data/poster/figs/bidir/e2e/'
    elif bidirTypeName == 'diffKI':
        IF_PLOT = True
        bidirType = '_diffKI'
        filetag = ''
        axinsetXlabel = ''
        dbnames = ['cntrlrho5_xi0.8', 'KI1000_cntrlrho5_xi0.8', 'KI1500_cntrlrho5_xi0.8']
        legendLabels = [r'$K^{I} = 500$', r'$K^{I} = 1000$', r'$K^I = 1500$']
        figFolder = '/homecentral/srao/cuda/data/poster/figs/diff_kff/'        
        axinsetXTicks = [r'$K^{I}$'+ '\n' + r'$500$', r'$K_{I}$'+ '\n' + r'$1000$', r'$K^I$'+ '\n' + '$1500$']
        axinsetCordinatesCV = [.2, .69, .11*1.6, .11]        
    else:
        IF_PLOT = True
        bidirType = ''
        filetag = ''
        axinsetXlabel = ''
        dbnames = ['cntrlrho5_xi0.8', 'kff200rho5_xi0.8', 'kffi400rho5_xi0.8', 'kffi800rho5_xi0.8']
        legendLabels = [r'$k_{ff}^{E,I} = 100$', r'$k_{ff}^{E,I} = 200$', r'$k_{ff}^I = 400$', r'$k_{ff}^I = 800$']
        figFolder = '/homecentral/srao/cuda/data/poster/figs/diff_kff/'        
        axinsetXTicks = [r'$k_{ff}^{E,I}$'+ '\n' + r'$100$', r'$k_{ff}^{E,I}$'+ '\n' + r'$200$', r'$k_{ff}^I$'+ '\n' + '$400$', r'$k_{ff}^I$'+'\n'+ '$800$']
        axinsetCordinatesCV = [.265, .69, .13*1.6, .13]
        axinsetCordinatesCV = [.5, .5, .13*1.6, .13]

   #    legendLabels = ['control', r'$k_{ff}^I = 400$', r'$k_{ff}^I = 800$', r'$k_{ff}^{E,I} = 200$']        
    
    meanCircVars = np.zeros((len(legendLabels), 2))
    
    for kk, klabel in enumerate(legendLabels):
        cv = np.load('/homecentral/srao/db/data/Selectivity_' + dbnames[kk] + filetag + '.npy')
        meanCircVars[kk, :] = PlotCircVars(cv, NE, NI, axE, axI, klabel)
#    axE.set_title('E population')
    axE.set_xlabel('Circular Variance')
    axE.set_ylabel('Count')
 #   axI.set_title('I population')
    axI.set_xlabel('Circular Variance')
#    axI.set_ylabel('Normalized count')
    axI.set_ylabel('Count')    
    axE.set_xlim(0.0, 1.0)
    axI.set_xlim(0.0, 1.0)    
    #plt.figure(figE.number)   
    #axE.legend(prop={'size': 10}, loc = 2, frameon = False, markerscale = 10.0)
    plt.figure(figI.number)   
#    plt.legend(prop={'size': 10}, loc = 2, frameon = False )

    filename = 'summary_ori_cvDistr_E' + bidirType

    minX = np.min([axE.get_xlim()[0], axI.get_xlim()[0]])
    maxX = np.max([axE.get_xlim()[1], axI.get_xlim()[1]])
    maxY = np.max([axE.get_ylim()[1], axI.get_ylim()[1]])
    axE.set_xlim(minX, maxX)
    axE.set_ylim(0, 4.5 ) #maxY)
    axE.set_xticks(np.arange(0.0, 1.1, 0.5))
    axE.set_yticks(np.arange(0, 4.6, 1.5))

    axI.set_xlim(minX, maxX)
    axI.set_ylim(0, 4.5) #maxY)
    axI.set_xticks(np.arange(0.0, 1.1, 0.5))
    axI.set_yticks(np.arange(0, 4.6, 1.5))
#    axI.set_xticks(np.arange(0.0, 1.1, 0.2))
 #   axI.set_yticks(np.arange(0, 4.6, 1.5))
    
    

    # ------------- SUMMARY FOR ALL P --------------
    # if (bidirTypeName == 'e2e' or bidirTypeName == 'i2i'):
    #     print "HERE -->  ", len(meanCircVars)        
    #     allP = range(9)
    #     meanCircVars = np.zeros((len(allP), 2))
    #     print "HERE -->  ", len(meanCircVars)
    #     axinsetXTicks = np.arange(0, 9, 2)
    #     for kk, klabel in enumerate(allP):
    #         cv = np.load('/homecentral/srao/db/data/Selectivity_' + dbnames[kk] + filetag + '.npy')    
    figSummary, axinset = plt.subplots()
    plt.figure(figSummary.number)
    xAxisP = np.array([0.0, 0.2, .5, .8])
    axinset.plot(np.arange(meanCircVars.shape[0]), meanCircVars[:, 0], 'k-o', label = 'E', linewidth = pltLineWidth)
    axinset.plot(np.arange(meanCircVars.shape[0]), meanCircVars[:, 1], 'r-o', label = 'I', linewidth = pltLineWidth)
#   axinset.set_yticks(np.arange(0.5, meanCircVars[:, 1].max()+.3, 0.2)) #np.arange(0, axinset.get_yticks()[-1], 2))
#   axinset.set_xticks(np.arange(meanCircVars.shape[0]))
    axinset.set_xlabel(axinsetXlabel)
    axinset.set_xticklabels(axinsetXTicks)
    axinset.set_xlabel(axinset.get_xlabel(), fontsize = 10);
    axinset.set_ylim(0.5, 0.8)
    axinset.set_ylabel('Mean CV')
    axinset.grid('on')
    axinset.set_xlim(-0.1, 0.9)
#    paperSize = [1.71*1.65, 1.21*1.65]

    paperSize = [2.0, 1.5]
    axPosition = [0.26, 0.25, .65, 0.65]
    if bidirTypeName == 'diffKI':
#        axinset.set_xlim(0, 2)
        axinset.set_xticks(np.arange(0, 2.5, 1.))
        axinset.set_xticklabels(axinsetXTicks)
        axinset.set_yticks(np.arange(.5, .81, .1))         
    # plt.ion()
    # plt.show()
    # kb.keyboard()
    figname = 'EI_summary_mean_circvar_distr_' + bidirType
    axinset.legend(bbox_to_anchor=(-0., 1.02, 1.0, .102), loc=3, ncol=2, mode="expand", borderaxespad=0., frameon=False, numpoints = 1)
    Print2Pdf(figSummary,  figFolder + figname,  paperSize, figFormat=figSaveFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition) #[0.18, 0.26, .7, .6 * 0.961])



 #   paperWidth = 5.0 #7.08661 # 2 columns
#    paperSize = [paperWidth, paperWidth / 1.61803398875] #[4.6,  4.0]
    Print2Pdf(figE,  figFolder + filename,  paperSize, figFormat=figSaveFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition) #[0.175, 0.26, .77, .6 * 0.961])  #[0.142, 0.15, .77, .74])
    filename = 'summary_ori_cvDistr_I' + bidirType
    Print2Pdf(figI,  figFolder + filename,  paperSize, figFormat=figSaveFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition) # [0.175, 0.26, .77, .6 * 0.961]) #[0.142, 0.15, .77, .74])
    #---------------------- Distr of firing rates 

    # plt.close('all')
    # figE, axE = plt.subplots()
    # figI, axI = plt.subplots()
    meanFiringRates = np.zeros((len(legendLabels), 2))
    for kk, klabel in enumerate(legendLabels):
        meanFiringRates[kk, :] = plotfr(dbnames[kk] + filetag, axE, axI, klabel, filetype = 'tuning', NE = 20000, NI = 20000, nBins = 100)
    # plt.figure(figE.number)
    # plt.xlabel('Firing rate (Hz)')
    # plt.ylabel('Normalized Count')
    # plt.title('Firing rate distribution, E population')
    # figname = 'summary_firing_rateDistr_E' + bidirType
    # plt.legend(prop={'size': 10}, loc = 0, frameon = False)
    # Print2Pdf(plt.gcf(),  figFolder + figname,  [4.6,  4.0], figFormat=figSaveFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .77, .74])
    # plt.figure(figI.number)    
    # plt.xlabel('Firing rate (Hz)')
    # plt.ylabel('Normalized Count')
    # plt.title('Firing rate distribution, I population')
    # figname = 'summary_firing_rateDistr_I' + bidirType
    # plt.legend(prop={'size': 10}, loc = 0, frameon = False)
    # Print2Pdf(plt.gcf(),  figFolder + figname,  [4.6,  4.0], figFormat=figSaveFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .77, .74])


# ----- PLOT LOG-FIRING DISTRs
    plt.close('all')
    figE, axE = plt.subplots()
    figI, axI = plt.subplots()
    for kk, klabel in enumerate(legendLabels):
        plotlogfr(dbnames[kk] + filetag, axE, axI, klabel, filetype = 'tuning', NE = 20000, NI = 20000, nBins = 100)
    plt.figure(figE.number)
    plt.xlabel('Log firing rate')
    plt.ylabel('Count')
#    plt.title('E population')
   # plt.legend(prop={'size': 10}, loc = 0, frameon = False)
    # minX = np.min([axE.get_xlim()[0], axI.get_xlim()[0]])
    # maxX = np.max([axE.get_xlim()[1], axI.get_xlim()[1]])
    maxY = np.max([axE.get_ylim()[1], axI.get_ylim()[1]])
    # axE.set_xlim(minX, maxX)
    # axI.set_xlim(minX, maxX)
    axE.set_ylim(0, maxY)
    axE.set_xlim(-4, 6)
    axE.set_yticks(np.arange(0, .46, .15))
    
    axI.set_ylim(0, maxY)
    axI.set_xlim(-4, 6)
    axI.set_yticks(np.arange(0, .46, .15))    
#    axE.legend(prop={'size': 10}, loc = 2, frameon = False, ncol = 2)
#    axE.legend(bbox_to_anchor=(-0., 0.6, 1.0, .102), loc=3, ncol=2, borderaxespad=0., frameon=False, numpoints = 1)    
    plt.figure(figI.number)
    print 'mean rates: \n', meanFiringRates
#    axinset = plt.axes([.7, .65, .18*1.6, .18])
#    figSummary, axdummy = plt.subplots()
    figSummary, axinset = plt.subplots()
#    figSummary = plt.figure()
    plt.figure(figSummary.number)    
 #   axinset = plt.axes(axinsetCordinatesCV)    #[.24, .65, .15*1.6, .15])    
    axinset.plot(np.arange(meanFiringRates.shape[0]), meanFiringRates[:, 0], 'k-o', label = 'E', linewidth = pltLineWidth)
    axinset.plot(np.arange(meanFiringRates.shape[0]), meanFiringRates[:, 1], 'r-o', label = 'I', linewidth = pltLineWidth)

    axinset.set_yticks(np.arange(0, meanFiringRates[:, 1].max() + 1, 4)) #np.arange(0, axinset.get_yticks()[-1], 2))
    axinset.set_xticks(np.arange(meanFiringRates.shape[0]))
#    axinsetXlabels = axinsetXlabels
    axinset.set_xticklabels(axinsetXTicks)
    axinset.set_xlabel(axinsetXlabel)
    #axinset.set_xlabel('Mean Activity(Hz)')    
    axinset.set_ylabel('Mean(Hz)')
    axinset.grid('on')
    axinset.set_xlabel(axinset.get_xlabel(), fontsize = 10);
 #   paperSize = [1.78, 1.21]
#    paperSize = [1.5*1.65, 1.21*1.65]
#    paperSize = [1.71*1.65, 1.21*1.65]            
    figname = 'EI_summary_mean_firingrate_' + bidirType
 #   axinset.legend(bbox_to_anchor=(-0.1, 1.02, 1.2, .102), loc=3, ncol=2, mode="expand", borderaxespad=0., frameon=False, numpoints = 1)
    axinset.legend(bbox_to_anchor=(-0., 1.02, 1.0, .102), loc=3, ncol=2, mode="expand", borderaxespad=0., frameon=False, numpoints = 1)    
#    plt.legend(prop={'size': 10}, loc = 0, frameon = False, numpoints = 1)
    Print2Pdf(figSummary,  figFolder + figname,  paperSize, figFormat=figSaveFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition) #[0.18, 0.26, .7, .6 * 0.961]) #[0.175, 0.26, .77, .6 * 0.961])



    # paperWidth = 5. #7.08661 # 2 columns
    # paperSize = [paperWidth, paperWidth / 1.61803398875] #[4.6,  4.0]
    figname = 'summary_log_firing_rateDistr_E' + bidirType
    Print2Pdf(figE,  figFolder + figname, paperSize, figFormat=figSaveFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition) #[0.22, 0.21, .7, .6 * 0.961]) #[0.13, 0.175, .77, .72])

    figname = 'summary_log_firing_rateDistr_I' + bidirType
    Print2Pdf(figI,  figFolder + figname,  paperSize, figFormat=figSaveFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition) #[0.22, 0.21, .7, .6 * 0.961]) #[0.18, 0.26, .77, .6 * 0.961]) #[0.13, 0.175, .77, .72])




