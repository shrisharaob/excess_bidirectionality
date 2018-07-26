basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
import Keyboard as kb

if __name__ == '__main__':
    NE = 20000
    NI = 20000
    figE, axE = plt.subplots()
    figI, axI = plt.subplots()
    figFormat = sys.argv[1]
    bidirTypeName = sys.argv[2] #'i2i'
    axinsetXTicks = [0.0, 0.2, 0.5, 0.8]
    edgecolors = ['b', 'g', 'r', 'c']
    edgecolorCounter = 0
#    p.insert(0, 'control')
    filetag = '_bidirI2I'
    axinsetXlabel = 'p'
    axinsetCordinatesCV = [.7, .69, .13*1.6, .13]
    p = [2, 5, 8]            
    if bidirTypeName == 'i2i':
#        p = [1, 2, 3, 4, 5, 6, 7, 8]
        fanoFileNamesnames = ['fanofactor' + filetag + '_p%s.npy'%(x) for x in p]
        fanoFileNamesnames.insert(0, 'fanofactor_cntrl.npy') 
        figFolder = './figs/publication_figures/frdistr/i2i/'
        legendLabels = ['p = 0.%s'%(x) for x in p]
        legendLabels.insert(0, 'control')
        pplotidx = [0, 2, 5, 8]        
    elif bidirTypeName == 'e2e':
#        p = [2, 5, 8]
        filetag = '_bidirE2E'        
        fanoFileNamesnames = ['fanofactor_' + filetag + '_p%s.npy'%(x) for x in p]
        fanoFileNamesnames.insert(0, 'fanofactor_cntrl.npy')         
        dbnames = ['cntrl', 'bidirE2E_p2', 'bidirE2E_p5', 'bidirE2E_p8']
        figFolder = './figs/publication_figures/frdistr/e2e/'        
        legendLabels = ['p = 0.%s'%(x) for x in p]
        legendLabels.insert(0, 'control')
        pplotidx = [0, 1, 2, 3]
    elif bidirTypeName == 'e2i':
#        p = [2, 5, 8]
#        p = [1, 2, 3, 5, 8]
#        p = [1, 2, 3, 4, 5, 6, 7, 8]        
        filetag = '_bidirE2I'        
        fanoFileNamesnames = ['fanofactor_' + filetag + '_p%s.npy'%(x) for x in p]
        fanoFileNamesnames.insert(0, 'fanofactor_cntrl.npy')         
        dbnames = ['cntrl', 'bidirE2I_p2', 'bidirE2I_p5', 'bidirE2I_p8']
        figFolder = './figs/publication_figures/frdistr/e2i/'        
        legendLabels = ['p = 0.%s'%(x) for x in p]
        legendLabels.insert(0, 'control')
        pplotidx = [0, 1, 2, 3]
    else:
        filetag = ''
        axinsetXlabel = ''
    bins = np.arange(0, 5, 0.1)
    meanFanoE = []
    meanFanoI = []    
    for n, nFile in enumerate(fanoFileNamesnames):
        print nFile
        if n == 0:
            frFolder = '/homecentral/srao/cuda/data/poster/cntrl/sim0/'
        else:
            frFolder = '/homecentral/srao/cuda/data/poster/bidir/'
            if bidirTypeName == 'e2e':
                frFolder = frFolder + 'e2e/p%s/'%(n)
            elif bidirTypeName == 'i2i':
                frFolder = frFolder + 'i2i/p%s/'%(n)
            elif bidirTypeName == 'e2i':
                frFolder = frFolder + 'e2i/p%s/'%(n)                
        print 'folder name ', frFolder, bidirTypeName
        if (bidirTypeName == 'e2i') and ( n == 4 or n == 6 or n == 7):
            fr = np.loadtxt(frFolder + 'firingrates_xi0.8_theta0_0.%s0_3.0_cntrst100.0_2000_tr0.csv'%(n))
        else:
            fr = np.loadtxt(frFolder + 'firingrates_xi0.8_theta0_0.%s0_3.0_cntrst100.0_100000_tr0.csv'%(n))            
        fanofactor = np.load(nFile)
        print fanofactor.shape
        fanofactorE = fanofactor[:NE]
        fanofactorE = fanofactor[fr[:NE] > 1.]
        meanFanoE.append(np.nanmean(fanofactorE))
        fanofactorI = fanofactor[NE:]
        fanofactorI = fanofactor[fr[NE:] > 1.]
        meanFanoI.append(np.nanmean(fanofactorI))
        print "mean FF: ", meanFanoE[n], meanFanoI[n]
        if np.intersect1d(pplotidx, n).size:
            IS_FILL = True
            if bidirTypeName == 'e2e':
                IS_FILL = False
                transperenceyAlpha = 1.0
            else:
                transperenceyAlpha = 0.45
            cntsE, binsE, patchesE = axE.hist(fanofactorE[~np.isnan(fanofactorE)], bins, normed = 1, histtype = 'step',  label = legendLabels[n], edgecolor = edgecolors[edgecolorCounter], linewidth = 0.5)
            plt.setp(patchesE, 'alpha', transperenceyAlpha)
            cntsI, binsI, patchesI = axI.hist(fanofactorI[~np.isnan(fanofactorI)], bins, normed = 1, histtype = 'step',  label = legendLabels[n], edgecolor = edgecolors[edgecolorCounter], linewidth = 0.5)
#            cntsI, binsI, patchesI = axI.hist(fanofactorI[~np.isnan(fanofactorI)], bins, normed = 1, histtype = 'step', fill = IS_FILL, label = legendLabels[n], edgecolor = edgecolors[edgecolorCounter], linewidth = 0.2)
            plt.setp(patchesI, 'alpha', transperenceyAlpha)
            edgecolorCounter += 1
    #        axI.hist(fanofactorI[~np.isnan(fanofactorI)], bins, normed = 1, label = legendLabels[n])
#            axI.legend(frameon = False)
#            axE.legend(frameon = False, ncol = 1)
#            plt.ion()
 #           plt.show()
  #          plt.waitforbuttonpress()
    axE.set_xlabel('Fano Factor')
    axE.set_ylabel('Count')    
#    axE.set_title('E population')
    axI.set_xlabel('Fano Factor')
    axI.set_ylabel('Count')
#    axI.set_title('I population')
    filename = 'fano_distr_E'
#    axE.set_rasterized(True)
    plt.draw()
    plt.figure(figE.number)
    plt.xlabel('Fano Factor')
    plt.ylabel('Count')
    if bidirTypeName == 'e2e':
        axE.set_ylim(0, 3.0)
        axE.set_xlim(0, 2.0)
        axI.set_xlim(0, 2.0)
        axE.set_xticks(np.arange(3.))
        axI.set_xticks(np.arange(3.))                
        axE.set_yticks(np.arange(4.))
        axI.set_yticks(np.arange(4.))
    if bidirTypeName == 'i2i':
        axE.set_yticks(np.arange(4.))
    if bidirTypeName == 'e2i':
        axE.set_ylim(0, 3.0)
        axE.set_xlim(0, 2.0)
        axI.set_xlim(0, 2.0)
        axE.set_xticks(np.arange(3.))
        axI.set_xticks(np.arange(3.))                
        axE.set_yticks(np.arange(4.))
        axI.set_yticks(np.arange(4.))        

#    paperSize = [1.71*1.65, 1.21*1.65]
 #   axPosition = [0.18, 0.26, .76, .6 * 0.961]

    paperSize = [2.0, 1.5]
    axPosition = [0.26, 0.25, .65, 0.65]

    
    Print2Pdf(figE,  figFolder + filename, paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition) #[0.142, 0.15, .77, .74])

    figMean, axMean = plt.subplots()
    print len(meanFanoE), len(meanFanoI)
#
    if bidirTypeName == 'i2i' :
        p = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        axI.set_yticks(np.arange(0, 4., 1.))        
    elif bidirTypeName == 'e2e':
        p = [0, 2, 5, 8]
    elif bidirTypeName == 'e2i':
#        p = [0, 1, 2, 3, 5, 8]
        p = [0, 1, 2, 3, 4, 5,6, 7, 8]         
        
    axMean.plot(p, meanFanoE, 'ko-', label = 'E', markersize = 2.0, linewidth = 0.5)
    axMean.plot(p, meanFanoI, 'ro-', label = 'I', markersize = 2.0, linewidth = 0.5)
    axMean.legend(numpoints = 1, frameon = False, loc = 0, ncol = 1, prop = {'size':8})
    axMean.set_xlabel('p')
#    axMean.set_ylabel('Mean Fano Factor')
    axMean.set_ylabel(r'$\overline{FF}$')    
    axMean.grid('off')
    if bidirTypeName == 'i2i':
        axMean.set_xlim(0, 9.0)
        axMean.set_ylim(0.8, 2.6)
        axMean.set_yticks(np.arange(0.8, 2.6, 0.4))
    if bidirTypeName == 'e2e':
        axMean.set_xlim(0, 8.25)
        axMean.set_ylim(0.8, 1.4)
        axMean.set_yticks(np.arange(0.8, 1.4, 0.2))
    if bidirTypeName == 'e2i':
        axMean.set_xlim(0, 8.50)        
        axMean.set_xticks(np.arange(0, 9, 4))
        axMean.set_yticks(np.linspace(0.70, 1.0, 3))
    filename = 'summary_p_vs_fano'  
    Print2Pdf(figMean, figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = [0.3, 0.25, .65, 0.65]) #axPosition)

#[0.19, 0.26, .76, .6 * 0.961]     ) #axPosition) #[0.142, 0.15, .77, .74])        
#    plt.show()
 #   plt.waitforbuttonpress()    
          


#    plt.figure(figI.number)
    axinsetCordinatesCV = [.6, .6, .18*1.6, .18]
    figSummary, axinset = plt.subplots()
 #   axinset = plt.axes(axinsetCordinatesCV)
    axinset.plot(np.array(p)*0.1, meanFanoE, 'k.-', label = 'E', linewidth = 0.4, markersize = 1.5)
    axinset.plot(np.array(p)*0.1, meanFanoI, 'r.-', label = 'I', linewidth = 0.4, markersize = 1.5)
#    axinset.legend(numpoints = 1, frameon = False, loc = 2, ncol = 2)
    axinset.set_xlabel('p')
    axinset.set_ylabel('Mean Fano Factor')
    axinset.set_ylim(0.8, 3.0)
    axinset.set_yticks(np.arange(0.8, 2.5, 0.8))
    axinset.set_xticks(np.arange(0., 0.91, 0.3))
    axinset.grid('off')
#    axinset.set_xlim(0, .825)
 #   axinset.set_ylim(0.8, .246)
 #   axinset.legend(bbox_to_anchor=(-0.1, 1.02, 1.2, .102), loc=3, ncol=2, mode="expand", borderaxespad=0., frameon=False, numpoints = 1)
#    axinset.legend(bbox_to_anchor=(loc=3, ncol=1, borderaxespad=0., frameon=False, numpoints = 1)    
#    axinset.set_yticks(np.arange(0.5, meanCircVars[:, 1].max()+.3, 0.2)) #np.arange(0, axinset.get_yticks()[-1], 2))
    if bidirTypeName == 'e2e':
        axinset.set_xticks(np.arange(0, .91, .3))
        axinset.set_yticks(np.arange(0.5, 2.0, 1.0))
#    axinsetXlabels = [0.0, 0.2, 0.5, 0.8]
  #  axinset.set_xlabel(axinsetXlabel)
#    axinset.set_xticklabels(axinsetXTicks)
 #   axinset.set_xlabel(axinset.get_xlabel(), fontsize = 10);
  #  axinset.set_ylim(axinset.get_ylim()[0], 1.0)
   # axinset.set_ylabel('Mean CV')
 #   axinset.grid('on')
    filename = 'fano_distr_I'
    Print2Pdf(figI,  figFolder + filename, paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition) # [0.142, 0.15, .77, .74])

    filename = 'summary_fano_distr_EI'
    Print2Pdf(figSummary,  figFolder + filename, paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition) # [0.142, 0.15, .77, .74])    
