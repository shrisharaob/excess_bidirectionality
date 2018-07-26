basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
import Keyboard as kb
import SetAxisProperties as AxPropSet

if __name__ == '__main__':
    NE = 20000
    NI = 20000
    plt.ioff()
    tau_syn = np.array([3, 6, 12])
    tau_colors = ['k', 'b', 'g', 'r']
    alphas_default = range(9)    
    bidirType = sys.argv[1]
    figFormat = sys.argv[2] #'png'
    filetag = 'I2I'
    figE, axE = plt.subplots()
    figI, axI = plt.subplots()
    figElog, axElog = plt.subplots()
    figIlog, axIlog = plt.subplots()
    figFolder = './figs/publication_figures/'
    dataFolder = './data/'
    print tau_syn
    for mm, mTau in enumerate(tau_syn):
        meanFanofactorsE = []
        meanFanofactorsI = []    
        if mTau == 3:
            alphas_default = [0, 1, 2, 3, 4, 5, 7, 8]
        else:
            alphas_default = range(9)
        mAlpha = alphas_default            
        for nn, nAlpha in enumerate(mAlpha):
            mnFilename = 'fanofactor_p%s_'%(nAlpha) + filetag + '_tau%s.npy'%(int(mTau))
            print 'p =', nAlpha, 'tau = ', mTau, ' --> loding file: ', mnFilename
            fano = np.load(dataFolder + mnFilename)
            meanFanofactorsE.append(np.nanmean(fano[:NE]))
            meanFanofactorsI.append(np.nanmean(fano[NE:]))
        axE.plot(np.array(mAlpha) * 0.1, np.array(meanFanofactorsE), '.-', linewidth = 0.4, markersize = 2.0)
        axI.plot(np.array(mAlpha) * 0.1, np.array(meanFanofactorsI), '.-', linewidth = 0.4, markersize = 2.0)        
        axElog.loglog(1-np.array(mAlpha[0:]) * 0.1, np.array(meanFanofactorsE[0:]), '.-', label = r'$\tau = %s$'%(mTau), linewidth = 0.5, markersize = 2.0)
        axIlog.loglog(1-np.array(mAlpha[0:]) * 0.1, np.array(meanFanofactorsI[0:]), '.-', label = r'$\tau = %s$'%(mTau), linewidth = 0.5, markersize = 2.0)
    AxPropSet.SetProperties(axE, [0, 1], [0, 30], '', r'$\overline{FF}$')
    AxPropSet.SetProperties(axI, [0, 1], [0, 140], r'$p$', r'$\overline{FF}$')    
    axElog.set_xlabel(r'$1-p$')
    axElog.set_ylabel(r'$\overline{FF}$')
    axIlog.set_xlabel(r'$1-p$')
    axIlog.set_ylabel(r'$\overline{FF}$')
    paperSize = [2.0, 1.5]
    axPosition = [0.28, 0.28, .65, 0.65]        
    filename = figFolder + 'fano_factor_vs_tau_summary_E_' + filetag
    Print2Pdf(figE,  filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
    filename = figFolder + 'fano_factor_vs_tau_summary_I_' + filetag
    Print2Pdf(figI,  filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
    filename = figFolder + 'fano_factor_vs_tau_summary_E_loglog' + filetag
    Print2Pdf(figElog,  filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
    filename = figFolder + 'fano_factor_vs_tau_summary_I_loglog' + filetag
    Print2Pdf(figIlog,  filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)    

            
            
            

        
        
        
        
