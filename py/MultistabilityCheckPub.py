import numpy as np
import code
import sys
import pylab as plt
from multiprocessing import Pool
from functools import partial
basefolder = "/homecentral/srao/Documents/code/mypybox"
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from Print2Pdf import Print2Pdf
import SetAxisProperties as AxPropSet
from Print2Pdf2Axes import Print2Pdf2Axes
from YYPlot import YYPlot

def MeanSqrdDiff(fr0, fr1):
    # fr1 & fr0 are vector with firing rates
    n0 = fr0.size
    n1 = fr1.size
    out = np.nan
    if n0 == n1:
        frdiff = fr0 - fr1
        out = np.dot(frdiff, frdiff) / n0
    return out

NE = 20000
tau_syn = 48
alpha = 0
bidirType = 'i2i'
#simDuration = '5002000'
simDuration = '1002000'
datafolder = '/homecentral/srao/cuda/data/pub/bidir/i2i/tau48/p%s/'%(alpha)
#tAxis = np.loadtxt(datafolder + 'frEstTime.csv')
#tAxis = np.arange(50, 5001, 50)
nChunks = 19
tAxis = np.arange(1, nChunks + 1, 1) * 50
filenamesTr0  = ['firingrates_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_%s_tr0_chnk%s.csv'%(alpha, tau_syn, simDuration, x) for x in np.linspace(1, nChunks, nChunks, dtype = int)]
filenamesTr1  = ['firingrates_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_%s_tr1_chnk%s.csv'%(alpha, tau_syn, simDuration, x) for x in np.linspace(1, nChunks, nChunks, dtype = int)]
#filenamesTr1.append('firingrates_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_%s_tr1.csv'%(alpha, tau_syn, simDuration))
#filenamesTr0.append('firingrates_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_%s_tr0.csv'%(alpha, tau_syn, simDuration))
print len(filenamesTr0), len(tAxis)
mseE = np.zeros((nChunks, ))
mseI = np.zeros((nChunks, ))
# mseE = np.zeros((nChunks + 1, ))
# mseI = np.zeros((nChunks + 1, ))
#for i in range(nChunks+1):
for i in range(nChunks):    
    print "loading file --> ", filenamesTr0[i]
    fr0 = np.loadtxt(datafolder + filenamesTr0[i])
    print "loading file --> ", filenamesTr1[i]    
    fr1 = np.loadtxt(datafolder + filenamesTr1[i])    
    mseE[i] = MeanSqrdDiff(fr0[:NE], fr1[:NE])
    mseI[i] = MeanSqrdDiff(fr0[NE:], fr1[NE:])
fgE, axE = plt.subplots()
fgI, axI = plt.subplots()
axE.plot(tAxis, mseE, 'ks-', linewidth = 0.4, markersize = 2.5, markeredgecolor = 'k')
axI.plot(tAxis, mseI, 'rs-', linewidth = 0.4, markersize = 2.5, markeredgecolor = 'r')
print mseE[-1], mseI[-1]
# fg, ax0, ax1 =  YYPlot(tAxis * 1e-3, results[:, 0], results[:, 1], 'o', True)
AxPropSet.SetProperties(axE, axE.get_xlim(), axE.get_ylim(), '', r'MSE($Hz^2$)')
AxPropSet.SetProperties(axI, axI.get_xlim(), axI.get_ylim(), 'T(s)', r'MSE($Hz^2$)')
# axI.set_xlabel('T(s)')        
# axE.set_ylabel(r'MSE($Hz^2$)')
# axI.set_ylabel(r'MSE($Hz^2$)')
# ymax0 = np.max(ax0.get_ylim())
# ymax1 = np.max(ax1.get_ylim())        
# if int(alpha) == 0:
#     ax0.set_ylim([-2, ymax0])
#     ax1.set_ylim([-2, ymax1])            
# elif int(alpha) == 9:
#     ax0.set_ylim([-2, ymax0])
#     ax1.set_ylim([-200, ymax1])            
# ax0.set_yticks([0, 0.5 * ymax0, ymax0])
# ax1.set_yticks([0, 0.5 * ymax1, ymax1])
#        plt.xlim([simTEnds[1], simTEnds[-1] + 2e-3])
#axE.set_xticks(tAxis)
#axI.set_xticks(tAxis)                
#axE.set_xticklabels(['%.0f'%(tmpstim) for tmpstim in tAxis])
#axI.set_xticklabels(['%.0f'%(tmpstim) for tmpstim in tAxis])        
figFolder = './figs/publication_figures/'
filenameE = 'MSE_%s_tau%s_p%s_T%s_E'%(bidirType, tau_syn, alpha, simDuration)
filenameI = 'MSE_%s_tau%s_p%s_T%s_I'%(bidirType, tau_syn, alpha, simDuration)        
axPosition = [0.26, 0.28, .65, 0.65]
paperSize = [4.0, 3.0]
figFormat = 'eps'
Print2Pdf(fgE,  figFolder + filenameE,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
Print2Pdf(fgI,  figFolder + filenameI,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)        

axE.set_xlim([600, 2500])
axI.set_xlim([600, 2500])
plt.show()
