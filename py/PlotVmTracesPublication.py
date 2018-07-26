basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
import Keyboard as kb
bf = '/homecentral/srao/cuda/data/pub/bidir/'
tau = int(sys.argv[1])
p = int(sys.argv[2])
figFolder = './figs/publication_figures/vmtraces/tau%s/p%s/'%(tau, p)
nTrials = 1
nNeruons = 50
dt = 0.025
simTIme = 12000
discardT = 2000
paperSize = [6.6, 0.8] #[10.2, 1.6] #[4.6*2,  3.6]
figFormat = 'eps'
plt.ioff()
tAxis = np.arange(0, simTIme - discardT, dt) * 1e-3
print tAxis[-1], tAxis.shape
frFilename = 'firingrates_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_12000_trvmtr.csv'%(p, tau)
if tau == 3:
    if p == 0:
        foldername = bf + 'p%s/'%(p)
    else:
        foldername = bf + 'i2i/p%s/'%(p)
else:
    foldername = bf + 'i2i/tau%s/p%s/'%(tau, p)        
fr = np.loadtxt(foldername + frFilename)
frSortedIdx=fr[:nNeruons].argsort()
vmFilename = 'vm_xi0.8_theta0_0.%s0_%s.0_12000_trvmtr.csv'%(p, tau)
vm = np.loadtxt(foldername + vmFilename)
print vm.shape
for mNeuron in range(nNeruons):
    plt.plot(tAxis, vm[:, frSortedIdx[mNeuron] + 1], 'k', linewidth = 0.2)
    filename = figFolder + 'vm_trace_NEURON%s_fr%.6s'%(mNeuron, fr[frSortedIdx[mNeuron]])
    print filename
    plt.ylim(-80, 60)
    plt.yticks([-80, 0, 60])
    plt.xlim(0, 10)
    plt.xticks([0, 5, 10])
    Print2Pdf(plt.gcf(),  filename, paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = [0.06, 0.25, .9, .65])        
#        plt.waitforbuttonpress()
    plt.clf()

    
        




 

