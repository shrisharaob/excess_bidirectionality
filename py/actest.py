basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
sys.path.append('/homecentral/Documents/code/tmp')
import SetAxisProperties as AxPropSet
from Print2Pdf import Print2Pdf
from DefaultArgs import DefaultArgs
from scipy.optimize import curve_fit

def func(x, a, tau, c):
    return a*np.exp(- x / tau) + c

def GetExpFit(xx, IF_PLOT = False):
#    acBinSize = 1 #ms
    timeLag = np.arange(xx.size)    
    popt, pcov = curve_fit(func, timeLag, xx)
    if IF_PLOT:
        tt = np.linspace(0, 500, xx.size)
        fittedFunc = func(tt, *popt)
        plt.plot(timeLag, xx, 'k.')
#        plt.plot(ac[:50, neuronType], 'go-')
        plt.plot(tt, fittedFunc, 'r')

    return popt

[tau_syn, K, NE, NI] = DefaultArgs(sys.argv[1:], ['', '', 20000, 20000])
p = [5, 6, 7, 8, 9]
tau_syn = int(tau_syn)
K = int(K)
NE = int(NE)
NI = int(NI)
# N 2e4, K 500
# N 2e4, K 2000
# N 1e4, K 1000
#dataFolderBase = '/homecentral/srao/cuda/data/pub/'
dataFolderBase = '/homecentral/srao/Documents/code/tmp/data/auto2'
if NE == 20000:
    if K == 500:
        if tau_syn == 3:
            fileName = 'long_tau_vs_ac_mat_tr1_bidirNI2E4I2I_tau3_p'
        else:
            fileName = 'long_tau_vs_ac_mat_tr1_bidirNI2E4I2I_tau%s_p'%(tau_syn)
                       # long_tau_vs_ac_mat_tr1_bidirNI2E4I2I_tau48_p8.npy
    elif K == 2000:
        if tau_syn == 3:
            fileName = 'long_tau_vs_ac_mat_tr1_bidir_K2000_T100I2I_tau%s_p'%(tau_syn)
            print fileName
        else:
            fileName = 'long_tau_vs_ac_mat_tr1_bidir_K2000_T100I2I_tau%s_p'%(tau_syn)
                       # long_tau_vs_ac_mat_tr1_bidir_K2000_T100I2I_tau48_p9.npy
elif NE == 10000:
    if tau_syn == 3:
        fileName = 'long_tau_vs_ac_mat_tr1_bidir_N1E4_K1000_T100I2I_tau3_p'        
    else:
        fileName = 'long_tau_vs_ac_mat_tr1_bidir_N1E4_K1000I2I_tau%s_p'%(tau_syn)
                    #long_tau_vs_ac_mat_tr1_bidir_N1E4_K1000I2I_tau48_p9.npy

plt.ion()                    
for k, kp in enumerate(p):
    print 'loading file --> ', fileName + '%s.npy'%(kp), '...', 
    sys.stdout.flush()
    try:
        ac = np.squeeze(np.load(dataFolderBase + fileName + '%s.npy'%(kp)))
        validStartIdx = 10
        out = GetExpFit(ac[validStartIdx:, 1], True)
        plt.title(r'$\tau = %s, \; p = 0.%s$'%(tau_syn, kp))        
        plt.waitforbuttonpress()
        plt.clf()
        print 'done'        
    except IOError:
        print 'file not found'
        

    
    
            






# plt.ioff()
# acp0t3 = np.load('');
# acp8t48 = np.load('long_tau_vs_ac_mat_tr1_bidir_K2000_T100I2I_tau48_p8.npy')
# acp8t3N1K1 = np.load('long_tau_vs_ac_mat_tr1_bidir_N1E4_K1000_T100I2I_tau3_p8.npy')
# plt.plot(np.squeeze(acp0t3[:, 1, :])), label = 'p0t3')
# plt.plot(np.squeeze(acp8t48[:, 1, :])), label = 'p0t3')
# plt.plot(np.squeeze(acp0t3[:, 1, :])), label = 'p0t3')
