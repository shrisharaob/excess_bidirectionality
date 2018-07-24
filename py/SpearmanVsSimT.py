import numpy as np
import pylab as plt
from functools import partial
from multiprocessing import Pool
from scipy.stats import spearmanr
import sys, os, ctypes
basefolder = "/homecentral/srao/Documents/code/mypybox"
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf



#os.system("gcc -fPIC -o GetFr.so -shared GetFr.c")
#os.system("gcc -g -ggdb -o GetFr.so -shared GetFr.c")
mycfunc = np.ctypeslib.load_library('GetFr.so', '.')
mycfunc.GetFr.restype = None
mycfunc.GetFr.argtypes = [np.ctypeslib.ndpointer(np.float64, flags = 'aligned, contiguous'),
                          np.ctypeslib.ndpointer(np.uint32, flags = 'aligned, contiguous'),
                          ctypes.c_ulong,
                          ctypes.c_double,
                          ctypes.c_double,
                          ctypes.c_uint,
                          np.ctypeslib.ndpointer(np.float64, flags = 'aligned, contiguous')]

def GetFrRate(st, tStart, NE, NI, tStop):
    requires = ['CONTIGUOUS', 'ALIGNED']
    spkTimes = st[:, 0]
    spkTimes = np.require(spkTimes, np.float64, requires) #double
    neuronIds = np.asarray(st[:, 1], dtype = 'uint32')
    neuronIds = np.require(neuronIds, np.uint32, requires) #unsigned int
    nSpks = np.uint32(spkTimes.size) #unsigned long
    nSpks = np.require(nSpks, np.uint32)
    tStart = np.require(tStart, np.float64)
    tStop = np.require(tStop, np.float64)
    nNeurons = np.require(NE+NI, np.uint32)
    fr = np.zeros((NE+NI, ))
    fr = np.require(fr, np.float64)
    mycfunc.GetFr(spkTimes, neuronIds, nSpks, tStart, tStop, nNeurons, fr)
    fre = fr[:NE]
    fri = fr[NE:]
    return fre, fri

def GetFrRate1(p, tau_syn, NE, NI, trialId, tStop):
    if int(tau_syn) == 3:
        bf = '/homecentral/srao/Documents/code/cuda/cudanw/data/pub/bidir/i2i/p%s/T320/'%(p)
        filename = 'fr_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_320000_tr%s_tStop%s.csv'%(p, tau_syn, trialId, int(tStop))
    elif int(tau_syn) == 6:
        bf = '/homecentral/srao/Documents/code/cuda/cudanw/data/pub/bidir/i2i/tau%s/p%s/T630/'%(tau_syn, p)
        filename = 'fr_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_630000_tr%s_tStop%s.csv'%(p, tau_syn, trialId, int(tStop))
    elif int(tau_syn) == 12:
        bf = '/homecentral/srao/Documents/code/cuda/cudanw/data/pub/bidir/i2i/tau%s/p%s/T1255/'%(tau_syn, p)
        filename = 'fr_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_1255000_tr%s_tStop%s.csv'%(p, tau_syn, trialId, int(tStop))                
    else: # 24ms and 48ms
        bf = '/homecentral/srao/Documents/code/cuda/cudanw/data/pub/bidir/i2i/tau%s/p%s/T5000s/'%(tau_syn, p)        
        filename = 'fr_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_5002000_tr%s_tStop%s.csv'%(p, tau_syn, trialId, int(tStop))
    # filename1 = 'fr_xi0.8_theta0_0.90_3.0_cntrst100.0_1000000_tr1_tStop%s.csv'%(tStop)
    print "loading file", filename
    fr = np.loadtxt(bf + filename)
    fre = fr[:NE]
    fri = fr[NE:]
    return fre, fri

# def GetFrRate(st, tStart, NE, NI, tStop):
#     fre = np.empty((NE, ))
#     fri = np.empty((NI, ))
#     fre[:] = np.nan
#     fri[:] = np.nan
#     idx0 = np.logical_and(st[:, 0] > tStart, st[:, 0] < tStop)
#     tInterval = 1e-3 * (tStop - tStart)
#     print tStop
#     for kNeuron in range(NE + NI):
# #        print kNeuron
#         idx1 = np.logical_and(idx0, st[:, 1] == kNeuron)
#         nSpikes = idx1.sum()
#         if(nSpikes > 0):
#             if(kNeuron < NE):
#                 fre[kNeuron] = float(nSpikes) / tInterval
#             else:
#                 fri[kNeuron - NE] = float(nSpikes) / tInterval
#     return fre, fri

if __name__ == "__main__":
    tau_syn = int(sys.argv[1])
    ne = 20000
    if tau_syn == 3:
        simDuration = 320000.0 - 10
    elif tau_syn == 6:
        simDuration = 630000.0 - 10
    elif tau_syn == 12:
        simDuration = 1225000.0 - 10
    elif tau_syn == 12:
        simDuration = 2500000.0 - 10
    else:
        simDuration = 5002000.0 - 10
    p = int(sys.argv[2])
    runType = 'plot'
    funcType = 'frfile' # 'spkFile'
    if len(sys.argv) > 4:
        runType = sys.argv[4]
    print runType, funcType
    if runType == 'compute':
        maxNworkers = int(sys.argv[3])
        npyFile0 = 'st0_p%s_taus%s.npy'%(p, tau_syn)
        npyFile1 = 'st1_p%s_taus%s.npy'%(p, tau_syn)
        if funcType == 'spkFile':
            if(os.path.isfile(npyFile0) & os.path.isfile(npyFile1)):
                print 'loading npy'
                st0 = np.load(npyFile0)
                st1 = np.load(npyFile1)
            else:
                bf = "/homecentral/srao/Documents/code/cuda/cudanw/data/pub/bidir/i2i/tau%s/p%s/"%(tau_syn, p)
                if tau_syn == 3:
                    if p == 0:
                        bf = "/homecentral/srao/Documents/code/cuda/cudanw/data/pub/bidir/p%s/"%(p)
                        filename0 = 'spkTimes_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_%s_tr100.csv'%(p, tau_syn, int(simDuration))
                        filename1 = 'spkTimes_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_%s_tr101.csv'%(p, tau_syn, int(simDuration))
                    else:
                        bf = "/homecentral/srao/Documents/code/cuda/cudanw/data/pub/bidir/i2i/p%s/"%(p)        
                        filename0 = 'spkTimes_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_%s_tr0.csv'%(p, tau_syn, int(simDuration))
                        filename1 = 'spkTimes_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_%s_tr1.csv'%(p, tau_syn, int(simDuration))
                else:
                    filename0 = 'spkTimes_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_%s_tr0.csv'%(p, tau_syn, int(simDuration))
                    filename1 = 'spkTimes_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_%s_tr1.csv'%(p, tau_syn, int(simDuration))
                print filename0
                print filename1
                print 'loading files...'
                st0 = np.loadtxt(bf + filename0, delimiter = ';')
                st1 = np.loadtxt(bf + filename1, delimiter = ';')
                np.save('st0_p%s_taus%s'%(p, tau_syn), st0)
                np.save('st1_p%s_taus%s'%(p, tau_syn), st1)
            print 'done'
            func0 = partial(GetFrRate, st0, tStart, ne, ne)
            func1 = partial(GetFrRate, st1, tStart, ne, ne)
        else:
        #------- use func2 and func3 when firing rates are already stored during the simulations ---------
        #def GetFrRate1(p, tau_syn, NE, NI, trialId, tStop):        
            func0 = partial(GetFrRate1, p, tau_syn, ne, ne, 0)
            func1 = partial(GetFrRate1, p, tau_syn, ne, ne, 1)
        tStart = 5000.0 #ms
        tStep = 1000.0
        tStop = np.arange(tStart + tStep, simDuration + 1., tStep)
        fre = np.zeros((tStop.size, ))
        fri = np.zeros((tStop.size, ))
        rhoE = np.zeros((tStop.size, ))
        pValE = np.zeros((tStop.size, ))
        rhoI = np.zeros((tStop.size, ))
        pValI = np.zeros((tStop.size, ))
        pool = Pool(np.min([tStop.size, 20, maxNworkers]))                    
        out0 = pool.map(func0, tStop)
        out1 = pool.map(func1, tStop)        
        for kk, kTStop in enumerate(tStop):
            print kTStop, simDuration
            fre[kk] = np.nanmean(out0[kk][0])
            fri[kk] = np.nanmean(out0[kk][1])
            tmpFre = out0[kk][0]
            tmpFri = out0[kk][1]
            tmpFre1 = out1[kk][0]
            tmpFri1 = out1[kk][1]
            tmp0, tmp1 = spearmanr(tmpFre[~np.isnan(tmpFre)], tmpFre1[~np.isnan(tmpFre1)])
            tmp0I, tmp1I =  spearmanr(tmpFri[~np.isnan(tmpFri)], tmpFri1[~np.isnan(tmpFri1)])
            if(tmp0.size == 1):
                rhoE[kk] = tmp0
            if(tmp1.size == 1):            
                pValE[kk] = tmp1
            if(tmp0I.size == 1):
                rhoI[kk] = tmp0I
            if(tmp1I.size == 1):            
                pValI[kk] = tmp1I
        print rhoE
        print rhoI
        plt.ioff()
        denum = np.sqrt(1e-3 * ((tStop - tStart)))
        np.save('sparman_p%s_taus%s.png'%(p, tau_syn), [denum, rhoE, pValE, rhoI, pValI])
        print '-------------------------------------------------------------'
        print 1.0 / denum    
        print '-------------------------------------------------------------'
    if runType == 'plot':
        taus_list = [3, 6, 12, 24, 48]
        fge, axe = plt.subplots()
        fgi, axi = plt.subplots()
#        linestypes = ['-', '--', '-.']
        for kk, kTaus in enumerate(taus_list):
            out = np.load('sparman_p%s_taus%s.png.npy'%(p, kTaus))
            denum = out[0, :]
            rhoE = out[1, :]
            rhoI = out[3, :]
            if kTaus == 3:
                tmpdata0 = np.sqrt(kTaus * 1e-3) / denum
                validIdxdata = tmpdata0[-1]
            if kTaus == 24:
                tmpdata0 = np.sqrt(kTaus * 1e-3) / denum                
                validIdx = tmpdata0 >= validIdxdata
                rhoE = rhoE[validIdx]
                rhoI = rhoI[validIdx]
                denum = denum[validIdx]                
            axe.plot(np.sqrt(kTaus * 1e-3) / denum, 1.0 - rhoE, label = r'$\tau_{syn} = %sms$'%(kTaus), linewidth = 0.4)
            axi.plot(np.sqrt(kTaus * 1e-3) / denum, 1.0 - rhoI, label = r'$\tau_{syn} = %sms$'%(kTaus), linewidth = 0.4)            
            # axe.plot(np.sqrt(kTaus * 1e-3) / denum, 1.0 - rhoE, linestypes[kk], color = 'k', label = r'$\tau_{syn} = %sms$'%(kTaus))
            # axi.plot(np.sqrt(kTaus * 1e-3) / denum, 1.0 - rhoI, linestypes[kk], color = 'k', label = r'$\tau_{syn} = %sms$'%(kTaus))
        # axe.set_xlabel(r'$\sqrt{\tau_{syn} /\ T}$', fontsize = 16)                        
        axe.set_ylabel(r'$1 - \rho$', fontsize = 20)
        # axe.set_title('E neurons', fontsize = 16)
        # axi.set_title('I neurons', fontsize = 16)        
        axi.set_xlabel(r'$\sqrt{\tau_{syn} /\ T}$', fontsize = 16)                
        axi.set_ylabel(r'$1 - \rho$', fontsize = 20)
        axe.set_xlim([0, 0.04])
        axe.set_xticks([0, 0.02, 0.04])
        axe.set_yticks([0, 0.5, 1.0])                
        axi.set_xlim([0, 0.04])
        axi.set_xticks([0, 0.02, 0.04])
        axi.set_yticks([0, 0.5, 1.0])                

        # plt.savefig('sparman_tau_syn_symmary.png')
        figFormat = 'eps'
        paperSize = [2.0, 1.5]
        axPosition = [0.27, .35, .6, .6]
        Print2Pdf(fge,  'sparman_tau_syn_symmary_E',  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
        Print2Pdf(fgi,  'sparman_tau_syn_symmary_I',  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
 
