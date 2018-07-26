#!/usr/bin/python

# ''' python ComputeAC.py 8 3 i2i 2000'''
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
from reportfig import ReportFig
from Print2Pdf import Print2Pdf
import time
import ipdb
import matplotlib as mpl
mpl.use('Agg')
sys.path.append(basefolder)
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


sys.path.append("/homecentral/srao/Documents/code/mypybox")
sys.path.append("/homecentral/srao/Documents/code/mypybox/utils")

def ProcessFigure(figHdl, filepath, IF_SAVE, IF_XTICK_INT = True, figFormat = 'eps', paperSize = [4, 3], titleSize = 10, axPosition = [0.25, 0.25, .65, .65], tickFontsize = 10, labelFontsize = 12, nDecimalsX = 1, nDecimalsY = 1):
    FixAxisLimits(figHdl)
    FixAxisLimits(plt.gcf(), IF_XTICK_INT, nDecimalsX, nDecimalsY)
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
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))	
    ax.set_xticks(xticks)
    ax.set_yticks([ymin, 0.5 *(ymin + ymax), ymax])
    plt.draw()


def GetTuningCurves(p, nPhis = 8, xi = 0.8, trNo = 0, N = 10000, K = 500, nPop = 2, T = 100000, bidirType = 'e2i'):
    NE = N
    NI = N
    tc = np.zeros((NE + NI, nPhis))
    tc[:] = np.nan
    cur_tc = np.zeros((NE + NI, nPhis))
    cur_tc[:] = np.nan
    phis = np.linspace(0, 180, nPhis, endpoint = False)
    # filebase = '/homecentral/srao/cuda/data/pub/bidir/%s/tau3/p%s/K%s/'%(bidirType, p, int(K))      
#    filebase = '/homecentral/srao/cuda/data/pub/bidir/N%s/%s/tau3/p%s/K%s/'%(int(N), bidirType, p, int(K))
    filebase = '/homecentral/srao/cuda/data/pub/bidir/N%s/KFFI/%s/tau3/p%s/K%s/'%(int(N), bidirType, p, int(K))    
    
    # filebase = '/homecentral/srao/cuda/'
    for i, iPhi in enumerate(phis):
	print i, iPhi
	try:
	    if i == 0:
		print 'loading from fldr: ', filebase

	    else:
		print iPhi
	    # print 'loading fr... ', 'firingrates_xi0.8_theta%s_0.%s0_3.0_cntrst100.0_%s_tr%s'%(int(iPhi), p, T, trNo ),
            fr = np.loadtxt(filebase + 'firingrates_xi0.8_theta%s_0.%s0_3.0_cntrst100.0_%s_tr%s.csv'%(int(iPhi), p, T, trNo ))
    	    # print 'done'
            # print 'N = ', fr.size
	    # print 'loading cur...',
	    curAtTheta = np.loadtxt(filebase + 'total_synaptic_cur_xi0.8_theta%s_0.%s0_3.0_cntrst100.0_%s_tr%s.csv'%(int(iPhi), p, T, trNo))
            # print '#nan = ', np.sum(np.isnan(curAtTheta))
	    # print 'done'	    
	    if(len(fr) == 1):
		if(np.isnan(fr)):
		    print 'file not found!'
	    tc[:, i] = fr
	    cur_tc[:, i] = curAtTheta
	except IOError:
	    print 'file not found!'
    return tc, cur_tc

def WrapTuningCurve(tc):
    return np.concatenate((tc, [tc[0]]))
    
def PlotInOutTcAux(p, nPhis = 8, xi = 0.8, trNo = 0, N = 20000, K = 500, nPop = 2, T = 100000, bidirType = 'e2i'):
    tc, cur_tc = GetTuningCurves(p, nPhis, xi, trNo, N, K, nPop, T, bidirType)
    tc.shape
    avgTcE = GetAvgTunningCurve(tc[:N, :])
    avgTcI = GetAvgTunningCurve(tc[N:, :])
    avgCurTcE = GetAvgTunningCurve(cur_tc[:N, :])
    avgCurTcI = GetAvgTunningCurve(cur_tc[N:, :])
    #ipdb.set_trace()
    print avgCurTcE
    return avgTcE, avgTcI, avgCurTcE, avgCurTcI

    #ipdb.set_trace()

def PlotInOutTc(pList, nPhis = 8, xi = 0.8, trNo = 0, N = 20000, K = 500, nPop = 2, T = 100000, bidirType = 'e2i', IF_NORMALIZE = 0):
    theta = np.linspace(-90, 90, nPhis, endpoint = False)
    theta = np.concatenate((theta, [90]))
    fg0, ax0 = plt.subplots()
    fg1, ax1 = plt.subplots()
    fg2, ax2 = plt.subplots()
    fg3, ax3 = plt.subplots()        
    for p in pList:
        avgTcE, avgTcI, avgCurTcE, avgCurTcI = PlotInOutTcAux(p, nPhis, xi, trNo, N, K, nPop, T, bidirType)
        print 'p = ', p, 'osi E:', OSI(avgTcE), ' I:', OSI(avgTcI) 
        if IF_NORMALIZE:
            plt.figure(fg0.number)
            plt.plot(theta, WrapTuningCurve(avgTcE) / np.max(avgTcE), 'o-', markersize = 0.6, lw = 0.5)
            plt.figure(fg1.number)
            plt.plot(theta, WrapTuningCurve(avgTcI) / np.max(avgTcI), 'o-', markersize = 0.6, lw = 0.5)
            plt.figure(fg2.number)            
            plt.plot(theta, WrapTuningCurve(avgCurTcE) / np.max(avgCurTcE), 'o--', markersize = 0.6, lw = 0.5)

            plt.figure(fg3.number)                        
            plt.plot(theta, WrapTuningCurve(avgCurTcI) / np.max(avgCurTcI), 'o--', label = 'p = %s'%(p*1e-1), markersize = 0.6, lw = 0.5)
            # plt.legend(frameon = False)
        else:
            plt.figure(fg0.number)            
            plt.plot(theta, WrapTuningCurve(avgTcE), 'o-')
            # plt.plot(theta, WrapTuningCurve(avgTcI), 'ro-')
            # plt.figure()
            # plt.plot(theta, WrapTuningCurve(avgCurTcE), 'ko--')
            # plt.plot(theta, WrapTuningCurve(avgCurTcI), 'ro--')

    paperSize = [2.0, 1.5]
    axPosition=[.25, .28, .65, .65]
    filename = './figs/tc_popAvg_E_vs_p_' + bidirType
    plt.figure(fg0.number)
    plt.xlabel('PO (deg)')
    plt.ylim(0, 1)
    ProcessFigure(fg0, filename, 1, paperSize = paperSize, axPosition = axPosition, titleSize=10, nDecimalsX=0, nDecimalsY=1, figFormat='svg', labelFontsize = 10, tickFontsize = 8)
    
    filename = './figs/tc_popAvg_I_vs_p_' + bidirType
    plt.figure(fg1.number)
    plt.xlabel('PO (deg)')
    plt.ylim(0, 1)
    ProcessFigure(fg1, filename, 1, paperSize = paperSize, axPosition = axPosition, titleSize=10, nDecimalsX=0, nDecimalsY=1, figFormat='svg', labelFontsize = 10, tickFontsize = 8)
    plt.show()

    filename = './figs/cur_tc_popAvg_E_vs_p_' + bidirType
    plt.figure(fg2.number)
    plt.xlabel('PO (deg)')
    plt.ylim(-0.5, 1)    
    ProcessFigure(fg2, filename, 1, paperSize = paperSize, axPosition = axPosition, titleSize=10, nDecimalsX=0, nDecimalsY=1, figFormat='svg', labelFontsize = 10, tickFontsize = 8)
    plt.show()    


    filename = './figs/cur_tc_popAvg_I_vs_p_' + bidirType
    plt.figure(fg3.number)
    plt.xlabel('PO (deg)')
    plt.ylim(-0.5, 1)        
    ProcessFigure(fg3, filename, 1, paperSize = paperSize, axPosition = axPosition, titleSize=10, nDecimalsX=0, nDecimalsY=1, figFormat='svg', labelFontsize = 10, tickFontsize = 8)
    plt.show()    



    
    plt.show()    
            

        
    
    # neuronIdx = np.random.choice(N, 100)
    # for i in neuronIdx:
    #     plt.plot(tc[i, :] / np.nanmax(tc[i, :]), 'ko-')
    #     plt.plot(cur_tc[i, :] / np.nanmax(np.abs(cur_tc[i, :])), 'g--')
    #     # plt.plot(cur_tc[i, :] / 80000., 'g--')	
    #     plt.waitforbuttonpress()
    #     plt.clf()

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
        tc = np.roll(tc, maxIdx * -1 + nTheta / 2)
        out = out + tc / nNeurons
    return out
        
	
def OSI(firingRate):
    out = np.nan
    nThetas = firingRate.size
    atTheta = np.linspace(0, 180.0, nThetas, endpoint = False)
    zk = np.dot(firingRate, np.exp(2j * atTheta * np.pi / 180))
    if(firingRate.mean() > 0.0):
        out = np.absolute(zk) / np.sum(firingRate)
    return out

def OSIOfPop(firingRates):
    nNeurons, nThetas = firingRates.shape
    out = np.zeros((nNeurons, ))    
    for i in range(nNeurons):
        out[i] = OSI(firingRates[i , :])
    return out

def PlotOSIHist(tc, NE = 20000):
    osi = OSIOfPop(tc)
    osiE = osi[:NE]
    osiI = osi[NE:]
    plt.hist(osiE[~np.isnan(osiE)], 26, normed = True, histtype = 'step', color = 'k')
    plt.hist(osiI[~np.isnan(osiI)], 26, normed = True, histtype = 'step', color = 'r')
    plt.show()

def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys    

def PlotFrHistAtTheta(p, theta = 0, xi = 0.8, trNo = 0, N = 20000, K = 500, nPop = 2, T = 100000, bidirType = 'e2i'):
    fg0, ax0 = plt.subplots()
    fg1, ax1 = plt.subplots() 
    ############### p = 0 ###############
    filebase = '/homecentral/srao/cuda/data/pub/bidir/%s/p%s/'%(bidirType, p)
    print 'loading from fldr: ', filebase
    fr = np.loadtxt(filebase + 'firingrates_xi0.8_theta%s_0.80_3.0_cntrst100.0_100000_tr0.csv'%(int(theta)))
    fre = fr[:N]
    fri = fr[N:]
    
    ax0.hist(np.log10(fre[fre>1e-2]), 20, normed = 1, histtype = 'step', linewidth = 0.5, label = 'p=0')
    ax1.hist(np.log10(fri[fri>1e-2]), 20, normed = 1, histtype = 'step', linewidth = 0.5, label = 'p=0')

    # x, y = ecdf(fre)
    # ax0.plot(x, y, linewidth = 0.5, label = 'p = 0')
    # x, y = ecdf(fri)
    # ax1.plot(x, y, linewidth = 0.5, label = 'p = 0')
    
    filebase = '/homecentral/srao/cuda/data/pub/bidir/%s/tau3/p%s/K%s/'%(bidirType, p, int(K))
    print 'loading from fldr: ', filebase
    fr = np.loadtxt(filebase + 'firingrates_xi0.8_theta%s_0.80_3.0_cntrst100.0_100000_tr0.csv'%(int(theta)))
    fre = fr[:N]
    fri = fr[N:]
    
    ax0.hist(np.log10(fre[fre>1e-2]), 20, normed = 1, histtype = 'step',linewidth = 0.5, label = 'p=0.8')
    ax1.hist(np.log10(fri[fri>1e-2]), 20, normed = 1, histtype = 'step', linewidth = 0.5, label = 'p=0.8')

    # x, y = ecdf(fre)
    # ax0.plot(x, y, linewidth = 0.5, label = 'p = 0.8')
    # x, y = ecdf(fri)
    # ax1.plot(x, y, linewidth = 0.5, label = 'p = 0.8')

    paperSize = [2.0, 1.5]
    axPosition=[.25, .28, .65, .65]
    filename = './figs/fr_distr_E_p_' + bidirType
    plt.figure(fg0.number)
    plt.xlabel('Log firing rates')
    plt.ylabel('Probability')
    # plt.xlim([0, 50])
    # plt.legend(loc = 4, frameon = False, numpoints = 1, prop = {'size': 8})        
    ProcessFigure(fg0, filename, 1, paperSize = paperSize, axPosition = axPosition, titleSize=10, nDecimalsX=0, nDecimalsY=1, figFormat='svg', labelFontsize = 10, tickFontsize = 8)
    plt.show()    

    plt.figure(fg1.number)
    plt.xlabel('Log firing rates')
    plt.ylabel('Probability')
    # plt.xlim([0, 100])    
    # plt.legend(loc = 4, frameon = False, numpoints = 1, prop = {'size': 8})    
    filename = './figs/fr_distr_I_p_' + bidirType
    ProcessFigure(fg1, filename, 1, paperSize = paperSize, axPosition = axPosition, titleSize=10, nDecimalsX=0, nDecimalsY=1, figFormat='svg', labelFontsize = 10, tickFontsize = 8)
    plt.show()


def ReadCSVClm(filename):
    f = open(filename, 'r') 
    z =  np.array(filter(None, [line.split() for line in f]), dtype = float)
    f.close()
    return z

def ReadCSV0(filename, columnNo):
    z = []
    i = 0
    with open(filename) as f:
	# ipdb.set_trace()
	for line in f:
	    i += 0
	    tmp = line.rstrip('\n')
	    tmp = tmp.split(' ')
	    # print 'line #', i, ' len(line) = ', len(line)
	    z.append(float(tmp[columnNo]))
    f.close()
    return np.array(z)
        # process(line)

def STA_Aux(stBinIdx, cur, n):
    out = []
    for i in stBinIdx:
	if i+n < len(cur):
	    out.append(cur[i:i+n])
    if len(out) > 0:
	return np.vstack(out).mean(0)
    else:
	return np.array([])

def STA(st, cur, kernelSize, discardTime = 50, dt = 0.05, neuronIdx = 0):
    stOfNeuron = st[st[:, 1] == neuronIdx, 0]
    stOfNeuron = stOfNeuron[stOfNeuron > discardTime]
    stBinIdx = np.asarray(stOfNeuron / dt, dtype = int) - int(discardTime / dt)
    nSpikes = float(len(stBinIdx))
    if nSpikes > 1:
	sta = STA_Aux(stBinIdx, cur, kernelSize) / nSpikes
    else:
	sta = np.empty((0, ))
    return sta 

def PopSTA(st, curFilename, NE, kernelSize, discardTime = 50, dt = 0.05):
    staE = np.zeros((kernelSize, ))
    counterE = 0
    for i in range(NE):
	print i
        cur = ReadCSV0(curFilename, i)
	tmp = STA(st, cur, kernelSize, discardTime, dt, i)
	if len(tmp)>1:
	    counterE += 1
	    staE[:] += tmp

    staI = np.zeros((kernelSize, ))
    counterI = 0
    for i in range(NE, NE + NE):
	print i
        cur = ReadCSV0(curFilename, i)
	tmp = STA(st, cur, kernelSize, discardTime, dt, i)
	if len(tmp)>1:
	    counterI += 1
	    staI[:] += tmp

    out = [staE / float(counterE), staI / float(counterI) ]
    np.save('sta_', out)
    return staE / float(counterE), staI / float(counterI)


def LoadCurndSpk(p, idx, xi = 0.8, T = 20000, trNo = 1):
    # print 'loading spikes from file: ',  
    # spkFilename = 'spkTimes_xi0.0_theta0_0.%s0_3.0_cntrst100.0_%s_tr0.csv'%(p, T)
    # print spkFilename
    # st = np.loadtxt(spkFilename, delimiter = ';')
    # print 'done'
    fn = 'current_ioft__xi0.8_theta0_0.%s0_3.0_cntrst100.0_%s_tr%s.csv'%(p, T, trNo)
    cur = ReadCSV0(fn, idx)
    return cur
    
    
def RunSTA(p, N = 20000, T = 5000):
    print 'loading spikes from file: ',  
    spkFilename = 'spkTimes_xi0.0_theta0_0.%s0_3.0_cntrst100.0_%s_tr0.csv'%(p, T)
    print spkFilename
    st = np.loadtxt(spkFilename, delimiter = ';')
    print 'done'
    fn = 'current_ioft__xi0.0_theta0_0.%s0_3.0_cntrst100.0_%s_tr0.csv'%(p, T)
    tic()
    out = PopSTA(st, fn, N, 1000, 35, 1000)
    toc()
    return out




def STA_Parral_AUX(st, curFilename, kernelSize, discardTime = 50, dt = 0.05, neuronIdx = 0):
    # print neuronIdx
    cur = ReadCSV0(curFilename, neuronIdx)    
    stOfNeuron = st[st[:, 1] == neuronIdx, 0]
    stOfNeuron = stOfNeuron[stOfNeuron > discardTime]
    stBinIdx = np.asarray(stOfNeuron / dt, dtype = int) - int(discardTime / dt)
    nSpikes = float(len(stBinIdx))
    if nSpikes > 1:
	sta = STA_Aux(stBinIdx, cur, kernelSize) / nSpikes
    else:
	sta = np.empty((0, ))
	
    return sta 

def PopSTAParallel(st, curFilename, NE, kernelSize, nCores, discardTime = 1000, dt = 0.05):
    staE = np.zeros((kernelSize, ))
    staI = np.zeros((kernelSize, ))    
    counterE = 0
    counterI = 0    
    pool = Pool(nCores)
    jobs = []
    STA_func = partial(STA_Parral_AUX, st, curFilename, kernelSize, discardTime, dt)
    outE = pool.map(STA_func, range(NE))
    outI = pool.map(STA_func, range(NE, 2 * NE))    
    #### remove empty
    for i in range(len(outE)):
	if len(outE[i]) > 0:
	    counterE += 1
	    staE += outE[i]
    print 'averaged over ', counterE, ' E neurons'

    for i in range(len(outI)):
	if len(outI[i]) > 0:
	    counterI += 1
	    staI += outI[i]
    print 'averaged over ', counterI, ' I neurons'	    

    if counterE > 0:
	staE = staE / float(counterE)
    if counterI > 0:
	staI = staI / float(counterI)
    out = [staE , staI ]
    np.save('sta_', out)
    pool.close()
    return staE, staI 

def RunParallelSTA(p, N = 20000, T = 5000):
    print 'loading spikes from file: ',  
    spkFilename = 'spkTimes_xi0.0_theta0_0.%s0_3.0_cntrst100.0_%s_tr0.csv'%(p, T)
    print spkFilename
    st = np.loadtxt(spkFilename, delimiter = ';')
    print 'done'
    fn = 'current_ioft__xi0.0_theta0_0.%s0_3.0_cntrst100.0_%s_tr0.csv'%(p, T)
    tic()
    out = PopSTAParallel(st, fn, N, 1000, 35, 1000)
    toc()
    return out
    

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"

