from DCT import *
from scipy.sparse import bsr_matrix
import numpy as np

def Dksparse(Kmax):
    k = []
    j = []
    data = []
    #loop through every row, calculate from (12) non-zero elements and add their coordinates to create sparse matrix
    for i in range(3,Kmax):
        if i>2 and i<282:
            k.append(i)
            j.append(2)
            data.append(1)
        elif i>=282 and i<570:
            for js in range(2,14):
                k.append(i)
                j.append(js)
                data.append(1)
        else:
            for js in range(2,28):
                k.append(i)
                j.append(js)
                data.append(1)
    D = bsr_matrix((data, (k, j)), shape=(Kmax, Kmax))
    return D

def STinit(c, D):
    ST = []
    P = DCTpower(c)
    #loop through [3,len(P)-27) so that we don't have issues with array bounds 
    for i in range(3,len(P)-27):    
        #if power of an element is bigger than that of its neighbours fetch the corresponding row from sparse matrix
        #and add it to the ST vector if it satisfies the second part of (11)
        if P[i] > P[i-1] and P[i] > P[i+1]:
            elems = np.array(D.getrow(i).nonzero()[1]).reshape(-1,1)
            idxes = np.concatenate((elems,-1*elems),axis=0) # we need to check both P[2,..] and P[..,-2]
            check = np.all(P[i]>P[idxes]+7)
            if check:
                ST.append(i)
    
    return np.array(ST,dtype=np.int16)

def Hz2Barks(f):
    return 13*np.arctan(np.multiply(0.00076,f))+3.5*np.arctan(np.square(np.divide(f,7500)))

def STreduction(ST, c, Tq):
    tonal = []
    tonal_for_removal = []
    P = DCTpower(c)
    P_m = np.zeros(P.shape[0],dtype=np.float32)
    #calculate power for every one of the initial tonal elements and check if they have higher value than Tq
    for i in ST:
        P_m[i] = 10*log10(np.power(10,(0.1*P[i-1]))+np.power(10,(0.1*P[i]))+np.power(10,(0.1*P[i+1])))
        if P_m[i] >= Tq[i]:
            tonal.append(i)
    #translate tonal indexes->frequencies(Hz)->frequencies(barks)
    freqs = calcFreqs(tonal)
    barks = Hz2Barks(freqs)
    #check if distance between 2 consecutive tonal components is less than 0.5 and "mark" the one with the least power so it is removed
    for i in range(1,len(barks)):
        if barks[i]-barks[i-1] < 0.5:
            if P_m[tonal[i]] > P_m[tonal[i-1]]:
                tonal_for_removal.append(tonal[i-1])
            else:
                tonal_for_removal.append(tonal[i])
    #add all tones that pass the first and second check to the final tonal components list 
    final_tonal = [tone for tone in tonal if tone not in tonal_for_removal]
    #return the list of tonal components and their power
    return final_tonal,P_m[final_tonal]

#turns a vector of indexes to frequencies based on (9)
def calcFreqs(i):
    freq_step = 44000/(32*36*2)
    return np.multiply(i,freq_step)

def SpreadFunc(ST, PM, Kmax):
    #initialiaze SF matrix with 0s so that every SF[i][k] where dz isn't in [-3,8] is a 0
    SF = np.zeros((Kmax,len(ST)))
    i = np.arange(Kmax)
    freq_i = calcFreqs(i)
    for k in range(len(ST)):
        #for every tonal component calculate their frequency in barks 
        freq_k = calcFreqs(k)
        #calculate a vector of z_i-z_k
        dz = Hz2Barks(freq_i) - Hz2Barks(freq_k)
        #find all cases of (17) in the z_i-z_k vector
        case1 = np.where((dz>=-3)&(dz<-1))
        case2 = np.where((dz>=-1)&(dz<0))
        case3 = np.where((dz>=0)&(dz<1))
        case4 = np.where((dz>=1)&(dz<8))
        #fill the SF matrix based on the different cases of (17)
        SF[case1 ,k] = 17*dz[case1] - 0.4*PM[k] + 11
        SF[case2, k] = (0.4*PM[k] + 6)*dz[case2]
        SF[case3, k] = -17*dz[case3]
        SF[case4, k] = (0.15*PM[k] - 17)*dz[case4] - 0.15*PM[k]
    return SF

def Masking_Thresholds(ST, PM, Kmax):
    SF=SpreadFunc(ST,PM,Kmax)
    T_m = np.zeros((Kmax,len(ST)))
    i = np.arange(Kmax)
    for k in range(len(ST)):
        T_m[i,k] = PM[k] - 0.275*Hz2Barks(calcFreqs(k)) + SF[i,k] - 6.025
    return T_m

def Global_Masking_Thresholds(Tm, Tq):
    Tg = 10*log10(np.power(10,(np.multiply(0.1,Tq))) + np.sum(np.power(10,0.1*Tm),axis=1))
    return Tg

def Psycho(c,D):
    tq_raw = np.load('Tq.npy',allow_pickle=True)
    Tq = tq_raw.tolist()[0]
    Kmax = len(c)
    ST = STinit(c, D)
    Tonal,PM = STreduction(ST, c, Tq)
    Tm = Masking_Thresholds(Tonal,PM,Kmax)
    Tg = Global_Masking_Thresholds(Tm,Tq)
    return Tg