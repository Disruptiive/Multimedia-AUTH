from psychoacoustic import *
import numpy as np
def critical_bands(K):
    #create array with length = K and turn it into frequencies
    freqs = calcFreqs(np.arange(K,dtype=np.int16))
    band_conditions = [freqs<100,(freqs>=100)&(freqs<200),(freqs>=200)&(freqs<300),(freqs>=300)&(freqs<400),(freqs>=400)&(freqs<510),(freqs>=510)&(freqs<630),(freqs>=630)&(freqs<770),(freqs>=770)&(freqs<920),(freqs>=920)&(freqs<1080),(freqs>=1080)&(freqs<1270),(freqs>=1270)&(freqs<1480),(freqs>=1480)&(freqs<1720),(freqs>=1720)&(freqs<2000),(freqs>=2000)&(freqs<2320),(freqs>=2320)&(freqs<2700),(freqs>=2700)&(freqs<3150),(freqs>=3150)&(freqs<3700),(freqs>=3700)&(freqs<4400),(freqs>=4400)&(freqs<5300),(freqs>=5300)&(freqs<6400),(freqs>=6400)&(freqs<7700),(freqs>=7700)&(freqs<9500),(freqs>=9500)&(freqs<12000),(freqs>=12000)&(freqs<15500),(freqs>=15500)]
    band_selection = np.arange(1,26,dtype=np.int16)
    #based on figure 2 convert frequency table to band number
    bands = np.select(band_conditions,band_selection,default=np.nan)
    return bands

def DCT_band_scale(c):
    bands = critical_bands(c.shape[0])
    #splts vector of band elements to 25 arrays(1 for every band)
    split_idx = np.where(np.diff(bands) > 0)[0]+1
    c_per_band = np.split(c,split_idx)
    #calculate sc(band) for every one the arrays(Bands)
    sc = [(lambda i: np.max(np.power(np.abs(c_per_band[i]),3/4)))(i) for i in range(len(c_per_band))]
    #calculate c for every element based on the sc of its band
    cs = np.zeros(c.shape[0])
    for i in range(cs.shape[0]):
        cs[i] = np.sign(c[i])*(np.abs(c[i])**(3/4)/sc[int(bands[i]-1)])
    #return a list of 25 vectors that contain the elements (cs) of every band
    return np.split(cs,split_idx),sc

def quantizer(x,b):
    #calculate how many levels there are before and after 0
    wb = 2/(2**b)
    levels_n = 2**b-1
    levels_per_side = int((levels_n-1)/2)
    levels = []
    symb_index = []
    #take care of edge cases
    if levels_n == 1:
        levels.append([-1,1])
        return np.zeros(len(x))
    elif levels_n == 0:
        return 0
    else:
        #add 1st level
        levels.append([-1,-levels_per_side*wb])
        #calculate levels before 0
        for i in range(levels_per_side-1,0,-1):
            levels.append([-(i+1)*wb,-i*wb])
        #add the level that contains 0 
        levels.append([-wb, wb])
        #calculate levels after 0
        for i in range(1,levels_per_side):
            levels.append([i*wb,(i+1)*wb])
        #add last level
        levels.append([levels_per_side*wb, 1])
        #calculate in which level every element of x belongs to
        for i in range(len(x)):
            for j in range(len(levels)):
                if x[i] >= levels[j][0] and x[i] <= levels[j][1]:
                    #[0,2*levels_per_side] -> [-levels_per_side,levels_per_side]
                    symb_index.append(j-levels_per_side)
                    break
                else:
                    continue
        return symb_index
    
def dequantizer(symb_index, b):
    wb = 2/(2**b)
    levels_n = 2**b-1
    levels_per_side = int((levels_n-1)/2)
    deq_level = []
    if b == 1:
        return np.zeros(len(symb_index))
    elif b == 0:
        return
    else:
        #calculate dequantizer levels before 0 
        for i in range(levels_per_side,0,-1):
            deq_level.append(-i*wb-wb/2)
        deq_level.append(0) #add 0 
        #calculate dequantizer levels after 0 
        for i in range(1,levels_per_side+1):
            deq_level.append(i*wb+wb/2)
    #dequantize elements based on their value 
    #first need to convert[-levels_per_side,levels_per_side] -> [0,2*levels_per_side] 
    symb_index_n = np.add(symb_index,levels_per_side)
    return [deq_level[symb_index_n[i]] for i in range(len(symb_index_n))]

def all_bands_quantizer(c, Tg):
    #convert any nan values to inf
    Tg = np.nan_to_num(Tg,nan=np.inf)
    cs,sc = DCT_band_scale(c)
    bands = critical_bands(c.shape[0])
    #split cs into 25 vectors, each vector i contains the elements of band_number-1
    split_idx = np.where(np.diff(bands) > 0)[0]+1
    c_per_band = np.split(c,split_idx)
    #do the same for Tg
    Tg_per_band = np.split(Tg,split_idx)
    B = np.zeros(len(sc))
    symb_indexes = []
    for i in range(len(sc)):
        bytes_used = 1
        while True:
            #for every band quantize,dequantize,calculate the power of error and compare it with Tg
            symb_index = quantizer(cs[i],bytes_used)
            x = dequantizer(symb_index,bytes_used)
            c_hat = np.sign(x)*np.power(np.abs(np.multiply(x,sc[i])),4/3)
            err = 10*log10(np.power(np.abs(c_per_band[i]-c_hat.reshape(-1,1)),2))
            #if pb(i)<=tg(i) not true for every element increase bits and try again
            if (np.all(err <= Tg_per_band[i].reshape(-1,1)) == True):
                B[i] = bytes_used
                symb_indexes.append(symb_index)
                break
            else:
                bytes_used+=1
    #when the algorithm is done quantize all bands with the most amount of bits that are needed
    Bytes_n = int(max(B))
    for i in range(len(sc)):
        symb_indexes[i] = quantizer(cs[i],Bytes_n)
    return symb_indexes,sc,Bytes_n

def all_bands_dequantizer(symb_index, B, SF):
    bands = critical_bands(len(symb_index))
    #split vector same as before
    split_idx = np.where(np.diff(bands) > 0)[0]+1
    symb = np.split(symb_index,split_idx)
    x_hats = []
    #dequantize symbols based on the bytes that were used and their scaling factor 
    for i in range(len(SF)):
        x = dequantizer(symb[i],B)
        c_hat = np.sign(x)*np.power(np.abs(np.multiply(x,SF[i])),4/3)
        x_hats.append(c_hat)
    return x_hats