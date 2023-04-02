from frame import *
from mp3 import *
from nothing import *
from DCT import *
from psychoacoustic import *
from quantization import *
from RLE import *
from huffman import *
import scipy.io.wavfile as wav
import numpy as np

def MP3codec0(wavin, h, M, N):
    Y_tot,encoder_info = MP3cod(wavin, h, M, N)
    xhat = MP3decod(Y_tot, h, M, N,encoder_info,wavin.shape[0])
    return Y_tot,xhat,encoder_info

def MP3cod(wavin, h, M, N):
    Ytot = coder0(wavin, h, M, N)
    K = M*N
    #calculate number of iterations 
    ds = wavin.shape[0]
    iters = int(ds/(K))
    D = Dksparse(K)
    #encoder_info contains all info needed to decode the signal(huffman symbols, probabilities, bytes used and scaling factors)
    encoder_info = [] 
    print("Started Encoding")
    #loop through every frame and encode it (dct->psycho->quantization->rle->huffman)                                     
    for i in range(iters):
        ybuf  =  Ytot[i*N:(i+1)*N,:] 
        c = frameDCT(ybuf) 
        Tg = Psycho(c,D)
        symb_index,SF,B = all_bands_quantizer(c,Tg)
        run_symbols= RLE(symb_index,K)
        probabilities,coded_str=huff(run_symbols)
        encoder_info.append([probabilities,coded_str,B,SF]) #store encoded data
        print(f"{i+1}/{iters-1}")
    return Ytot,encoder_info

def MP3decod(Y_tot, h, M, N,encoder_info,data_shape):
    K = M*N
    iters = int(data_shape/(K))
    #loop through every frame and decode it (ihuffman->irle->iquantization->idct) based on the info saved in the previous step
    for i in range(iters):
        print(f'{i}/{iters-1}')
        run_symbols = ihuff(encoder_info[i][1],encoder_info[i][0])
        symb_index = iRLE(run_symbols,K)
        #x_h stacked contains an array of elements for every band -> flatten it to a vector 
        x_h_stacked = all_bands_dequantizer(symb_index,encoder_info[i][2],encoder_info[i][3]) 
        x_h=[]
        for j in x_h_stacked:
            x_h.extend(j)
        x_h = np.array(x_h).reshape(-1,1)
        z = iframeDCT(x_h).reshape(N,M)
        if i == 0:
            Yfinal = z
        else:
            Yfinal = np.concatenate((Yfinal,z),axis=0)
    x_hat = decoder0(Yfinal, h, M, N, data_shape)

    return x_hat

def codec0(wavin, h, M, N):
    Ytot = coder0(wavin, h, M, N)
    x_hat = decoder0(Ytot, h, M, N,wavin.shape[0])
    return Ytot,x_hat

def coder0(wavin, h, M, N):
    H = make_mp3_analysisfb(h.reshape((len(h),)),M)
    L = H.shape[0]
    #calculate number of iterations 
    ds = wavin.shape[0]
    iters = int(ds/(M*N))
    x_buffer_size = M*(N-1)+L
    for i in range(iters):
        xbuf =  wavin[i*M*N : (i+1)*M*N + L - M]
        if i == iters - 1:
            #in the last iteration fill the extra L-M elements of buffer with zeros
            xbuf = np.append(xbuf,np.zeros(x_buffer_size-xbuf.shape[0],dtype=np.int16))
        Y = frame_sub_analysis(xbuf,H,N)
        Yc = donothing(Y)
        if i == 0:
            Ytot = Yc
        else:
            Ytot = np.concatenate((Ytot,Yc),axis=0)
    return Ytot

def decoder0(Y_tot, h, M, N,ds):
    iters = int(ds/(M*N))
    G = make_mp3_synthesisfb(h.reshape((len(h),)),M)
    L = G.shape[0]
    ybuf = np.zeros([int((N - 1) + L/M),M],dtype=np.float32) 
    sz = ybuf.shape
    #loop through frames filling the buffer and concatenate output of frame_sub_synthesis() in a vector
    for i in range(iters):
        ybuf  =  Y_tot[i*N:int((i+1)*N+L/M),:]
        if i == iters - 1:
            A = (sz[0]-ybuf.shape[0],sz[1])
            zeros = np.zeros(A)
            ybuf = np.concatenate((ybuf,zeros),axis=0)
        Yh = idonothing(ybuf)

        x = frame_sub_synthesis(Yh,G)
        if i == 0:
            x_hat = x
        else:
            x_hat = np.concatenate((x_hat,x),axis=0)
  
    return x_hat