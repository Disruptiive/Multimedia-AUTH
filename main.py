from subbands import *
from scipy.fftpack import *
from psychoacoustic import *
from scipy import signal
import matplotlib.pyplot as plt 
from numpy import log10
from os.path import getsize

#plots frequency(Hertz)-magnitude and then frequency(Barks)-magnitude
def plot_frequencies(h,M,L):
    H = make_mp3_analysisfb(h.reshape((len(h),)),M)
    fs = 44100
    for i in range(M):
        w, h = signal.freqz(H[:,i],worN=L,fs = fs)
        plt.plot(w, 10 * log10(np.square(abs(h))))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.show()
    #does the same as before but converts hertz->barks
    for i in range(M):
        w, h = signal.freqz(H[:,i],worN=L,fs = fs)
        plt.plot(Hz2Barks(w), 10 * log10(np.square(abs(h))))

    plt.xlabel('Frequency (Barks)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)

    plt.show()
h_raw = np.load('h.npy',allow_pickle=True)
h = h_raw.tolist()['h']
M = 32; N = 36; L = len(h);
plot_frequencies(h,M,L)

wav_file_name = 'myfile.wav'

print("Encoding-Decoding without any processing")
samplerate, data = wav.read(wav_file_name)
Ytot,x_hat = codec0(data, h, M, N)

#shift original and processed data to account for buffer lag
shift = L-M
output = np.copy(x_hat[:-shift])
data = np.copy(data[shift:])

#calculate SNR
signal = np.mean(np.float64(data)**2)
noise = np.mean(np.float64(data-output)**2)
snr = 10*np.log10(signal/noise)

wav.write("output_without_processing.wav", samplerate,x_hat.astype(np.int16))

print(f"SNR = {snr:.2f}")

print("Encoding-Decoding MP3")
samplerate,data = wav.read(wav_file_name)
Ytot,x_hat,encoded_info = MP3codec0(data, h, M, N)

#shift original and processed data to account for buffer lag
output_mp3 = np.copy(x_hat[:-shift])
data = np.copy(data[shift:])

#calculate SNR
signal = np.mean(np.float64(data)**2)
noise = np.mean(np.float64(data-output_mp3)**2)
snr = 10*np.log10(signal/noise)

wav.write("output_mp3.wav", samplerate,x_hat.astype(np.int16))

#calculate compression rate
final_huffman = []
for frame in encoded_info:
    final_huffman.extend(frame[1])

original_size = getsize(wav_file_name)*8 #size in bits
final_size = len(final_huffman)
compression_rate = 100*(original_size-final_size)/original_size

print(f"SNR = {snr:.2f}")
print(f"Compresison {compression_rate:.2f}%")