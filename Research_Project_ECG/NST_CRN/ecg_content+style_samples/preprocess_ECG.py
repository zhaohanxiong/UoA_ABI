import scipy.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import librosa
from biosppy.signals import ecg

dat = scipy.io.loadmat('style/Normal_full.mat')['val'][0,:]

# QRS subtraction, needs perfect QRS detection tho
#out = ecg.ecg(signal=dat, sampling_rate=300, show=False)
#for r in out['rpeaks']:
#    dat[(r-15):(r+15)] = np.linspace(dat[r-15],dat[r+15],30)
plt.plot(dat);plt.show();sys

# nfft = controls height, nperseg = controls width
#spc = scipy.signal.spectrogram(dat,nfft=38,noverlap=0,nperseg=5)[2]
y_pad = librosa.util.fix_length(dat,len(dat)+50//2)
spc = np.real(librosa.stft(y_pad,n_fft=64,win_length=32))
spc = spc - np.min(spc)
spc = spc/np.max(spc)*255
plt.imshow(spc);plt.show()
plt.imshow(np.real(spc));plt.show()

#scipy.io.savemat('A_noQRS.mat',mdict={'val':dat})
#cv2.imwrite('N.jpg',spc)

# backtransform spectrogram
#y_out = librosa.istft(spc,length=len(dat),win_length=32)
#plt.plot(y_out);plt.show()
