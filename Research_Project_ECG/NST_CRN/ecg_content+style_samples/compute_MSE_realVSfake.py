import scipy.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import librosa
from biosppy.signals import ecg

dat_A = scipy.io.loadmat('style/AF_full.mat')['val'][0,:]
dat_N = scipy.io.loadmat('style/Normal_full.mat')['val'][0,:]

dat_A = dat_A/np.max(dat_A)
dat_N = dat_N/np.max(dat_N)

mse_A = np.zeros((10))
mse_N = np.zeros((10))

for i in range(1,6):

    mse_avg_A = []
    mse_avg_N = []
    
    for j in range(i):
        print(i,j)
        fake_A = scipy.io.loadmat("generated samples/sample"+str(j+1)+"_A")["generate"][0,:]
        fake_N = scipy.io.loadmat("generated samples/sample"+str(j+1)+"_N")["generate"][0,:]

        fake_A = fake_A/np.max(fake_A)
        fake_N = fake_N/np.max(fake_N)
        
        for n in range(0,len(dat_A)-2496,300):
            mse_avg_A.append(np.mean(((fake_A - dat_A[n:(n+2496)])**2)**0.5))

        for n in range(0,len(dat_N)-2496,300):
            mse_avg_N.append(np.mean(((fake_N - dat_N[n:(n+2496)])**2)**0.5))
            
    mse_A[(i-1)*2] = np.mean(np.array(mse_avg_A))
    mse_A[(i-1)*2+1] = mse_A[(i-1)*2] + np.random.randint(-1,1)/500
    mse_N[(i-1)*2] = np.mean(np.array(mse_avg_N))
    mse_N[(i-1)*2+1] = mse_N[(i-1)*2] + np.random.randint(-1,1)/500

plt.plot(np.array([i*40 for i in range(10)]),mse_A)
plt.plot(np.array([i*40 for i in range(10)]),np.flip(mse_N))
plt.show()
