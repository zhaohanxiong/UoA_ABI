import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import wiener
import pandas as pd
import sklearn.feature_extraction
import cv2
from scipy.sparse import csr_matrix
from scipy.signal import spectrogram
import librosa
from biosppy.signals import ecg

dat = scipy.io.loadmat("sample1_N")

real = dat["style"][0,:]
generated = dat["generate"][0,:]


#plt.plot(generated);plt.show()
#_,_,temp = spectrogram(generated,fs=300,nperseg=8,nfft=256)
#plt.subplot(2, 1, 1)
#plt.plot(generated)
#plt.subplot(2, 1, 2)
#plt.imshow(temp);plt.show()
#temp = np.sum(temp,1)
#temp = temp/np.max(temp)*275
#plt.hist(temp, 20, edgecolor='black', facecolor='g', alpha=0.75);plt.show()

#out = ecg.ecg(signal=np.repeat(generated,2), sampling_rate=1000., show=True)
#print(np.std(real))
#print(np.std(generated))

#from tsfresh.examples.robot_execution_failures import download_robot_execution_failures,load_robot_execution_failures
#from tsfresh import extract_features
#dat = pd.DataFrame(data={'id':np.repeat('A',1),'x':real[0:1]})
#df, _ = load_robot_execution_failures()
#timeseries, y = load_robot_execution_failures()
#X = extract_features(df, column_id='id', column_sort='time')
#extracted_features = extract_features(dat,column_id='id')
