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


##pca
#out = ecg.ecg(signal=generated, sampling_rate=300., show=False)["templates"]
#scipy.io.savemat("gen.mat",mdict={"gen":out})
#out = ecg.ecg(signal=real, sampling_rate=300., show=False)["templates"]
#scipy.io.savemat("real.mat",mdict={"real":out})
#plt.subplot(2, 1, 1)
#plt.plot(generated)
#plt.subplot(2, 1, 2)
#plt.plot(real_pca)
#plt.show()

#### MATLAB CODE BELOW THIS

#[coeff_gen,~,latent_gen] = pca(gen');
#[coeff_real,~,latent_real] = pca(real');
# coeff:  Principal component coefficients, returned as a p-by-p matrix. Each column of coeff contains coefficients for one principal component.
#           The columns are in the order of descending component variance, latent.
# latent: Principal component variances, that is the eigenvalues of the covariance matrix of X, returned as a column vector.

latent_real = np.array([1.2222,0.0258,0.0051,0.0021,0.0015,0.0010,0.0006,0.0005,0.0002,0.0001])
latent_gen = np.array([1.8226,0.2107,0.1578,0.0710,0.0589,0.0209,0.0134,0.0085,0.0049,0.0021])

plt.plot(latent_real,color="green")
plt.plot(latent_gen,color="red")
plt.show()
