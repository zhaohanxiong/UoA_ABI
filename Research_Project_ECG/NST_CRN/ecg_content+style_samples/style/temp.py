import scipy.io
import matplotlib.pyplot as plt
import numpy as np

f = 'A1.mat'

temp = scipy.io.loadmat(f)['val']
temp = temp[0,:]

temp = temp - np.min(temp)
temp = temp/np.max(temp)*255

plt.plot(temp);plt.show()
#scipy.io.savemat(f,mdict={'val':temp[6000:7500,None]})
#scipy.io.savemat(f,mdict={'val':temp})
