import os
import scipy.io
import numpy as np
import h5py

files = os.listdir('temp')

Phi_files= [s for s in files if 'Phi' in s]
PS_files = [s for s in files if 'PS' in s]

for n in range(len(PS_files)):
	
	# load files
	#h5f = h5py.File(os.path.join('temp',Phi_files[n]),'r')
	#Phi = np.transpose(h5f['Phi'].value)
	#h5f.close()
	
	PS  = scipy.io.loadmat(os.path.join('temp',PS_files[n]))['PS_trajectory']
	for i in range(len(PS)):
		PS[i] = PS[i]/4
	
	PS_num = np.array([PS[i,0].shape[0] for i in range(len(PS))])
	
	# find interval wherePS is valid
	PS_markers = np.where(np.array([np.all(PS_num[i:i+10]<3) for i in range(len(PS_num)-10)]))
	PS_start,PS_end = np.min(PS_markers),np.max(PS_markers)
	
	#Phi = Phi[::4,::4,PS_start:PS_end:5]
	#scipy.io.savemat(Phi_files[n],mdict={'Phi':Phi})
	
	PS  = PS[PS_start:PS_end:5,:]
	scipy.io.savemat(PS_files[n],mdict={'PS_trajectory':PS})