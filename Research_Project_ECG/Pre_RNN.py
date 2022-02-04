'''
https://physionet.org/challenge/2017/

8508 single lead ECG recordings (9s to 60s, varying lengths), 300 Hz
Signal lengths range from 2714 to 18286 (9s to 60s) - but this could be different in the test data

% classifyResult: integer value where
%                     N = normal rhythm (0)
%                     A = AF (1)
%                     O = other rhythm (2)
%                     ~ = noisy recording (poor signal quality) (3)
'''

'''
Listed below are some large datasets to pre-train a network for audio classification

Datasets:
	- Audioset - Google
	- DCASE 2013
	- DCASE 2016
	- ESC-50
	- UrbanSound
'''

### This module contains functions for signal preprocessing for the ECG Classification Challenge
### Functions in this module:
##
## QRS_Proccessing(data)
## Generate_Spectrogram_Patch()
## Generate_Spectrogram_Patch_MWFD()
## Generate_Spectrogram_Patch_For_Recurrent()
## find_optimal_patch_width()
## Generate_Signal_Segments()
## Data_Augmentation_2_Multi_Augmentation()
## Generate_Spectrogram_Patch_Vertical()
## Generate_Spectrogram_Patch_Horizontal_Overlaping()
## Generate_Spectrogram_Patch_Horizontal()
## Generate_Spectrogram()
## Generate_HDF5_Training_Data_1500_Lengths()
## Generate_HDF5_Training_Data_v1()
## Read_Validation_Set()
## Data_Augmentation_1_Stacking_Multi_Frequency_Resolution_Spectrogram()

import numpy as np
import os,os.path
import sys
import scipy.io
import pandas as pd
import time
import h5py
import random
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from sklearn.model_selection import train_test_split
from scipy import signal
import librosa
from biosppy.signals import ecg
import soundfile as sf

'''
makes raw input signals
'''
def Raw_Signal_Input(patch_width=1500):

	temp = scipy.io.loadmat("Dev-Test")

	data		= temp["train"][0]
	labels		= temp["train_lab"]
	
	test_data	= temp["test"][0]
	test_lab	= temp["test_lab"]
	
	# training data
	signal_data,signal_lab = [],[]

	for i in range(len(data)):
		
		temp = data[i][0,:]
		
		for j in range(0,len(temp)-patch_width+1,patch_width):
			signal_data.append(temp[j:(j+patch_width)])
			signal_lab.append(labels[i])
		
	signal_data,signal_lab = np.array(signal_data),np.array(signal_lab)
	
	signal_data = np.reshape(signal_data,[-1,patch_width,1])
	signal_data,_,signal_lab,_ = train_test_split(signal_data,signal_lab,test_size=0,random_state=random.randint(0,9999))
	
	# training mean and sd
	train_mean = np.mean(signal_data)
	train_sd = np.std(signal_data)
	signal_data = (signal_data-train_mean)/train_sd
	
	return(signal_data,signal_lab,test_data,test_lab,train_mean,train_sd)

def Raw_Signal_Input_Cross_Validation(patch_width=1500):

	temp = scipy.io.loadmat("Dev-Test")

	ECG_class = dict()
	ECG_class['N'],ECG_class['A'],ECG_class['O'],ECG_class['~'] = [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]
	
	info_doc = np.array(pd.read_csv("training2017/REFERENCE.csv",header=None))
	ECG = np.array([scipy.io.loadmat("training2017/"+info_doc[n,0]+".mat")["val"][0] for n in range(info_doc.shape[0])])
	label = np.array([ECG_class[info_doc[n,1]] for n in range(info_doc.shape[0])])
	
	data,test_data,labels,test_lab = train_test_split(ECG,label,test_size=0.3,random_state=np.random.randint(0,9999))
	scipy.io.savemat("C:/Users/zxio506/Desktop/log/Dev-Test",mdict={"train":data,"train_lab":labels,"test":test_data,"test_lab":test_lab})
	
	# training data
	signal_data,signal_lab = [],[]

	for i in range(len(data)):
		
		temp = data[i]
		
		for j in range(0,len(temp)-patch_width+1,patch_width):
			signal_data.append(temp[j:(j+patch_width)])
			signal_lab.append(labels[i])
		
	signal_data,signal_lab = np.array(signal_data),np.array(signal_lab)
	
	signal_data = np.reshape(signal_data,[-1,patch_width,1])
	signal_data,_,signal_lab,_ = train_test_split(signal_data,signal_lab,test_size=0,random_state=random.randint(0,9999))
	
	# training mean and sd
	train_mean = np.mean(signal_data)
	train_sd = np.std(signal_data)
	signal_data = (signal_data-train_mean)/train_sd
	
	return(signal_data,signal_lab,test_data,test_lab,train_mean,train_sd)
	
############################################################################################################################################################
### August 2017 and ealier:
############################################################################################################################################################

'''
add new patient data, takes the old training data and label and returns the updated training set
'''
def Add_New_Training_Data(data_old,label_old):
	
	
	data_old,label_old = data_old.tolist(),label_old.tolist()
	
	
	temp = scipy.io.loadmat("AF Termination Challenge")
	data_new = temp["data"][:,None,:]
	
	for i in range(len(data_new)):
		data_old.append(data_new[i])
		label_old.append([0,1,0,0])
	
	label_old = np.array(label_old)

	return(data_old,label_old)

'''
loads the data and then generates training spectrograms from it
''' 
def Training_Spectrogram(patch_width=550):

	temp = scipy.io.loadmat("Dev-Test")

	data		= temp["train"][0]
	test_data	= temp["test"][0]
	labels		= temp["train_lab"]
	test_lab	= temp["test_lab"]
	
	#data,labels = Augmentation_Speed_Change(data,labels)
	#data,labels = Add_New_Training_Data(data,labels)
	
	
	
	signal_data,signal_lab = [],[]
	for i in range(len(data)):
		
		temp = signal.spectrogram(data[i][0,:],fs=1,nfft=38,noverlap=0,nperseg=4)[2] # nfft changes height, N-over-lap/N-per-seg changes width
		
		factor = 1
		if np.argmax(labels[i]) == 1 or np.argmax(labels[i]) == 2: factor = 2 # for AF/Other over sampling
		
		for j in range(0,temp.shape[1]-patch_width,patch_width//factor):
			
			signal_data.append(temp[:,j:(j+patch_width)])
			signal_lab.append(labels[i])
		
	signal_data,signal_lab = np.array(signal_data),np.array(signal_lab)
	
	signal_data = np.expand_dims(signal_data,axis=3)
	signal_data,_,signal_lab,_ = train_test_split(signal_data,signal_lab,test_size=0,random_state=random.randint(0,9999))
	
	train_mean = np.mean(signal_data)
	train_sd = np.std(signal_data)
	signal_data = (signal_data-train_mean)/train_sd
	
	return(signal_data,signal_lab,test_data,test_lab,train_mean,train_sd)



'''
This function applies augmentation according to the "scale" list by speeding or slowing down the signal
'''
def Augmentation_Speed_Change(train,train_label):
	
	train,_,train_label,_ = train_test_split(train,train_label,test_size=0,random_state=np.random.randint(0,9999))
	
	temp_lab = np.array([np.argmax(train_label[i]) for i in range(len(train_label))])
	ECG_A,ECG_O = train[temp_lab==1],train[temp_lab==2]
	
	scale = [1.1,0.9] 															# tune the scale
	
	aug_A,aug_O,aug_A_label,aug_O_label = [],[],[],[]
	
	for s in scale:
		for i in random.sample([i for i in range(len(ECG_A))],len(ECG_A)):										# tune the number of signals augmented
			aug_A.append( signal.resample( ECG_A[i][0,:] , int(len(ECG_A[i][0,:])*s) )[None,:] )
			aug_A_label.append([0,1,0,0])
			
	for s in scale:
		for i in random.sample([i for i in range(len(ECG_O))],len(ECG_O)//3):									# tune the number of signals augmented
			aug_O.append( signal.resample( ECG_O[i][0,:] , int(len(ECG_O[i][0,:])*s) )[None,:] )
			aug_O_label.append([0,0,1,0])
	
	train,train_label = train.tolist(),train_label.tolist()
	
	train.extend(aug_A)				;train.extend(aug_O)
	train_label.extend(aug_A_label)	;train_label.extend(aug_O_label)
	
	aug_train_label = np.array(train_label)
	
	return(train,aug_train_label)
'''
This function adds noise to the spectrograms
'''	
def Augmentation_Add_Noise(spec_train,spec_train_label):	
	
	spec_train,_,spec_train_label,_ = train_test_split(spec_train,spec_train_label,test_size=0,random_state=np.random.randint(0,9999))
	
	temp_lab = np.array([np.argmax(spec_train_label[i]) for i in range(len(spec_train_label))])
	spec_A,spec_O = spec_train[temp_lab==1],spec_train[temp_lab==2]

	Scale_Factor = 10 													# tune this hyper parameter
	
	aug_A,aug_O,aug_A_label,aug_O_label = [],[],[],[]
	
	range_A = np.max(spec_A)-np.min(spec_A)
	for i in range(spec_A.shape[0]//2):									# tune the number of signals augmented

		temp = np.random.normal(loc=0,scale=range_A//Scale_Factor ,size=[spec_train.shape[1],spec_train.shape[2]])
		
		aug_A.append(temp+spec_A[i])
		aug_A_label.append([0,1,0,0])
		
	range_B = np.max(spec_O)-np.min(spec_O)
	for i in range(spec_O.shape[0]//3000):									# tune the number of signals augmented
		
		temp = np.random.normal(loc=0,scale=range_B//Scale_Factor ,size=[spec_train.shape[1],spec_train.shape[2]])
		
		aug_O.append(temp+spec_O[i])
		aug_O_label.append([0,0,1,0])
		
	spec_train,spec_train_label = spec_train.tolist(),spec_train_label.tolist()
	
	spec_train.extend(aug_A)				;spec_train.extend(aug_O)
	spec_train_label.extend(aug_A_label)	;spec_train_label.extend(aug_O_label)
	
	spec_train,spec_train_label = np.array(spec_train),np.array(spec_train_label)
	
	return(spec_train,spec_train_label)



'''
This function processes the .mat files for the ECG classification challenge (physionet 2017)

the raw signals are converted into spectrograms, cropped to specific widths and then returned
'''
# def Training_Data_Spectrogram(patch_width=400):

	# ECG_class = dict()
	# ECG_class['N'],ECG_class['A'],ECG_class['O'],ECG_class['~'] = [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]
	
	# info_doc = np.array(pd.read_csv("training2017/REFERENCE.csv",header=None))
	# ECG = np.array([scipy.io.loadmat("training2017/"+info_doc[n,0]+".mat")["val"][0] for n in range(info_doc.shape[0])])
	# label = np.array([ECG_class[info_doc[n,1]] for n in range(info_doc.shape[0])])
	
	# data,test_data,labels,test_lab = train_test_split(ECG,label,test_size=0.2,random_state=np.random.randint(0,9999))

	# # scipy.io.savemat("Dev-Test",mdict={"train":data,"train_lab":labels,"test":test_data,"test_lab":test_lab})

	# data,labels = Augmentation_Speed_Change(data,labels)
	
	# signal_data,signal_lab = [],[]
	# for i in range(len(data)):
		
		# temp = signal.spectrogram(data[i],fs=1,nfft=38,noverlap=0,nperseg=4)[2] # nfft changes height, N-over-lap/N-per-seg changes width
		
		# for j in range(0,temp.shape[1]-patch_width,patch_width):
			
			# signal_data.append(temp[:,j:(j+patch_width)])
			# signal_lab.append(labels[i])
		
	# signal_data,signal_lab = np.array(signal_data),np.array(signal_lab)
	
	# signal_data,signal_lab = Augmentation_Add_Noise(signal_data,signal_lab)
	
	# signal_data = np.expand_dims(signal_data,axis=3)
	# signal_data,_,signal_lab,_ = train_test_split(signal_data,signal_lab,test_size=0,random_state=random.randint(0,9999))
	
	# train_mean = np.mean(signal_data)
	# train_sd = np.std(signal_data)
	# signal_data = (signal_data-train_mean)/train_sd
	
	# return(signal_data,signal_lab,test_data,test_lab,train_mean,train_sd)



############################################################################################################################################################
### May 2017 and ealier:
############################################################################################################################################################
'''
This function generates the training data for only 3 classes (noisy omitted)
'''
def Generate_Spectrogram_Patch_3_class():
	
	patch_width = 400
	
	t0 = time.time()
	
	ECG_class = dict()
	ECG_class['N'] = [1,0,0]
	ECG_class['A'] = [0,1,0]
	ECG_class['O'] = [0,0,1]
	
	### Reading Data
	ECG_signal = []
	ECG_labels = []
	
	info_doc_train = np.array(pd.read_csv(os.getcwd()+"/training2017/REFERENCE.csv",header=None))
	for n in range(info_doc_train.shape[0]):
		file_in = os.getcwd()+"/training2017/"+info_doc_train[n,0]+".mat"
		ECG_signal.append(scipy.io.loadmat(file_in)["val"][0])
		ECG_labels.append(info_doc_train[n,1])
	
	ECG_signal, ECG_labels  = np.array(ECG_signal), np.array(ECG_labels)
	#ECG_signal = np.array(ECG_signal);ECG_labels = np.array(ECG_labels);return(ECG_signal,ECG_labels)

## ---------------------------------------------------- Filtering out noisy signals -------------------------------------------------------------

	ECG_signal = ECG_signal[ECG_labels != "~"]
	ECG_labels = ECG_labels[ECG_labels != "~"]

## ----------------------------------------------------------------------------------------------------------------------------------------------

	### Splitting into train/train val/val sets
	dev_sig,test_sig,dev_label,test_label = train_test_split(ECG_signal,ECG_labels,test_size=0.2,random_state=random.randint(0,9999))
	_,dev_test_sig,_,dev_test_label = train_test_split(dev_sig,dev_label,test_size=0.05,random_state=random.randint(0,9999))

	dev_label = np.array([ECG_class[dev_label[i]] for i in range(len(dev_label))])
	test_label = np.array([ECG_class[test_label[i]] for i in range(len(test_label))])
	dev_test_label = np.array([ECG_class[dev_test_label[i]] for i in range(len(dev_test_label))])
	
	### Generating Training Spectrograms        
	spectrogram = [None]*len(dev_sig)
	for n in range(len(dev_sig)):
	
		data = dev_sig[n]
		spectrogram[n] = signal.spectrogram(data[500:],fs=1,nfft=38,noverlap=0,nperseg=4)[2]	# 20 x ___

	spec_patches = []
	spec_labels = []
	
	for n in range(len(spectrogram)):
		for i in range(0,spectrogram[n].shape[1]-patch_width,patch_width):
			spec_patches.append(spectrogram[n][:,i:(i+patch_width)])
			spec_labels.append(dev_label[n])

	spec_patches = np.array(spec_patches)
	spec_labels = np.array(spec_labels)

	### Development Data
	spec_patches = np.reshape(spec_patches,newshape=[spec_patches.shape[0],spec_patches.shape[1],spec_patches.shape[2],1])
	spec_patches,_,spec_labels,_ = train_test_split(spec_patches,spec_labels,test_size=0,random_state=random.randint(0,9999))
	
	### Normalising Spectrograms
	train_mean = np.mean(spec_patches)
	train_sd = np.sqrt(np.var(spec_patches))
	spec_patches = (spec_patches-train_mean)/train_sd
	
	### Developemnt Test Data
	max_length = np.max([len(dev_test_sig[i]) for i in range(len(dev_test_sig))])
	dev_test_sig = np.array([np.pad(dev_test_sig[i],(0,max_length-len(dev_test_sig[i])),"constant") for i in range(len(dev_test_sig))])	
	
	### Test Data
	max_length = np.max([len(test_sig[i]) for i in range(len(test_sig))])
	test_sig = np.array([np.pad(test_sig[i],(0,max_length-len(test_sig[i])),"constant") for i in range(len(test_sig))])
	
	### Writing to HDF5 file
	ECG_Sig = h5py.File('C:/Users/zxio506/Desktop/ECG Spectrogram Patch '+str(20)+'x'+str(patch_width)+'.h5','w')
	
	ECG_Sig.create_dataset("dev",					data = spec_patches)
	ECG_Sig.create_dataset("dev.label",				data = spec_labels)
	ECG_Sig.create_dataset("dev.mean",				data = train_mean)
	ECG_Sig.create_dataset("dev.sd",				data = train_sd)
	ECG_Sig.create_dataset("dev-test",				data = dev_test_sig)
	ECG_Sig.create_dataset("dev-test.label",		data = dev_test_label)	
	ECG_Sig.create_dataset("test",					data = test_sig)
	ECG_Sig.create_dataset("test.label",			data = test_label)
	
	ECG_Sig.close()
	
	print("Time Taken:",np.round(time.time()-t0,2),"Seconds")

	
	
'''
This function takes ECG signals and returns the location of the R peaks as well as other 
preprocessing steps. this function is good shit
'''
def QRS_Proccessing(data,sample_rate=300,show_graph=False):
	
	out = ecg.ecg(signal=data, sampling_rate=sample_rate, show=show_graph)
	
	ts 				= out[0] 	# time scale: x = sample, y = time in secs
	filtered 		= out[1] 	# filtered ECG signal
	rpeaks 			= out[2] 	# positions of R peaks
	templates_ts 	= out[3] 	# time scale of template
	templates 		= out[4] 	# cycles of the wave
	heart_rate_ts 	= out[5] 	# heart rate time scale
	heart_rate 		= out[6] 	# heart rate

	return(ts,filtered,rpeaks,templates_ts,templates,heart_rate_ts,heart_rate)



'''
This function produces spectrograms for the dataet
'''
def Generate_Spectrogram_Patch():
	
	patch_width = 400
	
	t0 = time.time()
	
	ECG_class = dict()
	ECG_class['N'] = [1,0,0,0]
	ECG_class['A'] = [0,1,0,0]
	ECG_class['O'] = [0,0,1,0]
	ECG_class['~'] = [0,0,0,1]
	
	### Reading Data
	ECG_signal = []
	ECG_labels = []
	
	info_doc_train = np.array(pd.read_csv(os.getcwd()+"/training2017/REFERENCE.csv",header=None))
	for n in range(info_doc_train.shape[0]):
		file_in = os.getcwd()+"/training2017/"+info_doc_train[n,0]+".mat"
		ECG_signal.append(scipy.io.loadmat(file_in)["val"][0])
		ECG_labels.append(info_doc_train[n,1])
	
	#ECG_signal = np.array(ECG_signal);ECG_labels = np.array(ECG_labels);return(ECG_signal,ECG_labels)

## ----------------------------------------------------------------------------------------------------------------------------------------------
## Preprocessing (QRS)
	
	#ECG_signal = np.array([QRS_Proccessing(ECG_signal[i])[1] for i in range(len(ECG_signal))]) # need to include this in the prediction step then

## ----------------------------------------------------------------------------------------------------------------------------------------------


	### Splitting into train/train val/val sets
	dev_sig,test_sig,dev_label,test_label = train_test_split(ECG_signal,ECG_labels,test_size=0.2,random_state=random.randint(0,9999))
	_,dev_test_sig,_,dev_test_label = train_test_split(dev_sig,dev_label,test_size=0.05,random_state=random.randint(0,9999))

	dev_label = np.array([ECG_class[dev_label[i]] for i in range(len(dev_label))])
	test_label = np.array([ECG_class[test_label[i]] for i in range(len(test_label))])
	dev_test_label = np.array([ECG_class[dev_test_label[i]] for i in range(len(dev_test_label))])
	
	### Generating Training Spectrograms        
	spectrogram = [None]*len(dev_sig)
	for n in range(len(dev_sig)):
		data = dev_sig[n]
		#spectrogram[n] = signal.spectrogram(data[500:],fs=1,nfft=18,noverlap=0,nperseg=4)[2]	# 10 x ___
		spectrogram[n] = signal.spectrogram(data[500:],fs=1,nfft=38,noverlap=0,nperseg=4)[2]	# 20 x ___
		#spectrogram[n] = signal.spectrogram(data[500:],fs=1,nfft=78,noverlap=0,nperseg=4)[2]	# 40 x ___
		#spectrogram[n] = signal.spectrogram(data[500:],fs=1,nfft=99,noverlap=0,nperseg=4)[2]	# 50 x ___
		
	spec_patches = []
	spec_labels = []
	
	for n in range(len(spectrogram)):
		for i in range(0,spectrogram[n].shape[1]-patch_width,patch_width):
			spec_patches.append(spectrogram[n][:,i:(i+patch_width)])
			spec_labels.append(dev_label[n])

	spec_patches = np.array(spec_patches)
	spec_labels = np.array(spec_labels)

	### Development Data
	spec_patches = np.reshape(spec_patches,newshape=[spec_patches.shape[0],spec_patches.shape[1],spec_patches.shape[2],1])
	spec_patches,_,spec_labels,_ = train_test_split(spec_patches,spec_labels,test_size=0,random_state=random.randint(0,9999))
	
	### Normalising Spectrograms
	train_mean = np.mean(spec_patches)
	train_sd = np.sqrt(np.var(spec_patches))
	spec_patches = (spec_patches-train_mean)/train_sd
	
	### Developemnt Test Data
	max_length = np.max([len(dev_test_sig[i]) for i in range(len(dev_test_sig))])
	dev_test_sig = np.array([np.pad(dev_test_sig[i],(0,max_length-len(dev_test_sig[i])),"constant") for i in range(len(dev_test_sig))])	
	
	### Test Data
	max_length = np.max([len(test_sig[i]) for i in range(len(test_sig))])
	test_sig = np.array([np.pad(test_sig[i],(0,max_length-len(test_sig[i])),"constant") for i in range(len(test_sig))])
	
	### Writing to HDF5 file
	ECG_Sig = h5py.File('C:/Users/zxio506/Desktop/ECG Spectrogram Patch '+str(20)+'x'+str(patch_width)+'.h5','w')
	
	ECG_Sig.create_dataset("dev",					data = spec_patches)
	ECG_Sig.create_dataset("dev.label",				data = spec_labels)
	ECG_Sig.create_dataset("dev.mean",				data = train_mean)
	ECG_Sig.create_dataset("dev.sd",				data = train_sd)
	ECG_Sig.create_dataset("dev-test",				data = dev_test_sig)
	ECG_Sig.create_dataset("dev-test.label",		data = dev_test_label)	
	ECG_Sig.create_dataset("test",					data = test_sig)
	ECG_Sig.create_dataset("test.label",			data = test_label)
	
	ECG_Sig.close()
	
	print("Time Taken:",np.round(time.time()-t0,2),"Seconds")


'''
This function does some simple analysis to find which patch width is the optimal to use
'''
def find_optimal_patch_width():

	info_doc = np.array(pd.read_csv("training2017/REFERENCE.csv",header=None))
	ECG = np.array([scipy.io.loadmat("training2017/"+info_doc[n,0]+".mat")["val"][0] for n in range(info_doc.shape[0])])
	label = np.array([info_doc[n,1] for n in range(info_doc.shape[0])])

	# spectrogram segments
	lengths = np.array([signal.spectrogram(ECG[n],fs=1,nfft=38,noverlap=0,nperseg=4)[2].shape[1] for n in range(len(ECG))])
	n_patch = np.array([np.sum(lengths//i) for i in range(200,2000)])
	w = np.array([281,321,375,450,562,750,1125])

	plt.plot(np.ones([2])*np.min(lengths),[-1000,np.max(n_patch)+10000],"--")
	plt.plot([i for i in range(200,2000)],n_patch)

	for i in range(len(w)):
		plt.plot([w[i]],[n_patch[w[i]-200]], marker='o', markersize=3, color="red")
		plt.text(w[i]+5,n_patch[w[i]-200],"("+str(n_patch[w[i]])+", "+str(w[i])+")",fontsize=12)
		
	plt.show()

	
	
	# raw signal segments
	lengths = np.array([len(ECG[n]) for n in range(len(ECG))])
	n_patch = np.array([np.sum(lengths//i) for i in range(200,3000)])
	
	plt.plot(np.ones([2])*np.min(lengths),[-1000,np.max(n_patch)+10000],"--")
	plt.plot([i for i in range(200,3000)],n_patch)
	plt.show()



'''
This function generates spectrograms of size 20 x ____ (patch width = 200) and produces the MWFD features resulting in a 
3 channel spectrogram patch input to the CNN
'''
def Generate_Spectrogram_Patch_MWFD():

	patch_width = 400
	
	t0 = time.time()
	
	ECG_class = dict()
	ECG_class['N'] = [1,0,0,0]
	ECG_class['A'] = [0,1,0,0]
	ECG_class['O'] = [0,0,1,0]
	ECG_class['~'] = [0,0,0,1]
	
	### Reading Data
	ECG_signal = []
	ECG_labels = []
	
	info_doc_train = np.array(pd.read_csv(os.getcwd()+"/training2017/REFERENCE.csv",header=None))
	for n in range(info_doc_train.shape[0]):
		file_in = os.getcwd()+"/training2017/"+info_doc_train[n,0]+".mat"
		ECG_signal.append(scipy.io.loadmat(file_in)["val"][0])
		ECG_labels.append(info_doc_train[n,1])
	
	### Splitting into train/train val/val sets
	dev_sig,test_sig,dev_label,test_label = train_test_split(ECG_signal,ECG_labels,test_size=0.2,random_state=random.randint(0,9999))
	_,dev_test_sig,_,dev_test_label = train_test_split(dev_sig,dev_label,test_size=0.05,random_state=random.randint(0,9999))

	dev_label = np.array([ECG_class[dev_label[i]] for i in range(len(dev_label))])
	test_label = np.array([ECG_class[test_label[i]] for i in range(len(test_label))])
	dev_test_label = np.array([ECG_class[dev_test_label[i]] for i in range(len(dev_test_label))])

	### Generating Training Spectrograms
	spectrogram = []
	for n in range(len(dev_sig)):

		data = dev_sig[n]
		spectrogram.append(signal.spectrogram(data[500:],fs=1,nfft=38,noverlap=0,nperseg=4)[2])	# 20 x ___

	spec_patches = []
	spec_labels = []
	
	for n in range(len(spectrogram)):
		for i in range(0,spectrogram[n].shape[1]-patch_width,patch_width):
			
			data = spectrogram[n][:,i:(i+patch_width)]

			### MFWD
			spec_patches.append(data);									spec_labels.append(dev_label[n])
			spec_patches.append(librosa.feature.delta(data,width=3));	spec_labels.append(dev_label[n])
			#spec_patches.append(librosa.feature.delta(data,width=5));	spec_labels.append(dev_label[n])
			
			### MWFD Stacked
			#data = np.stack((data,librosa.feature.delta(data,width=3),librosa.feature.delta(data,width=5)),2)
			#spec_patches.append(data);spec_labels.append(dev_label[n])
			
	spec_patches = np.array(spec_patches)
	spec_labels = np.array(spec_labels)
	
	### Development Data
	spec_patches = np.reshape(spec_patches,newshape=[spec_patches.shape[0],spec_patches.shape[1],spec_patches.shape[2],1])
	spec_patches,_,spec_labels,_ = train_test_split(spec_patches,spec_labels,test_size=0,random_state=random.randint(0,9999))

	### Normalising Spectrograms
	dev_mean = np.mean(spec_patches)
	dev_sd = np.sqrt(np.var(spec_patches))
	spec_patches = (spec_patches-dev_mean)/dev_sd
	
	### Development Test Data
	max_length = np.max([len(dev_test_sig[i]) for i in range(len(dev_test_sig))])
	dev_test_sig = np.array([np.pad(dev_test_sig[i],(0,max_length-len(dev_test_sig[i])),"constant") for i in range(len(dev_test_sig))])	
	
	### Test Data
	max_length = np.max([len(test_sig[i]) for i in range(len(test_sig))])
	test_sig = np.array([np.pad(test_sig[i],(0,max_length-len(test_sig[i])),"constant") for i in range(len(test_sig))])
	
	### Writing to HDF5 file
	ECG_Sig = h5py.File('ECG Spectrogram Patch '+str(20)+'x'+str(patch_width)+' MWFD.h5','w')
	
	ECG_Sig.create_dataset("dev",					data = spec_patches)
	ECG_Sig.create_dataset("dev.label",				data = spec_labels)
	ECG_Sig.create_dataset("dev.mean",				data = dev_mean)
	ECG_Sig.create_dataset("dev.sd",				data = dev_sd)
	ECG_Sig.create_dataset("dev-test",				data = dev_test_sig)
	ECG_Sig.create_dataset("dev-test.label",		data = dev_test_label)
	ECG_Sig.create_dataset("test",					data = test_sig)
	ECG_Sig.create_dataset("test.label",			data = test_label)
	ECG_Sig.close()
	
	print("Time Taken:",np.round(time.time()-t0,2),"Seconds")



'''
This function generates the spectrograms patches, as well as the corresponding full signals
'''
def Generate_Spectrogram_Patch_For_Recurrent():
	
	patch_width = 400
	
	t0 = time.time()
	
	ECG_class = dict()
	ECG_class['N'] = [1,0,0,0]
	ECG_class['A'] = [0,1,0,0]
	ECG_class['O'] = [0,0,1,0]
	ECG_class['~'] = [0,0,0,1]
	
	### Reading Data
	ECG_signal = []
	ECG_labels = []
	
	info_doc_train = np.array(pd.read_csv(os.getcwd()+"/training2017/REFERENCE.csv",header=None))
	for n in range(info_doc_train.shape[0]):
		file_in = os.getcwd()+"/training2017/"+info_doc_train[n,0]+".mat"
		ECG_signal.append(scipy.io.loadmat(file_in)["val"][0])
		ECG_labels.append(info_doc_train[n,1])

	### Splitting into train/train val/val sets
	dev_sig,test_sig,dev_label,test_label = train_test_split(ECG_signal,ECG_labels,test_size=0.2,random_state=random.randint(0,9999))
	_,dev_test_sig,_,dev_test_label = train_test_split(dev_sig,dev_label,test_size=0.05,random_state=random.randint(0,9999))

	dev_label = np.array([ECG_class[dev_label[i]] for i in range(len(dev_label))])
	test_label = np.array([ECG_class[test_label[i]] for i in range(len(test_label))])
	dev_test_label = np.array([ECG_class[dev_test_label[i]] for i in range(len(dev_test_label))])
	
	### Generating Training Spectrograms        
	spectrogram = [None]*len(dev_sig)
	for n in range(len(dev_sig)):
		spectrogram[n] = signal.spectrogram(dev_sig[n][500:],fs=1,nfft=38,noverlap=0,nperseg=4)[2]

	concat_dev_sig = [item for sublist in dev_sig for item in sublist] # finding the mean of the dev data
	seg_mean = np.mean(concat_dev_sig)
	seg_sd = np.std(concat_dev_sig)
	
	max_len = np.round(np.max([len(ECG_signal[n]) for n in range(len(ECG_signal))]),-1)
	dev_sig = pad_sequences(dev_sig,maxlen=max_len) # padding dev_sig for dimensional consistency

	spec_patches = []
	spec_labels = []
	sig_seg = [] # signal segments
	
	for n in range(len(spectrogram)):
		for i in range(0,spectrogram[n].shape[1]-patch_width,patch_width):
			spec_patches.append(spectrogram[n][:,i:(i+patch_width)])
			
			sig_seg.append(dev_sig[n][500:]) # add the corresponding dev_sig to every spectrogram patch
			spec_labels.append(dev_label[n])
			
	spec_patches = np.array(spec_patches)
	spec_labels = np.array(spec_labels)
	sig_seg = np.array(sig_seg)
	
	### Development Data
	spec_patches = np.reshape(spec_patches,newshape=[spec_patches.shape[0],spec_patches.shape[1],spec_patches.shape[2],1])
	
	### Normalising Spectrograms
	train_mean = np.mean(spec_patches)
	train_sd = np.sqrt(np.var(spec_patches))
	spec_patches = (spec_patches-train_mean)/train_sd

	
	## Signal Segments:
	### ******************************************************************************************************************************************
	### Normalising
	sig_seg = (sig_seg-seg_mean)/seg_sd
	
	### Reshaping to recurrent neural network input
	n_chunk = 30
	chunk_size = 593
	sig_seg = np.reshape(sig_seg,[sig_seg.shape[0],n_chunk,chunk_size])
	### ******************************************************************************************************************************************
	
	
	### Developemnt Test Data
	dev_test_sig = pad_sequences(dev_test_sig,maxlen=max_len)
	
	### Test Data
	test_sig = pad_sequences(test_sig,maxlen=max_len)
	
	### Writing to HDF5 file
	ECG_Sig = h5py.File('ECG Spectrogram Patch '+str(20)+'x'+str(patch_width)+' Recurrent.h5','w')
	
	ECG_Sig.create_dataset("dev",					data = spec_patches)
	ECG_Sig.create_dataset("dev.label",				data = spec_labels)
	ECG_Sig.create_dataset("dev.mean",				data = train_mean)
	ECG_Sig.create_dataset("dev.sd",				data = train_sd)
	ECG_Sig.create_dataset("dev-test",				data = dev_test_sig)
	ECG_Sig.create_dataset("dev-test.label",		data = dev_test_label)	

	ECG_Sig.create_dataset("dev-raw",				data = sig_seg)	
	ECG_Sig.create_dataset("dev-raw.mean",			data = seg_mean)
	ECG_Sig.create_dataset("dev-raw.sd",			data = seg_sd)
	
	ECG_Sig.create_dataset("test",					data = test_sig)
	ECG_Sig.create_dataset("test.label",			data = test_label)
	
	ECG_Sig.close()
	
	print("Time Taken:",np.round(time.time()-t0,2),"Seconds")



'''
This function generates signaal segments as 1D vectos for the neural network to train on
'''
def Generate_Signal_Segments():
	
	patch_width = 1500
	
	t0 = time.time()
	
	ECG_class = dict()
	ECG_class['N'] = [1,0,0,0]
	ECG_class['A'] = [0,1,0,0]
	ECG_class['O'] = [0,0,1,0]
	ECG_class['~'] = [0,0,0,1]
	
	### Reading Data
	ECG_signal = []
	ECG_labels = []
	
	info_doc_train = np.array(pd.read_csv(os.getcwd()+"/training2017/REFERENCE.csv",header=None))
	for n in range(info_doc_train.shape[0]):
		file_in = os.getcwd()+"/training2017/"+info_doc_train[n,0]+".mat"
		ECG_signal.append(scipy.io.loadmat(file_in)["val"][0])
		ECG_labels.append(info_doc_train[n,1])
	
	### Splitting into train/train val/val sets
	dev_sig,test_sig,dev_label,test_label = train_test_split(ECG_signal,ECG_labels,test_size=0.2,random_state=random.randint(0,9999))
	_,dev_test_sig,_,dev_test_label = train_test_split(dev_sig,dev_label,test_size=0.05,random_state=random.randint(0,9999))

	dev_label = np.array([ECG_class[dev_label[i]] for i in range(len(dev_label))])
	test_label = np.array([ECG_class[test_label[i]] for i in range(len(test_label))])
	dev_test_label = np.array([ECG_class[dev_test_label[i]] for i in range(len(dev_test_label))])
	
	### Generating Training Segments        
	spec_patches = []
	spec_labels = []
	
	for n in range(len(dev_sig)):
		data = np.array(dev_sig[n])
		for i in range(0,data.shape[0]-patch_width,patch_width):
			spec_patches.append(data[i:(i+patch_width)])
			spec_labels.append(dev_label[n])

	spec_patches = np.array(spec_patches)
	spec_labels = np.array(spec_labels)

	### Development Data
	spec_patches = np.reshape(spec_patches,newshape=[spec_patches.shape[0],spec_patches.shape[1],1,1])
	spec_patches,_,spec_labels,_ = train_test_split(spec_patches,spec_labels,test_size=0,random_state=random.randint(0,9999))
	
	### Normalising
	train_mean = np.mean(spec_patches)
	train_sd = np.sqrt(np.var(spec_patches))
	spec_patches = (spec_patches-train_mean)/train_sd
	
	### Developemnt Test Data
	max_length = np.max([len(dev_test_sig[i]) for i in range(len(dev_test_sig))])
	dev_test_sig = np.array([np.pad(dev_test_sig[i],(0,max_length-len(dev_test_sig[i])),"constant") for i in range(len(dev_test_sig))])	
	
	### Test Data
	max_length = np.max([len(test_sig[i]) for i in range(len(test_sig))])
	test_sig = np.array([np.pad(test_sig[i],(0,max_length-len(test_sig[i])),"constant") for i in range(len(test_sig))])
	
	### Writing to HDF5 file
	ECG_Sig = h5py.File('ECG Segments '+str(patch_width)+'x1.h5','w')
	
	ECG_Sig.create_dataset("dev",					data = spec_patches)
	ECG_Sig.create_dataset("dev.label",				data = spec_labels)
	ECG_Sig.create_dataset("dev.mean",				data = train_mean)
	ECG_Sig.create_dataset("dev.sd",				data = train_sd)
	ECG_Sig.create_dataset("dev-test",				data = dev_test_sig)
	ECG_Sig.create_dataset("dev-test.label",		data = dev_test_label)	
	ECG_Sig.create_dataset("test",					data = test_sig)
	ECG_Sig.create_dataset("test.label",			data = test_label)
	
	ECG_Sig.close()
	
	print("Time Taken:",np.round(time.time()-t0,2),"Seconds")
	
	

'''
This function applies the multiple width frequency delta data augmentation technique in the paper:
"Acoustic scene classification using convolutional neural netwroks ad multiple width frequency delta data augmentation"
The output will be a 4 layer patches of spectrograms with different frequency resolutios

There includes some helper functions to shift the signal in the time and frequency domains
'''
def speedx(sound_array,factor):
    #Multiplies the sound's speed by some `factor`
    indices = np.round(np.arange(0,len(sound_array),factor))
    indices = indices[indices<len(sound_array)].astype(int)
    return sound_array[indices.astype(int)]
def Data_Augmentation_2_Multi_Augmentation(data,label):
	
	t0 = time.time()
	
	### Seperating into classes
	data = np.array(data)
	label = np.array(label)
	ECG_N = data[np.where(label=='N')].tolist()
	ECG_A = data[np.where(label=='A')].tolist()
	ECG_O = data[np.where(label=='O')].tolist()
	ECG_n = data[np.where(label=='~')].tolist()

	### Augmentation
	shift = [0.81,0.93,1.07,1.23]
	for i in range(len(ECG_A)):
		for s in shift:
			ECG_A.append(speedx(ECG_A[i],s))

	for i in range(len(ECG_O)):
		ECG_O.append(speedx(ECG_O[i],1.15))
	
	for i in range(len(ECG_n)):
		for s in shift:
			ECG_n.append(speedx(ECG_n[i],s))
	
	#l = len(ECG_n)
	#for i in range(l):
	#	ind = random.sample([i for i in range(l)],5)
	#	for j in ind:
	#		
	#		if len(ECG_n[i]) > len(ECG_n[j]):
	#			ECG_n.append(ECG_n[i][:len(ECG_n[j])]+ECG_n[j])
	#		else:
	#			ECG_n.append(ECG_n[i]+ECG_n[j][:len(ECG_n[i])])
	
	signal = []
	signal.extend(ECG_N)
	signal.extend(ECG_A)
	signal.extend(ECG_O)
	signal.extend(ECG_n)
	signal = np.array(signal)
	
	labels = []	
	labels.extend(np.repeat('N',len(ECG_N)))
	labels.extend(np.repeat('A',len(ECG_A)))
	labels.extend(np.repeat('O',len(ECG_O)))
	labels.extend(np.repeat('~',len(ECG_n)))
	labels = np.array(labels)
	
	print("Image Augmentation Set Generated: ",time.time()-t0,"seconds")
	
	return(signal,labels)	



'''
This function generates patches of data for the spectrogram, taking into account class imbalance
'''
def Generate_Spectrogram_Patch_Vertical():

	t0 = time.time()
	
	ECG_class = dict()
	ECG_class['N'] = [1,0,0,0]
	ECG_class['A'] = [0,1,0,0]
	ECG_class['O'] = [0,0,1,0]
	ECG_class['~'] = [0,0,0,1]
	
	### Reading Data
	ECG_signal = []
	ECG_labels = []
	
	info_doc_train = np.array(pd.read_csv(os.getcwd()+"/training2017/REFERENCE.csv",header=None))
	for n in range(info_doc_train.shape[0]):
		file_in = os.getcwd()+"/training2017/"+info_doc_train[n,0]+".mat"
		ECG_signal.append(scipy.io.loadmat(file_in)["val"][0])
		ECG_labels.append(ECG_class[info_doc_train[n,1]])

	### Splitting into train/val sets
	ECG_train_sig,ECG_val_sig,ECG_train_label,ECG_val_label = train_test_split(ECG_signal,ECG_labels,test_size=0.2,random_state=32)

	### Generating Training Spectrograms
	spec_patches = []
	spec_labels = []
	
	patch_width = 300

	for n in range(len(ECG_train_sig)):

		data = ECG_train_sig[n][500:]
		
		#Pxx,_,_,_ = plt.specgram(data,NFFT=64,Fs=300,pad_to=256,noverlap=0)		# 129 x ___
		#Pxx,_,_,_ = plt.specgram(data,NFFT=8,Fs=300,pad_to=18,noverlap=0)			# 10  x ___
		#Pxx,_,_,_ = plt.specgram(data,NFFT=8,Fs=300,pad_to=39,noverlap=0)			# 20  x ___
		#Pxx,_,_,_ = plt.specgram(data,NFFT=8,Fs=300,pad_to=99,noverlap=0)			# 50  x ___
		#Pxx,_,_,_ = plt.specgram(data,NFFT=4,Fs=300,pad_to=18,noverlap=0)			# 10  x ___ (higher resolution)
		Pxx,_,_,_ = plt.specgram(data,NFFT=4,Fs=300,pad_to=39,noverlap=0)			# 20  x ___ (higher resolution)
		#Pxx,_,_,_ = plt.specgram(data,NFFT=4,Fs=300,pad_to=99,noverlap=0)			# 50  x ___ (higher resolution)
		#Pxx,_,_,_ = plt.specgram(data,NFFT=128,Fs=300,pad_to=39,noverlap=127)		# 10  x ___ (Very high resolution)
		#Pxx,_,_,_ = plt.specgram(data,NFFT=8,Fs=300,pad_to=19,noverlap=7)			# 10  x ___ (Very high resolution)
		
		for i in range(0,Pxx.shape[1]-patch_width,patch_width):

			spec_patches.append(Pxx[:,i:(i+patch_width)])
			spec_labels.append(ECG_train_label[n])
		
		print(n)
	
	spec_patches = np.array(spec_patches)
	spec_labels = np.array(spec_labels)
	
	spec_patches = np.reshape(spec_patches,newshape=[spec_patches.shape[0],spec_patches.shape[1],spec_patches.shape[2],1])
	spec_labels = np.reshape(spec_labels,newshape=[spec_labels.shape[0],4])
	spec_patches,_,spec_labels,_ = train_test_split(spec_patches,spec_labels,test_size=0,random_state=random.randint(0,9999))

	max_length = np.max([len(ECG_train_sig[i]) for i in range(len(ECG_train_sig))])
	ECG_train_sig = np.array([np.pad(ECG_train_sig[i],(0,max_length-len(ECG_train_sig[i])),"constant") for i in range(len(ECG_train_sig))])
	
	### Validation Data Generation
	max_length = np.max([len(ECG_val_sig[i]) for i in range(len(ECG_val_sig))])
	ECG_val_sig = np.array([np.pad(ECG_val_sig[i],(0,max_length-len(ECG_val_sig[i])),"constant") for i in range(len(ECG_val_sig))])
	
	### Normalising
	train_mean = np.mean(spec_patches)
	train_sd = np.sqrt(np.var(spec_patches))
	spec_patches = (spec_patches-train_mean)/train_sd
	
	### Writing to HDF5 file
	ECG_Sig = h5py.File('ECG Spectrogram Patch 20x300.h5','w')
	
	ECG_Sig.create_dataset("train",					data = spec_patches)
	ECG_Sig.create_dataset("train.label",			data = spec_labels)
	ECG_Sig.create_dataset("train.raw",				data = ECG_train_sig)
	
	ECG_Sig.create_dataset("train.mean",			data = train_mean)
	ECG_Sig.create_dataset("train.sd",				data = train_sd)
	
	ECG_Sig.create_dataset("traindev",				data = ECG_train_sig[0:int(len(ECG_train_sig)*0.01),:])
	ECG_Sig.create_dataset("traindev.label",		data = ECG_train_label[0:int(len(ECG_train_sig)*0.01)])	
	
	ECG_Sig.create_dataset("dev",					data = ECG_val_sig)
	ECG_Sig.create_dataset("dev.label",				data = ECG_val_label)
	
	ECG_Sig.close()
	
	print("Time Taken:",np.round((time.time()-t0)/60,2),"Minutes")



'''
This function does the same as Generage_Spectrogram_Patch_Horizontal except the patches are now over lapping
'''
def Generate_Spectrogram_Patch_Horizontal_Overlaping():

	t0 = time.time()
	
	ECG_class = dict()
	ECG_class['N'] = [1,0,0,0]
	ECG_class['A'] = [0,1,0,0]
	ECG_class['O'] = [0,0,1,0]
	ECG_class['~'] = [0,0,0,1]
	
	### Reading Data
	ECG_signal = []
	ECG_labels = []
	
	info_doc_train = np.array(pd.read_csv(os.getcwd()+"/training2017/REFERENCE.csv",header=None))
	for n in range(info_doc_train.shape[0]):
		file_in = os.getcwd()+"/training2017/"+info_doc_train[n,0]+".mat"
		ECG_signal.append(scipy.io.loadmat(file_in)["val"][0])
		ECG_labels.append(ECG_class[info_doc_train[n,1]])
	
	### Splitting into train/val sets
	ECG_train_sig,ECG_val_sig,ECG_train_label,ECG_val_label = train_test_split(ECG_signal,ECG_labels,test_size=0.2,random_state=32)
	
	### Generating Training Spectrograms
	spec_patches = []
	spec_labels = []

	for n in range(len(ECG_train_sig)):

		data = ECG_train_sig[n][500:]


		Pxx,_,_,_ = plt.specgram(data,NFFT=64,Fs=300,pad_to=256,noverlap=0)
		
		if Pxx.shape[1]<132:
			Pxx = np.tile(Pxx,int(np.ceil(132/Pxx.shape[1])))[:,0:132]
			
		for i in range(0,Pxx.shape[0]-20,20):
			for j in range(0,Pxx.shape[1]-131,20):
			
				spec_patches.append(Pxx[i:(i+20),j:(j+132)])
				spec_labels.append(ECG_train_label[n])

		print(n)
		
	spec_patches = np.array(spec_patches)
	spec_labels = np.array(spec_labels)
	
	spec_patches = np.reshape(spec_patches,newshape=[spec_patches.shape[0],spec_patches.shape[1],spec_patches.shape[2],1])
	spec_labels = np.reshape(spec_labels,newshape=[spec_labels.shape[0],4])

	### Validation Data Generation
	max_length = np.max([len(ECG_val_sig[i]) for i in range(len(ECG_val_sig))])
	ECG_val_sig = np.array([np.pad(ECG_val_sig[i],(0,max_length-len(ECG_val_sig[i])),"constant") for i in range(len(ECG_val_sig))])
	
	### Normalising
	train_mean = np.mean(spec_patches)
	train_sd = np.sqrt(np.var(spec_patches))
	spec_patches = (spec_patches-train_mean)/train_sd
	
	### Writing to HDF5 file
	ECG_Sig = h5py.File('ECG Spectrogram Patch Horizontal Overlapping.h5','w')
	ECG_Sig.create_dataset("train.signal.spctrgm",	data = spec_patches)
	ECG_Sig.create_dataset("train.label.spctrgm",	data = spec_labels)
	ECG_Sig.create_dataset("train.mean",			data = train_mean)
	ECG_Sig.create_dataset("train.sd",				data = train_sd)
	
	ECG_Sig.create_dataset("val.signal.raw",		data = ECG_val_sig)
	ECG_Sig.create_dataset("val.label.raw",			data = ECG_val_label)
	ECG_Sig.close()
	
	print("Time Taken:",np.round((time.time()-t0)/60,2),"Minutes")



'''
This function generates patches that are loarge in the horizontal direction than i nthe vertical direction.

The size of the patch is 20x50.

The reason this is done is so that the patches are take more information in the time direction than in the frequnecy
direction since the ECG signals are fairly consistent
'''
def Generate_Spectrogram_Patch_Horizontal():

	t0 = time.time()
	
	ECG_class = dict()
	ECG_class['N'] = [1,0,0,0]
	ECG_class['A'] = [0,1,0,0]
	ECG_class['O'] = [0,0,1,0]
	ECG_class['~'] = [0,0,0,1]
	
	### Reading Data
	ECG_signal = []
	ECG_labels = []
	
	info_doc_train = np.array(pd.read_csv(os.getcwd()+"/training2017/REFERENCE.csv",header=None))
	for n in range(info_doc_train.shape[0]):
		file_in = os.getcwd()+"/training2017/"+info_doc_train[n,0]+".mat"
		ECG_signal.append(scipy.io.loadmat(file_in)["val"][0])
		ECG_labels.append(ECG_class[info_doc_train[n,1]])
	
	### Splitting into train/val sets
	ECG_train_sig,ECG_val_sig,ECG_train_label,ECG_val_label = train_test_split(ECG_signal,ECG_labels,test_size=0.2,random_state=32)
	
	### Generating Training Spectrograms
	spec_patches = []
	spec_labels = []
	
	for n in range(len(ECG_train_sig)):

		data = ECG_train_sig[n][500:]
		Pxx,_,_,_ = plt.specgram(data,NFFT=64,Fs=300,pad_to=256,noverlap=0)
		
		if Pxx.shape[1]<132:
			Pxx = np.tile(Pxx,int(np.ceil(132/Pxx.shape[1])))[:,0:132]
				
		for i in range(0,Pxx.shape[0]-20,20):
			for j in range(0,Pxx.shape[1]-131,132):
			
				spec_patches.append(Pxx[i:(i+20),j:(j+132)])
				spec_labels.append(ECG_train_label[n])

		print(n)
		
	spec_patches = np.array(spec_patches)
	spec_labels = np.array(spec_labels)
	
	spec_patches = np.reshape(spec_patches,newshape=[spec_patches.shape[0],spec_patches.shape[1],spec_patches.shape[2],1])
	spec_labels = np.reshape(spec_labels,newshape=[spec_labels.shape[0],4])

	### Validation Data Generation
	max_length = np.max([len(ECG_val_sig[i]) for i in range(len(ECG_val_sig))])
	ECG_val_sig = np.array([np.pad(ECG_val_sig[i],(0,max_length-len(ECG_val_sig[i])),"constant") for i in range(len(ECG_val_sig))])
	
	### Normalising
	train_mean = np.mean(spec_patches)
	train_sd = np.sqrt(np.var(spec_patches))
	spec_patches = (spec_patches-train_mean)/train_sd
	
	### Writing to HDF5 file
	ECG_Sig = h5py.File('ECG Spectrogram Patch Horizontal.h5','w')
	ECG_Sig.create_dataset("train.signal.spctrgm",	data = spec_patches)
	ECG_Sig.create_dataset("train.label.spctrgm",	data = spec_labels)
	ECG_Sig.create_dataset("train.mean",			data = train_mean)
	ECG_Sig.create_dataset("train.sd",				data = train_sd)
	
	ECG_Sig.create_dataset("val.signal.raw",		data = ECG_val_sig)
	ECG_Sig.create_dataset("val.label.raw",			data = ECG_val_label)
	ECG_Sig.close()
	
	print("Time Taken:",np.round((time.time()-t0)/60,2),"Minutes")



'''
This function takes in the input of all the data and generates a spectrogram.
The first 500 data points of the raw signal is removed as it usually has noise (i think)
The resulting resolution of the spectrogram is 129*132 (low resolution)
The signals which produce spectrograms that have less length are repeated until they match 132
'''
def Generate_Spectrogram():

	t0 = time.time()
	
	ECG_class = dict()
	ECG_class['N'] = [1,0,0,0]
	ECG_class['A'] = [0,1,0,0]
	ECG_class['O'] = [0,0,1,0]
	ECG_class['~'] = [0,0,0,1]
	
	### Reading Data
	ECG_signal = []
	ECG_labels = []
	
	info_doc_train = np.array(pd.read_csv(os.getcwd()+"/training2017/REFERENCE.csv",header=None))
	for n in range(info_doc_train.shape[0]):
		file_in = os.getcwd()+"/training2017/"+info_doc_train[n,0]+".mat"
		ECG_signal.append( scipy.io.loadmat(file_in)["val"][0] )
		ECG_labels.append(ECG_class[info_doc_train[n,1]])
		
	ECG_signal = np.array(ECG_signal)
	ECG_labels = np.array(ECG_labels)
	
	rows = 129 # number of frequencies
	cols = 132 # number of time intervals		--- 277 (max length spectro)		--- 156 (high res/small seg)
	
	spectrogram = np.empty([len(ECG_signal),rows,cols])
	
	### Generating spectrograms
	for i in range(len(ECG_signal)):
		print(i)
		data = ECG_signal[i][500:]												# --- :3000](high res/small seg)

		Pxx,_, _,_ = plt.specgram(data,NFFT=64,Fs=300,pad_to=256,noverlap=0)  # --- NFFT=15 (high res/small seg)
		spectrogram[i,:,:] = np.tile(Pxx,int(np.ceil(cols/Pxx.shape[1])))[:,0:cols]
	
	spectrogram = np.reshape(spectrogram,newshape=[len(ECG_signal),rows,cols,1])
	
	### Splitting into train/val sets
	index = random.sample([i for i in range(spectrogram.shape[0])],int(spectrogram.shape[0]*0.8))
	ECG_train_image = np.array([spectrogram[i,:,:,:] for i in index])
	ECG_train_label = np.array([ECG_labels[i,:] for i in index])
	
	index2 = np.delete([i for i in range(spectrogram.shape[0])],index)
	ECG_val_image = np.array([spectrogram[i,:,:,:] for i in index2])
	ECG_val_label = np.array([ECG_labels[i,:] for i in index2])
	
	### Normalising
	train_mean = np.mean(ECG_train_image)
	train_sd = np.sqrt(np.var(ECG_train_image))
	
	ECG_train_image = (ECG_train_image-train_mean)/train_sd
	ECG_val_image = (ECG_val_image-train_mean)/train_sd
	
	### Writing to HDF5 file
	ECG_Sig = h5py.File('ECG Whole Spectrogram 129x132.h5','w')
	ECG_Sig.create_dataset("train.signal",	data = ECG_train_image)
	ECG_Sig.create_dataset("train.label",	data = ECG_train_label)
	ECG_Sig.create_dataset("val.signal",	data = ECG_val_image)
	ECG_Sig.create_dataset("val.label",		data = ECG_val_label)
	ECG_Sig.create_dataset("train.mean",	data = train_mean)
	ECG_Sig.create_dataset("train.sd",		data = train_sd)
	ECG_Sig.close()
	
	print("Time Taken:",np.round((time.time()-t0)/60,2),"Minutes")



'''
This function reads in all the data of the ECG classification problem but puts them in segmentas of 1500 lengths
'''
def Generate_HDF5_Training_Data_1500_Lengths():
	
	t0 = time.time()
	
	ECG_class = dict()
	ECG_class['N'] = [1,0,0,0] # normal rhythm
	ECG_class['A'] = [0,1,0,0] # atrial fibrillation
	ECG_class['O'] = [0,0,1,0] # other rhythm
	ECG_class['~'] = [0,0,0,1] # noisy recording
	
	### Reading Data into segments of signal and padding signals with 0s
	ECG_signal = []
	ECG_labels = []
	
	info_doc_train = np.array(pd.read_csv(os.getcwd()+"/training2017/REFERENCE.csv",header=None))
	for n in range(info_doc_train.shape[0]):
		
		file_in = os.getcwd()+"/training2017/"+info_doc_train[n,0]+".mat"
		signal =  scipy.io.loadmat(file_in)["val"][0] 
		labels = ECG_class[info_doc_train[n,1]]
		
		n_seg = int(np.floor(len(signal)/1500))
		for count in range(0,(n_seg)*1500,1500):
		
			ECG_signal.append(signal[count:(count+1500)])
			ECG_labels.append(labels)
		
		if len(signal)/1500 > n_seg:
			temp = np.zeros(shape=[1500])
			temp[0:(len(signal)%1500)] = signal[(count+1500):]
			ECG_signal.append(temp)
			ECG_labels.append(labels)

	ECG_signal = np.array(ECG_signal)
	ECG_labels = np.array(ECG_labels)
		
	ECG_signal+=np.abs(np.min(ECG_signal)) # shifting the values so that they are all positive
	
	### Splitting into training and testing sets
	index = random.sample([i for i in range(len(ECG_signal))],int(len(ECG_signal)*0.80))
	ECG_train_data = ECG_signal[index,:]
	ECG_train_label = ECG_labels[index,:]

	index = np.delete([i for i in range(len(ECG_signal))],index)
	ECG_test_data = ECG_signal[index,:]
	ECG_test_label = ECG_labels[index,:]
	
	### Normalizing & Unit Variance
	#TrainMean = np.sum(ECG_train_data)/np.sum(lengths)
	#TrainSd = np.std(ECG_train_data)
	
	#ECG_train_data = (ECG_train_data-TrainMean)/TrainSd
	#ECG_test_data = (ECG_test_data-TrainMean)/TrainSd
	
	### Saving data to hdf5 dataset
	ECG_Sig = h5py.File('ECG Dataset.h5','w')
	ECG_Sig.create_dataset("train.data",	data = ECG_train_data)
	ECG_Sig.create_dataset("train.label",	data = ECG_train_label)
	ECG_Sig.create_dataset("val.data",		data = ECG_test_data)
	ECG_Sig.create_dataset("val.label",		data = ECG_test_label)
	ECG_Sig.close()
	
	print("Data Loaded!	Time Taken:",np.round((time.time()-t0),2),"seconds")



'''
This function reads in all the data from the ECG signal classification challenge. 
The format is the one that is directly downloaded from the website

There are 2 folders: "training2017" and "sample2017". "sample2017" contains "validation"

TRAINING DATA - training2017 containins the data (.mat) and the labels (training2017\\REFERENCE.csv)
VALIDATION DATA - sample2017\\validation containins the data (.mat) and the labels (sample2017\\validation\\REFERENCE.csv)

output: a dictionary containing test/val data and labels
'''
def Generate_HDF5_Training_Data_v1():
	
	t0 = time.time()
	
	ECG_class = dict()
	ECG_class['N'] = [1,0,0,0] # normal rhythm
	ECG_class['A'] = [0,1,0,0] # atrial fibrillation
	ECG_class['O'] = [0,0,1,0] # other rhythm
	ECG_class['~'] = [0,0,0,1] # noisy recording
	
	
	
	### Reading Data
	ECG_signal = []
	ECG_labels = []
	
	info_doc_train = np.array(pd.read_csv(os.getcwd()+"/training2017/REFERENCE.csv",header=None))
	for n in range(info_doc_train.shape[0]):
		file_in = os.getcwd()+"/training2017/"+info_doc_train[n,0]+".mat"
		ECG_signal.append( scipy.io.loadmat(file_in)["val"][0] )
		ECG_labels.append(ECG_class[info_doc_train[n,1]])
		
	ECG_signal = np.array(ECG_signal)
	ECG_labels = np.array(ECG_labels)
	
	
	
	### creating a 2D matrix of signals and padding the empty spaces to accomidate the varying array length
	lengths = [len(ECG_signal[i]) for i in range(len(ECG_signal))]
	
	ECG_signal_pad = np.zeros(shape=[len(ECG_signal),max(lengths)])
	for n in range(len(ECG_signal)):
		ECG_signal_pad[n,0:len(ECG_signal[n])] = ECG_signal[n]
	
	ECG_signal_pad+=np.abs(np.min(ECG_signal_pad))
	
	index = random.sample([i for i in range(len(ECG_signal_pad))],int(len(ECG_signal_pad)*0.80))
	ECG_train_data = ECG_signal_pad[index,:]
	ECG_train_label = ECG_labels[index,:]

	index = np.delete([i for i in range(len(ECG_signal))],index)
	ECG_test_data = ECG_signal_pad[index,:]
	ECG_test_label = ECG_labels[index,:]
	
	
	### Normalizing & Unit Variance
	TrainMean = np.sum(ECG_train_data)/np.sum(lengths)
	TrainSd = np.std(ECG_train_data)
	
	#ECG_train_data = (ECG_train_data-TrainMean)/TrainSd
	#ECG_test_data = (ECG_test_data-TrainMean)/TrainSd
	
	
	
	### Saving data to hdf5 dataset
	ECG_Sig = h5py.File('ECG Dataset.h5','w')
	ECG_Sig.create_dataset("train.data",	data = ECG_train_data)
	ECG_Sig.create_dataset("train.label",	data = ECG_train_label)
	ECG_Sig.create_dataset("test.data",		data = ECG_test_data)
	ECG_Sig.create_dataset("test.label",	data = ECG_test_label)
	ECG_Sig.create_dataset("train.mean",	data = TrainMean)
	ECG_Sig.create_dataset("train.sd",		data = TrainSd)
	ECG_Sig.close()
	
	print("Data Loaded!	Time Taken:",np.round((time.time()-t0),2),"seconds")



'''
This function reads the validation data set from the directory "sample2017/validation/"
This path will however MOST LIKEY be changed in the future so please keep that in mind

The function reads the data and outputs both the ECG signal and the label
'''
def Read_Validation_Set():
        
 	### Reading Data
	ECG_signal = []
	ECG_labels = []
	
	info_doc_test =  np.array(pd.read_csv(os.getcwd()+"/sample2017/validation/REFERENCE.csv",header=None))
	for n in range(info_doc_test.shape[0]):
		
		file_in = os.getcwd()+"/training2017/"+info_doc_train[n,0]+".mat"
		signal =  scipy.io.loadmat(file_in)["val"][0] 
		labels = ECG_class[info_doc_train[n,1]]

		ECG_signal.append(signal)
		ECG_labels.append(labels)

	ECG_signal = np.array(ECG_signal)
	ECG_labels = np.array(ECG_labels)
		
	ECG_signal+=np.abs(np.min(ECG_signal)) # shifting the values so that they are all positive
	
	return(ECG_signal,ECG_labels)



'''
This function augments the data via the method in:
"Improved musical onset detection with convolutional neural networks"
to produce 3 layer spectrogram inputs (with different freq resolutions but compressed into the same dimension)
'''
def Data_Augmentation_1_Stacking_Multi_Frequency_Resolution_Spectrogram():
	
	t0 = time.time()
	
	ECG_class = dict()
	ECG_class['N'] = [1,0,0,0]
	ECG_class['A'] = [0,1,0,0]
	ECG_class['O'] = [0,0,1,0]
	ECG_class['~'] = [0,0,0,1]
	
	### Reading Data
	ECG_signal = [] 
	ECG_labels = []
	
	info_doc_train = np.array(pd.read_csv(os.getcwd()+"/training2017/REFERENCE.csv",header=None))
	for n in range(info_doc_train.shape[0]):
		file_in = os.getcwd()+"/training2017/"+info_doc_train[n,0]+".mat"
		ECG_signal.append(scipy.io.loadmat(file_in)["val"][0])
		ECG_labels.append(ECG_class[info_doc_train[n,1]])

	### Splitting into train/val sets
	ECG_train_sig,ECG_val_sig,ECG_train_label,ECG_val_label = train_test_split(ECG_signal,ECG_labels,test_size=0.2,random_state=32)	

	### Generating Training Spectrograms
	spec_patches = []
	spec_labels = []
	
	patch_width = 200
	
	for n in range(len(ECG_train_sig)):

		data = ECG_train_sig[n][500:]

		Pxx,_,_,_ = plt.specgram(data,NFFT=4,Fs=300,pad_to=39,noverlap=0)		# 20  x ___
		#Pxx,_,_,_ = plt.specgram(data,NFFT=4,Fs=300,pad_to=79,noverlap=0)		# 40  x ___
		#Pxx,_,_,_ = plt.specgram(data,NFFT=4,Fs=300,pad_to=159,noverlap=0)		# 80  x ___
		
		for i in range(0,Pxx.shape[1]-patch_width,patch_width):
			
			spec_patches.append(Pxx[:,i:(i+patch_width)])
			spec_labels.append(ECG_train_label[n])
		
		print(n)
	
	spec_patches = np.array(spec_patches)
	#spec_patches = block_reduce(spec_patches,(1,4,1))
	
	spec_labels = np.array(spec_labels)
	
	spec_patches = np.reshape(spec_patches,newshape=[spec_patches.shape[0],spec_patches.shape[1],spec_patches.shape[2],1])
	spec_labels = np.reshape(spec_labels,newshape=[spec_labels.shape[0],4])

	### Validation Data Generation
	max_length = np.max([len(ECG_val_sig[i]) for i in range(len(ECG_val_sig))])
	ECG_val_sig = np.array([np.pad(ECG_val_sig[i],(0,max_length-len(ECG_val_sig[i])),"constant") for i in range(len(ECG_val_sig))])
	
	### Normalising
	train_mean = np.mean(spec_patches)
	train_sd = np.sqrt(np.var(spec_patches))
	spec_patches = (spec_patches-train_mean)/train_sd
	
	### Writing to HDF5 file
	ECG_Sig = h5py.File('ECG Spectrogram Patch Stacked.h5','w')
	ECG_Sig.create_dataset("train.signal.spctrgm",	data = spec_patches)
	ECG_Sig.create_dataset("train.label.spctrgm",	data = spec_labels)
	ECG_Sig.create_dataset("train.mean",			data = train_mean)
	ECG_Sig.create_dataset("train.sd",				data = train_sd)
	
	ECG_Sig.create_dataset("val.signal.raw",		data = ECG_val_sig)
	ECG_Sig.create_dataset("val.label.raw",			data = ECG_val_label)
	ECG_Sig.close()
	
	print("Time Taken:",np.round((time.time()-t0)/60,2),"Minutes")
