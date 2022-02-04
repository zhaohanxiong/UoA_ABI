# 2534 (16082 segments) N & 383 (2444 segments) AF signals in training set
# 2542 N & 375 AF signals in testing set

import tensorflow as tf
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from biosppy.signals import ecg
from sklearn.model_selection import train_test_split



### Function to Simulate Data
def ECG_Style_Transfer_Generation(content_sig,style_sig,N_FFT=64,W_LEN=64,mean=0,sd=1):
	
	# normalize data between 0-255
	content_sig = content_sig - np.min(content_sig)
	content_sig = content_sig/np.max(content_sig)*255

	style_sig = style_sig - np.min(style_sig)
	style_sig = style_sig/np.max(style_sig)*255

	# convert to spectrogram
	a_content = np.log1p(np.abs(librosa.stft(content_sig,N_FFT,win_length=W_LEN)))
	a_style   = np.log1p(np.abs(librosa.stft(style_sig,N_FFT,win_length=W_LEN)))

	N_CHANNELS,N_SAMPLES = a_content.shape[0],a_content.shape[1]

	a_style= a_style[:N_CHANNELS,:N_SAMPLES]
	
	# Compute content and style feats
	def conv_2d(incoming,kernel):

		kernel_tf = tf.constant(kernel, dtype='float32')
		conv = tf.nn.conv2d(incoming, kernel_tf, strides=[1,1,1,1], padding="SAME")
		net = tf.nn.relu(conv)
		
		return(net)

	N_FILTERS = 1024

	a_content_tf = np.ascontiguousarray(a_content.T[None,None,:,:])
	a_style_tf   = np.ascontiguousarray(a_style.T[None,None,:,:])

	# filter shape is [filter_height, filter_width, in_channels, out_channels]
	kernel1 = np.random.randn(1, 16, N_CHANNELS, N_FILTERS) * np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 16))

	g = tf.Graph()
	with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:

		# data shape is [batch, in_height, in_width, in_channels]
		x = tf.placeholder('float32', [1,1,N_SAMPLES,N_CHANNELS], name="x")
		
		net1 = conv_2d(x,kernel1)

		content_features = net1.eval(feed_dict={x: a_content_tf})
		
		features1   = np.reshape(net1.eval(feed_dict={x: a_style_tf}), (-1,N_FILTERS))
		style_gram1 = np.matmul(features1.T, features1) / N_SAMPLES

	ALPHA = 0.2 # larger weights content more, smaller weights style more

	with tf.Graph().as_default():

		# Build graph with variable input
		x = tf.Variable(np.random.randn(1,1,N_SAMPLES,N_CHANNELS).astype(np.float32)*1e-3, name="x")

		net1 = conv_2d(x,kernel1)

		content_loss = ALPHA * 2 * tf.nn.l2_loss(net1 - content_features)

		feats1 = tf.reshape(net1, (-1, int(net1.get_shape()[3])))
		gram1  = tf.matmul(tf.transpose(feats1), feats1) / N_SAMPLES

		style_loss = 2 * tf.nn.l2_loss(gram1 - style_gram1)

		# Overall loss
		loss = content_loss + style_loss
		opt = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options={'maxiter': 100})

		# Optimization
		with tf.Session() as sess:
		
			sess.run(tf.global_variables_initializer())
			opt.minimize(sess)
			result = x.eval()

	# Invert spectrogram and save the result
	a = np.zeros_like(a_content)
	a[:N_CHANNELS,:] = np.exp(result[0,0].T) - 1

	# phase reconstruction
	p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
	for i in range(100):
		S = a * np.exp(1j*p)
		output = librosa.istft(S)
		p = np.angle(librosa.stft(output, N_FFT))

	# Post process ecg, flip QRS peaks, and filter signal
	r_peak_pos = ecg.ecg(signal=content_sig, sampling_rate=300, show=False)[2]

	for r in r_peak_pos:
		if output[r] < output[r-10] and output[r] < output[r+10]:
			base = np.mean([output[r+10],output[r-10]])
			output[(r-15):(r+15)] += 3*(base - output[(r-15):(r+15)])

	filtered_output = ecg.ecg(signal=output, sampling_rate=300, show=False)[1]

	# normalize according to training set mean and standard deviation
	filtered_output = (filtered_output - mean)/sd
	
	return(filtered_output)



### Load Data
ECG = scipy.io.loadmat("C:/Users/Administrator/Desktop/cinc2017/ECG.mat")
train_dat,train_lab = ECG["train_dat"],ECG["train_lab"]
train_mean,train_sd = ECG['train_mean'][0][0],ECG['train_sd'][0][0]

Template = scipy.io.loadmat("ecg_content+style_samples/ECGtemplateA.mat")['dat']
Para_Ref = scipy.io.loadmat("ecg_content+style_samples/ParameterRefA.mat")['ref']



### Geneate Data
n_gen = 1 # number of samples to generate for each real sample

len_gen = np.unique(np.argmax(ECG["train_lab"],1),return_counts=True)[1][1]*n_gen
out_arr,count = np.zeros([int(len_gen/2),1500]),0

for i in range(len(ECG["train_lab"])):
	if np.argmax(ECG["train_lab"][i],0) == 1: # if AF
		if np.random.choice([True,False]): # adjust probability of augmentation
		
			print("Processing: ",count+1," of ",len_gen)
			
			style_sig = ECG["train_dat"][i,:,0]
			
			# use heart rate to sample content signal
			hr = round(np.mean(ecg.ecg(signal=style_sig,sampling_rate=300,show=False)[6]),-1) # round hr to nearest 10
			if np.isnan(hr):
				r_peak_pos = ecg.ecg(signal=style_sig,sampling_rate=300,show=False)[2]
				if len(r_peak_pos) > 1:
					hr = 80/(np.mean(np.diff(r_peak_pos))/300)
				else:
					hr = 80
			if hr > 120:
				hr = 120
				row_ind = np.random.choice(np.where(np.logical_and(Para_Ref[:,0]==hr,Para_Ref[:,1]==5))[0],n_gen,replace=False)
			elif hr < 60:
				hr = 60
				row_ind = np.random.choice(np.where(np.logical_and(Para_Ref[:,0]==hr,Para_Ref[:,1]==5))[0],n_gen,replace=False)
			else:
				row_ind = np.random.choice(np.where(np.logical_and(Para_Ref[:,0]==hr,Para_Ref[:,1]<20))[0],n_gen,replace=False)
			
			# generate samples with different templates
			for n in row_ind:
				content_sig = Template[n,500:2000]
				out = ECG_Style_Transfer_Generation(content_sig,style_sig,N_FFT=64,W_LEN=64,mean=train_mean,sd=train_sd)
				out_arr[count,:] = np.pad(out,6,mode="linear_ramp")
				count += 1 

#'''
# add normal too
Template = scipy.io.loadmat("ecg_content+style_samples/ECGtemplateN.mat")['dat']
Para_Ref = scipy.io.loadmat("ecg_content+style_samples/ParameterRefN.mat")['ref']

len_gen = np.unique(np.argmax(ECG["train_lab"],1),return_counts=True)[1][0]
out_arr2,count = np.zeros([int(len_gen/2),1500]),0

for i in range(len(ECG["train_lab"])):
	if np.argmax(ECG["train_lab"][i],0) == 0: # if N
		if np.random.choice([True,False,False,False]): # adjust probability of augmentation
		
			print("Processing: ",count+1," of ",len_gen)
			
			style_sig = ECG["train_dat"][i,:,0]
			
			# use heart rate to sample content signal
			hr = round(np.mean(ecg.ecg(signal=style_sig,sampling_rate=300,show=False)[6]),-1)
			hr += np.random.choice([5,-5,0])			
			
			if np.isnan(hr):
				r_peak_pos = ecg.ecg(signal=style_sig,sampling_rate=300,show=False)[2]
				if len(r_peak_pos) > 1:
					hr = 80/(np.mean(np.diff(r_peak_pos))/300)
				else:
					hr = 80
					
			if hr < 50:
				hr = 50
				
			if hr > 150:
				hr = 150
			
			row_ind = np.where(Para_Ref[:,0]==hr)[0][0]
			
			# generate samples with different templates
			content_sig = Template[row_ind,500:2000]
			out = ECG_Style_Transfer_Generation(content_sig,style_sig,N_FFT=64,W_LEN=64,mean=train_mean,sd=train_sd)
			out_arr2[count,:] = np.pad(out,6,mode="linear_ramp")
			count += 1
#'''

# add new data to dataset and shuffle
train_dat = np.concatenate((train_dat,out_arr[:,:,None]))
train_dat = np.concatenate((train_dat,out_arr2[:,:,None]))

train_lab = np.concatenate((train_lab,np.repeat([[0],[1]],out_arr.shape[0],axis=1).T))
train_lab = np.concatenate((train_lab,np.repeat([[1],[0]],out_arr2.shape[0],axis=1).T))

train_dat,_,train_lab,_ = train_test_split(train_dat,train_lab,test_size=0,random_state=9999)



### Save output
scipy.io.savemat("C:/Users/Administrator/Desktop/cinc2017/SimulatedECG.mat",
							mdict={"train_dat":		train_dat,
								   "train_lab":		train_lab,
								   "train_mean":	train_mean,
								   "train_sd":		train_sd,
								   "test_dat":		ECG["test_dat"][0],
								   "test_lab":		ECG["test_lab"][0]})