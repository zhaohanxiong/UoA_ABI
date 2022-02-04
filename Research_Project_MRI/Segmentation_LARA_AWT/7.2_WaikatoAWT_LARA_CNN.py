import tflearn
import tensorflow as tf

import os
import sys
import cv2
import h5py
import random
import scipy.io
import numpy as np
import matplotlib.pyplot as plt	

n1 = 272 # x
n2 = 272 # y

fm 			= 16	# feature map scale
kkk 		= 5		# kernel size
keep_rate 	= 0.5   # dropout

### Evaluation Function ------------------------------------------------------------------------------------------------------------------
def evaluate(data,CNN_model,log_path,mu=0,sd=1):

	# set up 
	test_input,test_output = data["label"],data["awt"]
	
	# initialize output log file
	mse_masked_score,mse_scores,dice_score = [],[],[]
	
	# loop through all test patients
	for i in range(test_input.shape[0]):
		
		# compile each MRI image into a stack by their centroids
		pred = np.zeros([576,576,176])
		
		for j in range(test_input.shape[3]):
			
			# find the center of mass of the mask
			midpoint = data["centroid"][i,j,:]
			
			# extract the patches from the midpoint
			if not np.any(np.isnan(midpoint)):
				
				n11,n12 = int(midpoint[0] - n1//2),int(midpoint[0] + n1//2)
				n21,n22 = int(midpoint[1] - n2//2),int(midpoint[1] + n2//2)

				# make prediction for slices with midpoints
				data_i = [(test_input[i,n11:n12,n21:n22,j][:,:,None]-mu)/sd]
				data_o = CNN_model.predict(data_i)

				pred[n11:n12,n21:n22,j] = data_o[0,:,:,0]

		# mask out the non atrial wall pixels in the prediction
		pred[test_output[i] == 0] = 0
		
		# Evaluation
		mse        = np.mean((test_output[i] - pred)**2)
		mse_masked = np.mean((test_output[i][test_output[i] > 0] - pred[test_output[i] > 0])**2)
		
		t,p  = test_output[i] > 0,pred > 0
		dice = 2 * np.sum(p * t) / (np.sum(p) + np.sum(t))
		
		# store scores
		mse_masked_score.append(mse_masked)
		mse_scores.append(mse)
		dice_score.append(dice)

	# overall score
	f = open(log_path,"a")
	f.write("\nOVERALL MSE MASKED AVEARGE = "+str(np.round(np.mean(np.array(mse_masked_score)),3))+"\n")
	f.write("\nOVERALL MSE AVEARGE        = "+str(np.round(np.mean(np.array(mse_scores)),3))+"\n")
	f.write("\nOVERALL DSC AVEARGE        = "+str(np.round(np.mean(np.array(dice_score)),3))+"\n")
	f.write("\n\n")
	f.close()

	return(np.array(mse_masked_score))

def save_best(data,CNN_model,mu=0,sd=1):

	print("\nSaving Best Outputs...\n")

	# set up 
	test_input,test_output = data["label"],data["awt"]
	
	# loop through all test patients
	for i in range(test_input.shape[0]):
		
		# compile each MRI image into a stack by their centroids
		pred = np.zeros([576,576,176])
		
		for j in range(test_input.shape[3]):
			
			# find the center of mass of the mask
			midpoint = data["centroid"][i,j,:]
			
			# extract the patches from the midpoint
			if not np.any(np.isnan(midpoint)):
				
				n11,n12 = int(midpoint[0] - n1//2),int(midpoint[0] + n1//2)
				n21,n22 = int(midpoint[1] - n2//2),int(midpoint[1] + n2//2)

				# make prediction for slices with midpoints
				data_i = [(test_input[i,n11:n12,n21:n22,j][:,:,None]-mu)/sd]
				data_o = CNN_model.predict(data_i)

				pred[n11:n12,n21:n22,j] = data_o[0,:,:,0]
		
		# mask out the non atrial wall pixels in the prediction
		pred[test_output[i] == 0] = 0
		
		# save to output
		scipy.io.savemat("Prediction Sample/test"+"{0:03}".format(i)+".mat",mdict={"input_seg": test_input[i],
		                                                                           "true":      test_output[i],
																				   "pred":      pred})

### Computation Graph ------------------------------------------------------------------------------------------------------------------
# 3d convolution operation
def tflearn_conv_2d(net,nb_filter,kernel,stride,dropout=1.0,is_train=True):

	net = tflearn.layers.conv.conv_2d(net,nb_filter,kernel,stride,padding="same",activation="linear",bias=False,trainable=is_train)
	net = tflearn.layers.normalization.batch_normalization(net)
	net = tflearn.activations.prelu(net)
	
	net = tflearn.layers.core.dropout(net,keep_prob=dropout)
	
	return(net)

# 3d deconvolution operation
def tflearn_deconv_2d(net,nb_filter,kernel,stride,dropout=1.0,is_train=True):

	net = tflearn.layers.conv.conv_2d_transpose(net,nb_filter,kernel,
												[net.shape[1].value*stride,net.shape[2].value*stride,nb_filter],
												[1,stride,stride,1],padding="same",activation="linear",bias=False,trainable=is_train)
	net = tflearn.layers.normalization.batch_normalization(net)
	net = tflearn.activations.prelu(net)
	net = tflearn.layers.core.dropout(net,keep_prob=dropout)
	
	return(net)

# merging operation
def tflearn_merge_2d(layers,method):
	
	net = tflearn.layers.merge_ops.merge(layers,method,axis=3)
	
	return(net)

# level 0 input
layer_0a_input	= tflearn.layers.core.input_data(shape=[None,n1,n2,1])

# level 1 down
layer_1a_conv 	= tflearn_conv_2d(net=layer_0a_input,nb_filter=fm,kernel=kkk,stride=1,is_train=True)
layer_1a_stack	= tflearn_merge_2d([layer_0a_input]*fm,"concat")

layer_1a_add	= tflearn_merge_2d([layer_1a_conv,layer_1a_stack],"elemwise_sum")
layer_1a_down	= tflearn_conv_2d(net=layer_1a_add,nb_filter=fm*2,kernel=2,stride=2,is_train=True)

# level 2 down
layer_2a_conv 	= tflearn_conv_2d(net=layer_1a_down,nb_filter=fm*2,kernel=kkk,stride=1,is_train=True)
layer_2a_conv 	= tflearn_conv_2d(net=layer_2a_conv,nb_filter=fm*2,kernel=kkk,stride=1,is_train=True)

layer_2a_add	= tflearn_merge_2d([layer_1a_down,layer_2a_conv],"elemwise_sum")
layer_2a_down	= tflearn_conv_2d(net=layer_2a_add,nb_filter=fm*4,kernel=2,stride=2,is_train=True)

# level 3 down
layer_3a_conv 	= tflearn_conv_2d(net=layer_2a_down,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)
layer_3a_conv 	= tflearn_conv_2d(net=layer_3a_conv,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)
layer_3a_conv 	= tflearn_conv_2d(net=layer_3a_conv,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)

layer_3a_add	= tflearn_merge_2d([layer_2a_down,layer_3a_conv],"elemwise_sum")
layer_3a_down	= tflearn_conv_2d(net=layer_3a_add,nb_filter=fm*8,kernel=2,stride=2,dropout=keep_rate,is_train=True)

# level 4 down
layer_4a_conv 	= tflearn_conv_2d(net=layer_3a_down,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
layer_4a_conv 	= tflearn_conv_2d(net=layer_4a_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
layer_4a_conv 	= tflearn_conv_2d(net=layer_4a_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

layer_4a_add	= tflearn_merge_2d([layer_3a_down,layer_4a_conv],"elemwise_sum")
layer_4a_down	= tflearn_conv_2d(net=layer_4a_add,nb_filter=fm*16,kernel=2,stride=2,dropout=keep_rate,is_train=True)

# level 5
layer_5a_conv 	= tflearn_conv_2d(net=layer_4a_down,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
layer_5a_conv 	= tflearn_conv_2d(net=layer_5a_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
layer_5a_conv 	= tflearn_conv_2d(net=layer_5a_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

layer_5a_add	= tflearn_merge_2d([layer_4a_down,layer_5a_conv],"elemwise_sum")
layer_5a_up		= tflearn_deconv_2d(net=layer_5a_add,nb_filter=fm*8,kernel=2,stride=2,dropout=keep_rate,is_train=True)

# level 4 up
layer_4b_concat	= tflearn_merge_2d([layer_4a_add,layer_5a_up],"concat")
layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_concat,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

layer_4b_add	= tflearn_merge_2d([layer_4b_conv,layer_4b_concat],"elemwise_sum")
layer_4b_up		= tflearn_deconv_2d(net=layer_4b_add,nb_filter=fm*4,kernel=2,stride=2,dropout=keep_rate,is_train=True)

# level 3 up
layer_3b_concat	= tflearn_merge_2d([layer_3a_add,layer_4b_up],"concat")
layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_concat,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

layer_3b_add	= tflearn_merge_2d([layer_3b_conv,layer_3b_concat],"elemwise_sum")
layer_3b_up		= tflearn_deconv_2d(net=layer_3b_add,nb_filter=fm*2,kernel=2,stride=2,is_train=True)

# level 2 up
layer_2b_concat	= tflearn_merge_2d([layer_2a_add,layer_3b_up],"concat")
layer_2b_conv 	= tflearn_conv_2d(net=layer_2b_concat,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)
layer_2b_conv 	= tflearn_conv_2d(net=layer_2b_conv,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)

layer_2b_add	= tflearn_merge_2d([layer_2b_conv,layer_2b_concat],"elemwise_sum")
layer_2b_up		= tflearn_deconv_2d(net=layer_2b_add,nb_filter=fm,kernel=2,stride=2,is_train=True)

# level 1 up
layer_1b_concat	= tflearn_merge_2d([layer_1a_add,layer_2b_up],"concat")
layer_1b_conv 	= tflearn_conv_2d(net=layer_1b_concat,nb_filter=fm*2,kernel=kkk,stride=1,is_train=True)

layer_1b_add	= tflearn_merge_2d([layer_1b_conv,layer_1b_concat],"elemwise_sum")

# level 0 classifier
layer_0b_conv	= tflearn.layers.conv.conv_2d(layer_1b_add,1,1,1,trainable=True)
layer_0b_clf	= tflearn.activations.relu(layer_0b_conv)

# loss function
def mse_loss(y_pred,y_true):

	with tf.name_scope("mse_function"):
		
		# compute distance loss
		masked_true = tf.boolean_mask(y_true, tf.greater(y_true, tf.zeros_like(y_true)))
		masked_pred = tf.boolean_mask(y_pred, tf.greater(y_true, tf.zeros_like(y_true)))
		
		mse = tf.reduce_mean(tf.pow(masked_true - masked_pred,2))

	return(mse)

# Optimizer
regress = tflearn.layers.estimator.regression(layer_0b_clf,optimizer='adam',loss=mse_loss,learning_rate=0.00001)
model   = tflearn.models.dnn.DNN(regress,tensorboard_dir='log')

### Training ------------------------------------------------------------------------------------------------------------------
# set main directory
os.chdir("WaikatoAWT Test Set")

# load pre-trained model
model.load("UtahAWTmodel/AWTmodel_Utah")

# set up log file
log_path = "log/log.txt"

# load training data (0 = background, 1 = RA wall, 2 = LA wall, 3 = RA endo, 4 = LA endo)
train_data,test_data = h5py.File("Training.h5","r"),h5py.File("Testing.h5","r")

# preprocess input and output data
train_input,train_output = train_data["label"],train_data["awt"]

# keep track of best dice score
f = open(log_path,"w");f.close()
best_mse = 1000

for n in range(1000):
	
	f = open(log_path,"a");f.write("-"*50+" Epoch "+str(n+1)+"\n");f.close()

	# run 1 epoch
	model.fit(train_input,train_output,n_epoch=1,show_metric=True,batch_size=16,shuffle=True)
	
	# evaluate current performance
	MSEs = evaluate(test_data,model,log_path) #,train_mean,train_sd)
	
	# if the model is currently the best
	if np.mean(MSEs) < best_mse:
	
		best_mse = np.mean(MSEs)
		save_best(test_data,model) #,train_mean,train_sd)
		model.save("log/AWTmodel_Waikato")