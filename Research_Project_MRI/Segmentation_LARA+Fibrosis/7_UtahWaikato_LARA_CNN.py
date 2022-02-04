import tflearn
import tensorflow as tf

import h5py
import numpy as np
import sys
import os
import scipy.io
import cv2
import random
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt	
from QingXia_Utils import multilabel_random_flip_2d,multilabel_random_scale_2d,multilabel_random_rotation_2d,multilabel_elastic_deformation_2d

n1 = 272 # x
n2 = 272 # y

fm 			= 16	# feature map scale
kkk 		= 5		# kernel size
keep_rate 	= 0.8   # dropout

### Evaluation Function ------------------------------------------------------------------------------------------------------------------
def evaluate(data,CNN_model,log_path,mu=0,sd=1):

	# initialize output log file
	f1_scores,score1,score2,score3 = [],[],[],[]
	
	# loop through all test patients
	for i in range(data["image"].shape[0]):
		
		# compile each MRI image into a stack by their centroids
		pred = np.zeros([576,576,44])
		
		for j in range(data["image"].shape[3]):
			
			# find the center of mass of the mask
			midpoint = data["centroid"][i,j,:]
			
			# extract the patches from the midpoint
			if not np.any(np.isnan(midpoint)):
				
				n11,n12 = int(midpoint[0] - n1//2),int(midpoint[0] + n1//2)
				n21,n22 = int(midpoint[1] - n2//2),int(midpoint[1] + n2//2)

				# make prediction for slices with midpoints
				data_i = [(data["image"][i,n11:n12,n21:n22,j][:,:,None]-mu)/sd]
				data_o = CNN_model.predict(data_i)

				pred[n11:n12,n21:n22,j] = np.argmax(data_o,3)[0]
	
		# Evaluation (0 = background, 1 = RA+LA wall, 2 = RA endo, 3 = LA endo)
		true_flat,pred_flat = data["label"][i].flatten(),pred.flatten()
		
		temp = f1_score(true_flat,pred_flat,average=None)[1:]

		# store scores
		f1_scores.append(np.mean([temp[1],temp[2]]))
		score1.append(temp[0])
		score2.append(temp[1])
		score3.append(temp[2])

	# overall score
	f = open(log_path,"a")
	f.write("\nOVERALL DSC AVEARGE = "+str(np.round(np.mean(np.array(f1_scores)),5))+"\n")
	f.write("\n      RA+LA EPI DSC = "+str(np.round(np.mean(np.array(score1)),5)))
	f.write("\n        RA ENDO DSC = "+str(np.round(np.mean(np.array(score2)),5)))
	f.write("\n        LA ENDO DSC = "+str(np.round(np.mean(np.array(score3)),5)))
	f.write("\n\n")
	f.close()

	return(np.array(f1_scores))

def save_best(data,CNN_model,mu=0,sd=1):
		
	print("\nSaving Best Outputs...\n")

	# loop through all test patients
	for i in range(data["image"].shape[0]):
		
		# compile each MRI image into a stack by their centroids
		pred = np.zeros([576,576,44])
		
		for j in range(data["image"].shape[3]):
			
			# find the center of mass of the mask
			midpoint = data["centroid"][i,j,:]
			
			# extract the patches from the midpoint
			if not np.any(np.isnan(midpoint)):
				
				n11,n12 = int(midpoint[0] - n1//2),int(midpoint[0] + n1//2)
				n21,n22 = int(midpoint[1] - n2//2),int(midpoint[1] + n2//2)

				# make prediction for slices with midpoints
				data_i = [(data["image"][i,n11:n12,n21:n22,j][:,:,None]-mu)/sd]
				data_o = CNN_model.predict(data_i)

				pred[n11:n12,n21:n22,j] = np.argmax(data_o,3)[0]

		# save to output
		scipy.io.savemat("Prediction Sample/test"+"{0:03}".format(i)+".mat",mdict={"true":data["label"][i],"pred":pred,"lgemri":data["image"][i]})
	
def online_augmentation(data,label):
	#train_image,train_label,train_mean,train_sd = online_augmentation(train_Image,train_Label)
	print("\nPerforming Online Data Augmentation...\n")
	
	# initialize arrays for storing the data
	data_aug = np.zeros([data.shape[0],data.shape[1],data.shape[2],data.shape[3]]) # N x X x Y x 1
	label_aug = np.zeros([label.shape[0],label.shape[1],label.shape[2],label.shape[3]]) # N x X x Y x 4
	
	# loop through all the data
	for i in range(data.shape[0]):
	
		temp_img,temp_lab = data[i,:,:,0],label[i,:,:,:]
		
		# probably of augmentation
		if random.uniform(0,1) >= 0:

			if random.choice([True,False]):
				temp_img,temp_lab = multilabel_random_scale_2d(temp_img,temp_lab,(0.2,0.2))
			if random.choice([True,False]):
				temp_img,temp_lab = multilabel_random_rotation_2d(temp_img,temp_lab,(20,20))
			if random.choice([True,False]):
				temp_img,temp_lab = multilabel_random_flip_2d(temp_img,temp_lab)
			if random.choice([True,False]):
				temp_img,temp_lab = multilabel_elastic_deformation_2d(temp_img,temp_lab,temp_img.shape[0]*3,temp_img.shape[0]*0.10)
			
			temp_lab[temp_lab<0.5]  = 0
			temp_lab[temp_lab>=0.5] = 1

		# append data
		data_aug[i,:,:,0],label_aug[i,:,:,:] = temp_img,temp_lab
		
	# normalize the data
	data_mean,data_sd = np.mean(data_aug),np.std(data_aug)
	data_aug = (data_aug - data_mean)/data_sd
	
	return(data_aug,label_aug,data_mean,data_sd)

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
layer_0b_conv	= tflearn.layers.conv.conv_2d(layer_1b_add,4,1,1,trainable=True)
layer_0b_clf	= tflearn.activations.softmax(layer_0b_conv)

# loss function
def dice_loss_2d(y_pred,y_true):
	
	with tf.name_scope("dice_loss_2D_function"):
		
		# compute dice scores for each individually
		y_pred1,y_true1 = y_pred[:,:,:,1],y_true[:,:,:,1]
		intersection1   = tf.reduce_sum(y_pred1*y_true1)
		union1          = tf.reduce_sum(y_pred1*y_pred1) + tf.reduce_sum(y_true1*y_true1)
		dice1           = (2.0 * intersection1 + 1.0) / (union1 + 1.0)
		
		y_pred2,y_true2 = y_pred[:,:,:,2],y_true[:,:,:,2]
		intersection2   = tf.reduce_sum(y_pred2*y_true2)
		union2          = tf.reduce_sum(y_pred2*y_pred2) + tf.reduce_sum(y_true2*y_true2)
		dice2           = (2.0 * intersection2 + 1.0) / (union2 + 1.0)
		
		y_pred3,y_true3 = y_pred[:,:,:,3],y_true[:,:,:,3]
		intersection3   = tf.reduce_sum(y_pred3*y_true3)
		union3          = tf.reduce_sum(y_pred3*y_pred3) + tf.reduce_sum(y_true3*y_true3)
		dice3           = (2.0 * intersection3 + 1.0) / (union3 + 1.0)
		
	return(1.0 - (dice1*0.333 + dice2*0.333 + dice3*0.333))

# Optimizer
regress = tflearn.layers.estimator.regression(layer_0b_clf,optimizer='adam',loss=dice_loss_2d,learning_rate=0.0001)
model   = tflearn.models.dnn.DNN(regress,tensorboard_dir='log')

### Training ------------------------------------------------------------------------------------------------------------------
# set main directory
os.chdir("UtahWaikato Test Set")

# load pre-trained model
model.load("pre-train/LARAmodel")

# set up log file
log_path = "log/log.txt"

# load training data and apply mean and standard deviation normalization
train_data,test_data = h5py.File("Training.h5","r"),h5py.File("Testing.h5","r")
train_mean,train_sd  = np.mean(train_data["image"]),np.std(train_data["image"])

# keep track of best dice score
best_DSC = 0
f = open(log_path,"w");f.write("mean: "+str(np.round(train_mean,5))+"\t std: "+str(np.round(train_sd,5)));f.close()

for n in range(1000):
	
	f = open(log_path,"a");f.write("\n"+"-"*50+" Epoch "+str(n+1));f.close()

	# online augmentation
	train_image,train_label,train_mean,train_sd = online_augmentation(train_data["image"],train_data["label"])
	
	# run 1 epoch
	model.fit(train_image,train_label,n_epoch=1,show_metric=True,batch_size=8,shuffle=True)
	
	# evaluate current performance
	DSCs = evaluate(test_data,model,log_path,train_mean,train_sd)
	
	# if the model is currently the best
	if np.mean(DSCs) > best_DSC:
		best_DSC = np.mean(DSCs)
		save_best(test_data,model,train_mean,train_sd)
		model.save("log/LARAmodel_UtahWaikato")
