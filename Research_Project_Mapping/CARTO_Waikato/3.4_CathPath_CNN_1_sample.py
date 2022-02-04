import os
import cv2
import h5py
import tflearn
import numpy as np
import tensorflow as tf

n1 = 128 # x
n2 = 208 # y

fm 			= 8  	# feature map scale
kkk 		= 5		# kernel size
keep_rate 	= 0.75  # dropout

### Computation Graph ------------------------------------------------------------------------------------------------------------------
# 3d convolution operation
def tflearn_conv_3d(net,nb_filter,kernel,stride,dropout=1.0,is_train=True):

	net = tflearn.layers.conv.conv_3d(net,nb_filter,kernel,[1,stride,stride,1,1],
									  padding="same",activation="linear",bias=False,trainable=is_train)
	net = tflearn.layers.normalization.batch_normalization(net)
	net = tflearn.activations.prelu(net)
	net = tflearn.layers.core.dropout(net,keep_prob=dropout)
	
	return(net)

# 3d deconvolution operation
def tflearn_deconv_3d(net,nb_filter,kernel,stride,dropout=1.0,is_train=True):

	net = tflearn.layers.conv.conv_3d_transpose(net,nb_filter,kernel,
												[net.shape[1].value*stride,net.shape[2].value*stride,net.shape[3].value*1,nb_filter],
												[1,stride,stride,1,1],padding="same",activation="linear",bias=False,trainable=is_train)
	net = tflearn.layers.normalization.batch_normalization(net)
	net = tflearn.activations.prelu(net)
	net = tflearn.layers.core.dropout(net,keep_prob=dropout)
	
	return(net)

# merging operation
def tflearn_merge_3d(layers,method):
	
	net = tflearn.layers.merge_ops.merge(layers,method,axis=4)
	
	return(net)

# level 0 input
layer_0a_input	= tflearn.layers.core.input_data(shape=[None,n1,n2,44,1])

# level 1 down
layer_1a_conv 	= tflearn_conv_3d(net=layer_0a_input,nb_filter=fm,kernel=kkk,stride=1,is_train=True)
layer_1a_stack	= tflearn_merge_3d([layer_0a_input]*fm,"concat")

layer_1a_add	= tflearn_merge_3d([layer_1a_conv,layer_1a_stack],"elemwise_sum")
layer_1a_down	= tflearn_conv_3d(net=layer_1a_add,nb_filter=fm*2,kernel=2,stride=2,is_train=True)

# level 2 down
layer_2a_conv 	= tflearn_conv_3d(net=layer_1a_down,nb_filter=fm*2,kernel=kkk,stride=1,is_train=True)
layer_2a_conv 	= tflearn_conv_3d(net=layer_2a_conv,nb_filter=fm*2,kernel=kkk,stride=1,is_train=True)

layer_2a_add	= tflearn_merge_3d([layer_1a_down,layer_2a_conv],"elemwise_sum")
layer_2a_down	= tflearn_conv_3d(net=layer_2a_add,nb_filter=fm*4,kernel=2,stride=2,is_train=True)

# level 3 down
layer_3a_conv 	= tflearn_conv_3d(net=layer_2a_down,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)
layer_3a_conv 	= tflearn_conv_3d(net=layer_3a_conv,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)
layer_3a_conv 	= tflearn_conv_3d(net=layer_3a_conv,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)

layer_3a_add	= tflearn_merge_3d([layer_2a_down,layer_3a_conv],"elemwise_sum")
layer_3a_down	= tflearn_conv_3d(net=layer_3a_add,nb_filter=fm*8,kernel=2,stride=2,dropout=keep_rate,is_train=True)

# level 4 down
layer_4a_conv 	= tflearn_conv_3d(net=layer_3a_down,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
layer_4a_conv 	= tflearn_conv_3d(net=layer_4a_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
layer_4a_conv 	= tflearn_conv_3d(net=layer_4a_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

layer_4a_add	= tflearn_merge_3d([layer_3a_down,layer_4a_conv],"elemwise_sum")
layer_4a_down	= tflearn_conv_3d(net=layer_4a_add,nb_filter=fm*16,kernel=2,stride=2,dropout=keep_rate,is_train=True)

# level 5
layer_5a_conv 	= tflearn_conv_3d(net=layer_4a_down,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
layer_5a_conv 	= tflearn_conv_3d(net=layer_5a_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
layer_5a_conv 	= tflearn_conv_3d(net=layer_5a_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

layer_5a_add	= tflearn_merge_3d([layer_4a_down,layer_5a_conv],"elemwise_sum")
layer_5a_up		= tflearn_deconv_3d(net=layer_5a_add,nb_filter=fm*8,kernel=2,stride=2,dropout=keep_rate,is_train=True)

# level 4 up
layer_4b_concat	= tflearn_merge_3d([layer_4a_add,layer_5a_up],"concat")
layer_4b_conv 	= tflearn_conv_3d(net=layer_4b_concat,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
layer_4b_conv 	= tflearn_conv_3d(net=layer_4b_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
layer_4b_conv 	= tflearn_conv_3d(net=layer_4b_conv,nb_filter=fm*16,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

layer_4b_add	= tflearn_merge_3d([layer_4b_conv,layer_4b_concat],"elemwise_sum")
layer_4b_up		= tflearn_deconv_3d(net=layer_4b_add,nb_filter=fm*4,kernel=2,stride=2,dropout=keep_rate,is_train=True)

# level 3 up
layer_3b_concat	= tflearn_merge_3d([layer_3a_add,layer_4b_up],"concat")
layer_3b_conv 	= tflearn_conv_3d(net=layer_3b_concat,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
layer_3b_conv 	= tflearn_conv_3d(net=layer_3b_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)
layer_3b_conv 	= tflearn_conv_3d(net=layer_3b_conv,nb_filter=fm*8,kernel=kkk,stride=1,dropout=keep_rate,is_train=True)

layer_3b_add	= tflearn_merge_3d([layer_3b_conv,layer_3b_concat],"elemwise_sum")
layer_3b_up		= tflearn_deconv_3d(net=layer_3b_add,nb_filter=fm*2,kernel=2,stride=2,is_train=True)

# level 2 up
layer_2b_concat	= tflearn_merge_3d([layer_2a_add,layer_3b_up],"concat")
layer_2b_conv 	= tflearn_conv_3d(net=layer_2b_concat,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)
layer_2b_conv 	= tflearn_conv_3d(net=layer_2b_conv,nb_filter=fm*4,kernel=kkk,stride=1,is_train=True)

layer_2b_add	= tflearn_merge_3d([layer_2b_conv,layer_2b_concat],"elemwise_sum")
layer_2b_up		= tflearn_deconv_3d(net=layer_2b_add,nb_filter=fm,kernel=2,stride=2,is_train=True)

# level 1 up
layer_1b_concat	= tflearn_merge_3d([layer_1a_add,layer_2b_up],"concat")
layer_1b_conv 	= tflearn_conv_3d(net=layer_1b_concat,nb_filter=fm*2,kernel=kkk,stride=1,is_train=True)

layer_1b_add	= tflearn_merge_3d([layer_1b_conv,layer_1b_concat],"elemwise_sum")

# level 0 classifier
layer_0b_conv	= tflearn.layers.conv.conv_3d(layer_1b_add,2,1,1,trainable=True)
layer_0b_clf	= tflearn.activations.softmax(layer_0b_conv)

# loss function
def dice_loss_3d(y_pred,y_true):
	
	with tf.name_scope("dice_loss_3D_function"):
		
		# compute dice scores for each individually
		y_pred1,y_true1 = y_pred[:,:,:,:,1],y_true[:,:,:,:,1]
		intersection1   = tf.reduce_sum(y_pred1*y_true1)
		union1          = tf.reduce_sum(y_pred1*y_pred1) + tf.reduce_sum(y_true1*y_true1)
		dice1           = (2.0 * intersection1 + 1.0) / (union1 + 1.0)
		
	return(1.0 - dice1)

# Optimizer
regress = tflearn.layers.estimator.regression(layer_0b_clf,optimizer='adam',loss=dice_loss_3d,learning_rate=0.0001)
model   = tflearn.models.dnn.DNN(regress,tensorboard_dir='log')

### Pre Process ------------------------------------------------------------------------------------------------------------------
# set main directory
os.chdir("C:/Users/Administrator/Desktop/Catheter Path Test Set")

n = 0

# load data
waikato_data = h5py.File("waikato path.h5","r")

in_dat,lab = data["test.data"][n],data["test.label"][n]

### Predict ----------------------------------------------------------------------------------------------------------------------
model.load("log/CathPath_model")

pred = np.argmax(model.predict([in_dat])[0],3)
f1   = np.sum(pred[lab==1])*2.0 / (np.sum(pred) + np.sum(lab))

print("F1 Score: "+str(np.round(f1,4)*100))

# write to image stack
os.mkdir("Waikato Data/Image Stack/"+pat_file)
os.mkdir("Waikato Data/Image Stack/"+pat_file+"/true")
os.mkdir("Waikato Data/Image Stack/"+pat_file+"/pred")
os.mkdir("Waikato Data/Image Stack/"+pat_file+"/input")

lab,pred,in_dat = np.uint8(lab),np.uint8(pred),np.uint8(in_dat)

for i in range(lab.shape[2]):
	img_name = "{0:03}".format(i+1)+".tif"
	cv2.imwrite("Waikato Data/Image Stack/"+pat_file+"/true/"+img_name,lab[:,:,i])
	cv2.imwrite("Waikato Data/Image Stack/"+pat_file+"/pred/"+img_name,pred[:,:,i])
	cv2.imwrite("Waikato Data/Image Stack/"+pat_file+"/input/"+img_name,in_dat[:,:,i,0]*255)
