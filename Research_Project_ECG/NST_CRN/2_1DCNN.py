from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.conv import conv_1d,max_pool_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.activations import relu
from tflearn.initializations import variance_scaling
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.models.dnn import DNN

import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
import os
import scipy.io
import matplotlib.pyplot as plt

input_sig_len,dropout_keep_prob = 1500,0.8

### Helper Function
def prediction(pred_data,ground_truth,CNN_model):
	
	pred = []
	
	for n in range(pred_data.shape[0]):
		pred_prob = np.array(CNN_model.predict(pred_data[n]))
		pred.append(np.argmax(np.multiply.reduce(pred_prob,0)))
		
	f1 = f1_score(np.array(pred),ground_truth,average=None)
	
	f = open("log/log.txt","a")
	f.write("\n  F1 Score: "+str(np.round(np.mean(f1),3))+"   F1: "+str(f1)+"\n\n")
	f.close()

	return(np.array(pred),np.mean(f1))

def visualize_activations(CNN_model,layer,stimuli):

	get_activation = DNN(layer,session=CNN_model.session)
	activation = get_activation.predict(stimuli)

	plt.imshow(activation[0,:,:],cmap='gnuplot2')
	plt.show()
	#visualize_activations(ECGmodel,interm,[ECG["train_dat"][0]])
	
### Computational Graph ###########################################################################################
def build_model(input_dim,keep_prob):

	# Input Block
	block_input_layer = input_data(shape=[None,input_dim,1])
	block_input = conv_1d(block_input_layer,64,16,activation="linear",bias=False,weights_init=variance_scaling())
	block_input = batch_normalization(block_input)
	block_input = relu(block_input)

	# ---------------------------------------------
	# Residual Block 1
	block_1 = conv_1d(block_input,64,16,activation="linear",bias=False,weights_init=variance_scaling())
	block_1 = batch_normalization(block_1)
	block_1 = relu(block_1)
	block_1 = dropout(block_1,keep_prob)
	block_1 = conv_1d(block_1,64,16,activation="linear",bias=False,weights_init=variance_scaling())

	skip_1 = max_pool_1d(block_input,1)
	block_1 = merge([skip_1,block_1],mode='elemwise_sum')

	# Residual Block 2
	block_2 = batch_normalization(block_1)
	block_2 = relu(block_2)
	block_2 = dropout(block_2,keep_prob)
	block_2 = conv_1d(block_2,64,16,activation="linear",bias=False,weights_init=variance_scaling())
	block_2 = batch_normalization(block_2)
	block_2 = relu(block_2)
	block_2 = dropout(block_2,keep_prob)
	block_2 = conv_1d(block_2,64,16,strides=2,activation="linear",bias=False,weights_init=variance_scaling())

	skip_2 = max_pool_1d(block_1,2)
	block_2 = merge([skip_2,block_2],mode='elemwise_sum')

	# Residual Block 3
	block_3 = batch_normalization(block_2)
	block_3 = relu(block_3)
	block_3 = dropout(block_3,keep_prob)
	block_3 = conv_1d(block_3,64,16,activation="linear",bias=False,weights_init=variance_scaling())
	block_3 = batch_normalization(block_3)
	block_3 = relu(block_3)
	block_3 = dropout(block_3,keep_prob)
	block_3 = conv_1d(block_3,64,16,activation="linear",bias=False,weights_init=variance_scaling())

	skip_3 = max_pool_1d(block_2,1)
	block_3 = merge([skip_3,block_3],mode='elemwise_sum')

	# Residual Block 4
	block_4 = batch_normalization(block_3)
	block_4 = relu(block_4)
	block_4 = dropout(block_4,keep_prob)
	block_4 = conv_1d(block_4,64,16,activation="linear",bias=False,weights_init=variance_scaling())
	block_4 = batch_normalization(block_4)
	block_4 = relu(block_4)
	block_4 = dropout(block_4,keep_prob)
	block_4 = conv_1d(block_4,64,16,strides=2,activation="linear",bias=False,weights_init=variance_scaling())

	skip_4 = max_pool_1d(block_3,2)
	block_4 = merge([skip_4,block_4],mode='elemwise_sum')

	# ---------------------------------------------
	# Residual Block 5
	block_5 = batch_normalization(block_4)
	block_5 = relu(block_5)
	block_5 = dropout(block_5,keep_prob)
	block_5 = conv_1d(block_5,128,16,activation="linear",bias=False,weights_init=variance_scaling())
	block_5 = batch_normalization(block_5)
	block_5 = relu(block_5)
	block_5 = dropout(block_5,keep_prob)
	block_5 = conv_1d(block_5,128,16,activation="linear",bias=False,weights_init=variance_scaling())

	# Residual Block 6
	block_6 = batch_normalization(block_5)
	block_6 = relu(block_6)
	block_6 = dropout(block_6,keep_prob)
	block_6 = conv_1d(block_6,128,16,activation="linear",bias=False,weights_init=variance_scaling())
	block_6 = batch_normalization(block_6)
	block_6 = relu(block_6)
	block_6 = dropout(block_6,keep_prob)
	block_6 = conv_1d(block_6,128,16,strides=2,activation="linear",bias=False,weights_init=variance_scaling())

	skip_6 = max_pool_1d(block_5,2)
	block_6 = merge([skip_6,block_6],mode='elemwise_sum')

	# Residual Block 7
	block_7 = batch_normalization(block_6)
	block_7 = relu(block_7)
	block_7 = dropout(block_7,keep_prob)
	block_7 = conv_1d(block_7,128,16,activation="linear",bias=False,weights_init=variance_scaling())
	block_7 = batch_normalization(block_7)
	block_7 = relu(block_7)
	block_7 = dropout(block_7,keep_prob)
	block_7 = conv_1d(block_7,128,16,activation="linear",bias=False,weights_init=variance_scaling())

	skip_7 = max_pool_1d(block_6,1)
	block_7 = merge([skip_7,block_7],mode='elemwise_sum')

	# Residual Block 8
	block_8 = batch_normalization(block_7)
	block_8 = relu(block_8)
	block_8 = dropout(block_8,keep_prob)
	block_8 = conv_1d(block_8,128,16,activation="linear",bias=False,weights_init=variance_scaling())
	block_8 = batch_normalization(block_8)
	block_8 = relu(block_8)
	block_8 = dropout(block_8,keep_prob)
	block_8 = conv_1d(block_8,128,16,strides=2,activation="linear",bias=False,weights_init=variance_scaling())

	skip_8 = max_pool_1d(block_7,2)
	block_8 = merge([skip_8,block_8],mode='elemwise_sum')

	# ---------------------------------------------
	# Residual Block 9
	block_9 = batch_normalization(block_8)
	block_9 = relu(block_9)
	block_9 = dropout(block_9,keep_prob)
	block_9 = conv_1d(block_9,192,16,activation="linear",bias=False,weights_init=variance_scaling())
	block_9 = batch_normalization(block_9)
	block_9 = relu(block_9)
	block_9 = dropout(block_9,keep_prob)
	block_9 = conv_1d(block_9,192,16,activation="linear",bias=False,weights_init=variance_scaling())

	# Residual Block 10
	block_10 = batch_normalization(block_9)
	block_10 = relu(block_10)
	block_10 = dropout(block_10,keep_prob)
	block_10 = conv_1d(block_10,192,16,activation="linear",bias=False,weights_init=variance_scaling())
	block_10 = batch_normalization(block_10)
	block_10 = relu(block_10)
	block_10 = dropout(block_10,keep_prob)
	block_10 = conv_1d(block_10,192,16,strides=2,activation="linear",bias=False,weights_init=variance_scaling())

	skip_10 = max_pool_1d(block_9,2)
	block_10 = merge([skip_10,block_10],mode='elemwise_sum')

	# Residual Block 11
	block_11 = batch_normalization(block_10)
	block_11 = relu(block_11)
	block_11 = dropout(block_11,keep_prob)
	block_11 = conv_1d(block_11,192,16,activation="linear",bias=False,weights_init=variance_scaling())
	block_11 = batch_normalization(block_11)
	block_11 = relu(block_11)
	block_11 = dropout(block_11,keep_prob)
	block_11 = conv_1d(block_11,192,16,activation="linear",bias=False,weights_init=variance_scaling())

	skip_11 = max_pool_1d(block_10,1)
	block_11 = merge([skip_11,block_11],mode='elemwise_sum')

	# Residual Block 12
	block_12 = batch_normalization(block_11)
	block_12 = relu(block_12)
	block_12 = dropout(block_12,keep_prob)
	block_12 = conv_1d(block_12,192,16,activation="linear",bias=False,weights_init=variance_scaling())
	block_12 = batch_normalization(block_12)
	block_12 = relu(block_12)
	block_12 = dropout(block_12,keep_prob)
	block_12 = conv_1d(block_12,192,16,strides=2,activation="linear",bias=False,weights_init=variance_scaling())

	skip_12 = max_pool_1d(block_11,2)
	block_12 = merge([skip_12,block_12],mode='elemwise_sum')

	# ---------------------------------------------
	# Residual Block 13
	block_13 = batch_normalization(block_12)
	block_13 = relu(block_13)
	block_13 = dropout(block_13,keep_prob)
	block_13 = conv_1d(block_13,256,16,activation="linear",bias=False,weights_init=variance_scaling())
	block_13 = batch_normalization(block_13)
	block_13 = relu(block_13)
	block_13 = dropout(block_13,keep_prob)
	block_13 = conv_1d(block_13,256,16,activation="linear",bias=False,weights_init=variance_scaling())

	# Residual Block 14
	block_14 = batch_normalization(block_13)
	block_14 = relu(block_14)
	block_14 = dropout(block_14,keep_prob)
	block_14 = conv_1d(block_14,256,16,activation="linear",bias=False,weights_init=variance_scaling())
	block_14 = batch_normalization(block_14)
	block_14 = relu(block_14)
	block_14 = dropout(block_14,keep_prob)
	block_14 = conv_1d(block_14,256,16,strides=2,activation="linear",bias=False,weights_init=variance_scaling())

	skip_14 = max_pool_1d(block_13,2)
	block_14 = merge([skip_14,block_14],mode='elemwise_sum')

	# Residual Block 15
	block_15 = batch_normalization(block_14)
	block_15 = relu(block_15)
	block_15 = dropout(block_15,keep_prob)
	block_15 = conv_1d(block_15,256,16,activation="linear",bias=False,weights_init=variance_scaling())
	block_15 = batch_normalization(block_15)
	block_15 = relu(block_15)
	block_15 = dropout(block_15,keep_prob)
	block_15 = conv_1d(block_15,256,16,activation="linear",bias=False,weights_init=variance_scaling())

	skip_15 = max_pool_1d(block_14,1)
	block_15 = merge([skip_15,block_15],mode='elemwise_sum')

	# Residual Block 16
	block_16 = batch_normalization(block_15)
	block_16 = relu(block_16)
	block_16 = dropout(block_16,keep_prob)
	block_16 = conv_1d(block_16,256,16,activation="linear",bias=False,weights_init=variance_scaling())
	block_16 = batch_normalization(block_16)
	block_16 = relu(block_16)
	block_16 = dropout(block_16,keep_prob)
	block_16 = conv_1d(block_16,256,16,strides=2,activation="linear",bias=False,weights_init=variance_scaling())

	skip_16 = max_pool_1d(block_15,2)
	block_16 = merge([skip_16,block_16],mode='elemwise_sum')

	# ---------------------------------------------
	# Output Block
	block_output_a = batch_normalization(block_16)
	block_output_b = relu(block_output_a)
	block_output_c = fully_connected(block_output_b,2,activation='softmax')

	trainer = regression(block_output_c,optimizer='adam',loss="categorical_crossentropy",learning_rate=0.001)
	model = DNN(trainer,tensorboard_dir='temp')

	return(block_8,model)
######################################################################################################################

### Training
#dat = scipy.io.loadmat('ecg_content+style_samples/style/Normal_full.mat')['val'][0,:]
os.chdir("C:/Users/Administrator/Desktop/cinc2017")

# load train and test data
ECG = scipy.io.loadmat("ECG") # ECG UpdatedECG
train_dat,train_lab = ECG["train_dat"],ECG["train_lab"]
test_dat,test_lab = ECG["test_dat"][0],ECG["test_lab"][0]

# best model
f = open("log/log.txt","w");f.close()
best_f1 = 0.9

# build model
interm,ECGmodel = build_model(input_sig_len,dropout_keep_prob)

# visualize intermediate activations
#visualize_activations(ECGmodel,interm,dat[None,2600:4100,None]);sys.exit()

### Epoch Tuning
for n in range(0,50):

	f = open("log/log.txt","a");f.write("\n"+"-"*80+" Epoch: "+str(n+1));f.close()
	ECGmodel.fit(train_dat,train_lab,n_epoch=1,shuffle=True,show_metric=True,batch_size=64)
	pred,f1 = prediction(test_dat,test_lab,ECGmodel)
	
	if f1 > best_f1:
		best_f1 = f1
		ECGmodel.save("log/ECG_model")
		scipy.io.savemat("log/pred",mdict={'pred':pred})
