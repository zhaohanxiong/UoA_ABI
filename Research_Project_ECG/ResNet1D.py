import tflearn
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score

n,steps = 5,1500
# Residual blocks --- 32 layers: n=5, 56 layers: n=9, 110 layers: n=18

def prediction(pred_data,pred_label,DNN_model):
	
	pred,answer = [],[]
	
	for n in range(pred_data.shape[0]):
	
		Sxx = (pred_data[n][0,:] - train_mean)/train_sd

		Sxx_all = np.array([Sxx[i:(i+steps)] for i in range(0,len(Sxx)-steps,300)])
		Sxx_all = np.expand_dims(Sxx_all,axis=2)
		pred_prob = np.array(DNN_model.predict(Sxx_all))
		
		pred.append(np.argmax(np.multiply.reduce(pred_prob,0)))		#pred.append(np.argmax(pred_prob.mean(0)))
		answer.append(np.argmax(pred_label[n]))

	f1 = f1_score(pred,answer,average=None)
	
	f = open("log.txt","a")
	f.write("\n  Accuracy: "+str(np.round(np.sum(np.array(pred)==np.array(answer))/len(pred),3)))
	f.write("\n  F1 Score: "+str(np.round(np.mean(f1[:3]),3))+"   F1: "+str(f1)+"\n\n")
	f.close()

	return(np.mean(f1[:3]))

def residual_block_1D(incoming,n_blocks,out_channels,downsample=False):
	
	resnet = incoming
	in_channels = incoming.shape[2].value 
	strides = [2 if downsample else 1][0]
	
	for i in range(n_blocks):
		
		identity = resnet
		
		resnet = tflearn.batch_normalization(resnet)
		resnet = tflearn.activations.relu(resnet)
		resnet = tflearn.layers.conv.conv_1d(resnet,out_channels,3,strides,'same','linear',True,'variance_scaling','zeros','L2',0.0001)
		
		if downsample:
			identity = tflearn.layers.conv.avg_pool_1d(identity,strides)
	
		if in_channels != out_channels:
		
			ch = (out_channels - in_channels) // 2
			identity = tf.pad(identity,[[0,0],[0,0],[ch,ch]])
			in_channels = out_channels
		
		resnet = resnet + identity
		
	return(resnet)

# Building Residual Network
net = tflearn.layers.core.input_data(shape=[None,steps,1])
net = tflearn.layers.conv.conv_1d(net,16,3,regularizer='L2',weight_decay=0.0001)

net = residual_block_1D(net,  n,16)
net = residual_block_1D(net,  1,32,downsample=True)
net = residual_block_1D(net,n-1,32)
net = residual_block_1D(net,  1,64,downsample=True)
net = residual_block_1D(net,n-1,64)
net = tflearn.batch_normalization(net)
net = tflearn.activations.relu(net)
net = tf.reduce_mean(net,1)
net = tflearn.fully_connected(net,4,activation='softmax')

mom = tflearn.Momentum(0.1,lr_decay=0.1,decay_step=32000,staircase=True)
net = tflearn.regression(net,optimizer=mom,loss='categorical_crossentropy')
model = tflearn.DNN(net,clip_gradients=0.)

# Training
from Pre_RNN import Raw_Signal_Input
X,Y,Test_X,Test_Y,train_mean,train_sd = Raw_Signal_Input(steps)

f = open("log.txt","w");f.write("Training Set Mean: "+str(train_mean)+" Training Set Standard Deviation: "+str(train_sd));f.close()

model.fit(X,Y,n_epoch=50,shuffle=True,show_metric=True,batch_size=128)

for n in range(51,75):
	
	model.fit(X,Y,n_epoch=1,shuffle=True,show_metric=True,batch_size=128)
	
	f = open("log.txt","a");f.write("\n"+"-"*80+" Epoch: "+str(n));f.close()
	output = prediction(Test_X,Test_Y,model)
	
	if output >= 0.84: model.save("C:/Users/Administrator/Desktop/log/ResNet1D"+str(n))
