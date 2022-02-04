from tflearn.layers.core import input_data,fully_connected
from tflearn.layers.conv import conv_2d,residual_block,global_avg_pool,max_pool_2d
from tflearn.layers.normalization import batch_normalization
from tflearn.activations import relu
from tflearn.layers.estimator import regression
from tflearn.models.dnn import DNN

import tflearn
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

def visualize_activations(CNN_model,layer,stimuli):

	# plt.imshow(data["train"][8][:,:,0]);plt.show()
	
	get_activation = tflearn.DNN(layer,session=CNN_model.session)
	activation = get_activation.predict(stimuli)
	
	ind = np.sort(np.random.choice([i for i in range(activation.shape[3])], 16, replace = False))
	
	count = 0
	for _ in range(4):
		for _ in range(4):
			plt.subplot(4, 4, count+1)
			plt.imshow(activation[0][:,:,ind[count]],cmap='jet')
			count += 1

	plt.show()

# Input Block
n = 8 # 32 layers: n=5, 56 layers: n=9, 110 layers: n=18

net1 = input_data(shape=[None,60,60,1])
#net = max_pool_2d(net,4)
net2 = conv_2d(net1,16,3,regularizer='L2',weight_decay=0.0001)	# layer 1
net3 = residual_block(net2,n,16)								# layer 19
net4 = residual_block(net3,1,32)								# layer 21
net5 = residual_block(net4,n-1,32)								# layer 37
net6 = residual_block(net5,1,64)								# layer 39
net7 = residual_block(net6,n-1,64)								# layer 55
net8 = global_avg_pool(relu(batch_normalization(net7)))

# Regression
net 	= fully_connected(net8,2,activation='sigmoid')
trainer = regression(net,optimizer='rmsprop',loss='mean_square',learning_rate=0.001)
model 	= DNN(trainer)


working_dir = "C:\\Users\\Administrator\\Desktop\\"
data = scipy.io.loadmat(working_dir+"Electrogram_Data.mat")

#plt.imshow(data["train"][8][:,:,0],cmap='jet');plt.show();sys.exit()
sys.exit()
model.load("best model\\model5")
visualize_activations(model,net5,data["train"][8][None,:,:,:])
