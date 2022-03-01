import os
import tflearn
import scipy.io
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

#os.chdir("C:/Users/zxio506/Desktop")

n1 = 32 # x
n2 = 32 # y

# level 0 input
net	= tflearn.layers.core.input_data(shape=[None,n1,n2,18])

n = 9
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.residual_block(net, n, 16)
net = tflearn.residual_block(net, 1, 32, downsample=False)
net = tflearn.residual_block(net, n-1, 32)
net = tflearn.residual_block(net, 1, 64, downsample=False)
net = tflearn.residual_block(net, n-1, 64)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')

# level 0 classifier
net	= tflearn.layers.conv.conv_2d(net,2,1,1,trainable=True)
layer_0b_clf = tflearn.activations.softmax(net)

# classifier
regress = tflearn.layers.estimator.regression(layer_0b_clf,optimizer='adam',
					loss="softmax_categorical_crossentropy",learning_rate=0.00001)
model = tflearn.models.dnn.DNN(regress)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# train test
d_mat = scipy.io.loadmat("train.mat")
d_mat_test = scipy.io.loadmat("test.mat")

best_f1 = 0

for n in range(100):

	# train
	model.fit(d_mat["X_train"],d_mat["Y_train"],n_epoch=1,show_metric=True,batch_size=10,shuffle=True)
	
	# train
	pred_train_raw = model.predict(d_mat["X_train"])

	# test
	pred_raw = model.predict(d_mat_test["X_test"])
	pred = np.argmax(pred_raw,3)

	score = []
	for i in range(d_mat_test["Y_test"].shape[0]):
		
		true_flat,pred_flat = d_mat_test["Y_test"][i].flatten(),pred[i].flatten()
		
		f1 = f1_score(true_flat,pred_flat,average=None)[1]

		score.append(f1)
		
	score = np.array(score)
	
	if np.mean(score) > best_f1: 
		best_f1 = np.mean(score)
		scipy.io.savemat("out/pred_train"+str(n)+".mat",mdict={"pred":pred_train_raw})
		scipy.io.savemat("out/pred"+str(n)+".mat",mdict={"pred":pred_raw,"scores":score})

	print("="*50)
	print(best_f1)
	print("="*50)
