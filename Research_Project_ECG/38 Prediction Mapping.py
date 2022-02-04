from tflearn.data_utils import to_categorical
from tflearn.data_augmentation import ImageAugmentation
import tflearn

from wavenet_utils import dilated_conv1d

import numpy as np
import scipy.io
from sklearn.metrics import f1_score

### helper function for evaluation
def prediction(pred_data,pred_label):
	
	pred,answer = [],[]
	
	for n in range(len(pred_data)):
		
		pred_prob = np.array(model.predict([pred_data[n]]))

		pred.append(np.argmax(pred_prob))
		answer.append(np.argmax(pred_label[n]))
		
	f1 = f1_score(pred,answer,average=None)
	
	f = open("log.txt","a")
	f.write("\n  Accuracy: "+str(np.round(np.sum(np.array(pred)==np.array(answer))/len(pred),3)))
	f.write("\n  F1 Score: "+str(np.round(np.mean(f1[:3]),3))+"   F1: "+str(f1)+"\n\n")
	f. close()
	
	return(np.mean(f1[:3]))

### Neural network
n_classes,max_len = 4,58

net = tflearn.input_data([None,max_len,n_classes])
net = tflearn.lstm(net,128,return_seq=True,dynamic=True)
net = tflearn.lstm(net,128,return_seq=True)
net = tflearn.lstm(net,128)

net = tflearn.fully_connected(net,n_classes,activation='softmax')
net = tflearn.regression(net,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy')
model = tflearn.DNN(net)

### Training
temp = scipy.io.loadmat("C:/Users/Administrator/Desktop/predictions1")
X,Y,Test_X,Test_Y = temp["train_pred"][0],temp["train_label"][0],temp["test_pred"][0],temp["test_label"][0]

X 		= np.array([ np.pad(X[i],((0,max_len-len(X[i])),(0,0)),mode="constant") for i in range(len(X))]) 
Test_X 	= np.array([ np.pad(Test_X[i],((0,max_len-len(Test_X[i])),(0,0)),mode="constant") for i in range(len(Test_X))]) 

Y,Test_Y = to_categorical(Y,n_classes),to_categorical(Test_Y,n_classes)

# # #
# temp2 = scipy.io.loadmat("C:/Users/Administrator/Desktop/predictions2")
# X2,Y2,Test_X2,Test_Y2 = temp2["train_pred"][0],temp2["train_label"][0],temp2["test_pred"][0],temp2["test_label"][0]

# X2 		= np.array([ np.pad(X2[i],((0,max_len-len(X2[i])),(0,0)),mode="constant") for i in range(len(X2))])
# Test_X2 = np.array([ np.pad(Test_X2[i],((0,max_len-len(Test_X2[i])),(0,0)),mode="constant") for i in range(len(Test_X2))]) 

# Y2,Test_Y2 = to_categorical(Y2,n_classes),to_categorical(Test_Y2,n_classes)

# X,Test_X = np.concatenate((X,X2)),np.concatenate((Test_X,Test_X2))
# Y,Test_Y = np.concatenate((Y,Y2)),np.concatenate((Test_Y,Test_Y2))
# # #

f = open("log.txt","w");f.close()
model.fit(X,Y,n_epoch=20,show_metric=True,shuffle=True,batch_size=64)
for i in range(21,100):
	
	model.fit(X,Y,n_epoch=1,show_metric=True,shuffle=True,batch_size=64)
	
	f = open("log.txt","a");f.write("\n"+"-"*80+" Epoch: "+str(i));f.close()
	output = prediction(Test_X,Test_Y)
	
	if output > 0.863: model.save("C:/Users/Administrator/Desktop/log/SeqMap"+str(i))
