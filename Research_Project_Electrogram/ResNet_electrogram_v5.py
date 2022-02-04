from tflearn.layers.core import input_data,fully_connected
from tflearn.layers.conv import conv_2d,residual_block,global_avg_pool,max_pool_2d
from tflearn.layers.normalization import batch_normalization
from tflearn.activations import relu
from tflearn.layers.estimator import regression
from tflearn.models.dnn import DNN

import scipy.io
import numpy as np
import copy
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

###############################################################################################################################################
# Input Block
n = 8 # 32 layers: n=5, 56 layers: n=9, 110 layers: n=18

net1 = input_data(shape=[None,60,60,1])
#net = max_pool_2d(net,4)
net2 = conv_2d(net1,16,3,regularizer='L2',weight_decay=0.0001)
net3 = residual_block(net2,n,16)
net4 = residual_block(net3,1,32,downsample=True)
net5 = residual_block(net4,n-1,32)
net6 = residual_block(net5,1,64,downsample=True)
net7 = residual_block(net6,n-1,64)
net8 = global_avg_pool(relu(batch_normalization(net7)))

# Regression
net 	= fully_connected(net8,2,activation='sigmoid')
trainer = regression(net,optimizer='rmsprop',loss='mean_square',learning_rate=0.001)
model 	= DNN(trainer)

###############################################################################################################################################

def write_to_file(str):

	# open, write, close file
	f = open(working_dir+"log.txt","a")
	f.write(str)
	f.close()

def grid_search_threshold_vals(pred_dat,true_dat):
	
	pred_dat,true_dat,scores = pred_dat.flatten(),true_dat.flatten(),np.zeros([100,100])
	
	for i in range(10,50):
		for j in range(10,50):

			thres1,thres2 = i/100,j/100
			
			pp,gt = copy.copy(pred_dat),copy.copy(true_dat)
			pp[pp < (np.max(pp)*thres1)] = 0
			pp[pp >=(np.max(pp)*thres1)] = 1
			#gt[gt < (np.max(gt)*thres2)] = 0
			#gt[gt >=(np.max(gt)*thres2)] = 1

			scores[i,j] = np.mean(f1_score(pp,gt,average=None))
	
	x1,x2 = np.where(scores == np.max(scores))

	return(x1[0]/100,x2[0]/100)

def predict_and_evaluate(CNN_model,data,label,label_PS,mu=0,sd=1):
	
	f1_all,MSE_all = [],[]
	
	#thres1 = [0.24, 0.49, 0.49, 0.10, 0.38, 0.44, 0.14, 0.10, 0.33, 0.11]
	#thres2 = [0.19, 0.32, 0.45, 0.25, 0.47, 0.49, 0.44, 0.21, 0.19, 0.13]
	
	# set up window size for labels
	factor 			= 4 # how much Phi was downsampled
	sub_region_size = 8 # in original 240 x 240 resolution
	n_window 		= int(sub_region_size / factor)

	# Making predictions for each data	
	for n in range(0,data.shape[1]):
		
		print("Processing Test Data "+str(n))
		
		data_n,label_n = (data[0,n]-mu)/sd,np.squeeze(label[n,:,:])
		
		# feed patches
		out,pred_batch_size = [],256
		for i in range(0,data_n.shape[0],pred_batch_size):
			out.extend(CNN_model.predict(  data_n[i:(i+pred_batch_size),:,:,:] ))
		
		out = np.array(out)
		
		# reconstruct patches
		pred,pred_MSE = np.zeros([data_n.shape[1]//n_window,data_n.shape[2]//n_window]),[]
		for i in range(data_n.shape[0]):
			pred[int(np.round(out[i][0]*pred.shape[0])),int(np.round(out[i][1]*pred.shape[1]))] += 1
			pred_MSE.append( (((label_PS[0,n][i,0]/30*120)-(out[i][0]/30*120))**2 + 
									((label_PS[0,n][i,1]/30*120)-(out[i][1]/30*120))**2)**0.5 )
			
		#pred = scipy.io.loadmat("C:\\Users\\Administrator\\Desktop\\model\\temp_pred"+str(n+1)+".mat")["pred"]	
		#scipy.io.savemat("C:\\Users\\Administrator\\Desktop\\model\\temp_pred"+str(n+1)+".mat",mdict={'pred':pred,'label':label_n})
		
		# identify driver region
		thres_pred,thres_true = grid_search_threshold_vals(pred,label_n);print(thres_pred,thres_true)
		#thres_pred,thres_true = thres1[n],thres2[n]
		
		pred[pred < (np.max(pred)*thres_pred)] = 0
		pred[pred >=(np.max(pred)*thres_pred)] = 1
		
		label_n[label_n < (np.max(label_n)*thres_true)] = 0
		label_n[label_n >=(np.max(label_n)*thres_true)] = 1

		# evaluate and write to output
		pred,true = pred.flatten(),label_n.flatten()
		
		f1 = np.mean(f1_score(pred,true,average=None))
		MSE = np.mean(np.array(pred_MSE))
		
		write_to_file("F1 Score: "+str(round(f1,3))+"\t")
		write_to_file("MSE: "+str(round(MSE,3))+"\n")

		# store f1 score
		f1_all.append(f1)
		MSE_all.append(MSE)

	# get overall mean and output to file
	f1_all = np.mean(np.array(f1_all))
	write_to_file("\n   Overall F1 Score: "+str(round(f1_all,3))+"\t")
	
	MSE_all = np.mean(np.array(MSE_all))
	write_to_file("   Overall MSE: "+str(round(MSE_all,3))+"\n")
	
	return(f1_all)
	
def predict_single_data(CNN_model,data,mu,sd):

	# format the shape of the data
	if data.shape[2] > data.shape[0] and data.shape[2] > data.shape[1]:
		data = np.rollaxis(data,2,0)
	if len(data.shape) == 3:
		data = data[:,:,:,None]
	
	# set up window size for labels
	factor 			= 4 # how much Phi was downsampled
	sub_region_size = 8 # in original 240 x 240 resolution
	n_window 		= int(sub_region_size / factor)
	
	data = (data-mu)/sd

	# feed patches
	out,pred_batch_size = [],256
	for i in range(0,data.shape[0],pred_batch_size):
		out.extend(CNN_model.predict( data[i:(i+pred_batch_size),:,:,:] ))
	
	out = np.array(out)
	
	# reconstruct patches
	pred = np.zeros([data.shape[1]//n_window,data.shape[2]//n_window])
	for i in range(data.shape[0]):
		pred[int(np.round(out[i][0]*(pred.shape[0]-1))),int(np.round(out[i][1]*(pred.shape[1]-1)))] += 1
	
	#scipy.io.savemat("C:\\Users\\Administrator\\Desktop\\temp_pred.mat",mdict={'pred':pred})
	plt.imshow(pred,cmap='jet');plt.show()
	return(pred)
	
### Training ##################################################################################################################################

model.load("best model\\model5")

data_to_predict = scipy.io.loadmat("C:\\Users\\Administrator\\Desktop\\Phi_Germany.mat")["ECP"]
pred = predict_single_data(model,data_to_predict,0,1);sys.exit()

working_dir = "C:\\Users\\Administrator\\Desktop\\"
data = scipy.io.loadmat(working_dir+"Electrogram_Data.mat")
write_to_file(str(data["train_mean"][0][0])+" "+str(data["train_sd"][0][0])+"\n")
predict_and_evaluate(model,data["test"],data["test_label"],data["test_label_PS"],data["train_mean"],data["train_sd"]);sys.exit()

for n in range(0,50):
	
	write_to_file("\n"+"-"*50+" Epoch "+str(n+1)+"\n")
	model.fit(data["train"],data["train_label"],n_epoch=1,show_metric=True,batch_size=128,shuffle=True)
	if predict_and_evaluate(model,data["test"],data["test_label"],data["test_label_PS"],data["train_mean"],data["train_sd"]) >= 0.85:
		model.save(working_dir+"model\\model"+str(n+1))
