import tflearn
import tensorflow as tf
from QingXia_Utils import *

data_type   = "la_4ch"  # data to use from UK biobank, 1 = LA, 2 = RA
fm 			= 16	    # feature map scale
kkk 		= 5		    # kernel size
keep_rate 	= 0.75	    # dropout

if data_type == "la_4ch":
	nx,ny = 80,96
elif data_type == "la_2ch":
	nx,ny = 64,80

# set main directory
os.chdir("/hpc/zxio506/UK_Biobank_Labelled/"+data_type+" Test")

### Evaluation Function ------------------------------------------------------------------------------------------------------------------
def evaluate(test_image,test_Label,CNN_model,log_path,mu=0,sd=1):

	# initialize output log file
	mse_scores = []
	
	# loop through all test patients
	for i in range(test_image.shape[0]):
		
		# make a prediction
		output = CNN_model.predict([(test_image[i][:,:,None]-mu)/sd])
		pred   = np.argmax(output[0],2)
		
		# get COM
		true = ndimage.measurements.center_of_mass(true > 0)
		pred = ndimage.measurements.center_of_mass(pred > 0)
		
		# evaluate
		mse_scores.append(np.sqrt((true[0] - pred[0])**2 + (true[1] - pred[1])**2))

	# overall score
	f = open(log_path,"a")
	f.write("\n\nOVERALL MSE = "+str(np.nanmean(mse_scores))+"\n\n\n")
	f.close()

	return(f1_scores)

def online_augmentation(data,label):
	
	# normalize the data
	data_mean,data_sd = np.mean(data),np.std(data)
	data = (data - data_mean)/data_sd
	
	return(data,label,data_mean,data_sd)

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
layer_0a_input	= tflearn.layers.core.input_data(shape=[None,nx,ny,1])

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
layer_0b_conv	= tflearn.layers.conv.conv_2d(layer_1b_add,3,1,1,trainable=True)
layer_0b_clf	= tflearn.activations.softmax(layer_0b_conv)

# loss function - C:\Users\Administrator\AppData\Roaming\Python\Python35\site-packages\tflearn\objectives.py
def dice_loss_2d(y_pred,y_true):
	
	with tf.name_scope("dice_loss_2D_function"):
		
		y_pred1,y_true1 = y_pred[:,:,:,1],y_true[:,:,:,1]
		intersection1   = tf.reduce_sum(y_pred1*y_true1)
		union1          = tf.reduce_sum(y_pred1*y_pred1) + tf.reduce_sum(y_true1*y_true1)
		dice1           = (2.0 * intersection1 + 1.0) / (union1 + 1.0)
		
		y_pred2,y_true2 = y_pred[:,:,:,2],y_true[:,:,:,2]
		intersection2   = tf.reduce_sum(y_pred2*y_true2)
		union2          = tf.reduce_sum(y_pred2*y_pred2) + tf.reduce_sum(y_true2*y_true2)
		dice2           = (2.0 * intersection2 + 1.0) / (union2 + 1.0)
		
	return(1.0 - (dice1*0.5 + dice2*0.5))

# Optimizer
regress = tflearn.layers.estimator.regression(layer_0b_clf,optimizer='adam',loss=dice_loss_2d,learning_rate=0.0001)
model   = tflearn.models.dnn.DNN(regress,tensorboard_dir='log')

### Training ------------------------------------------------------------------------------------------------------------------

# set up log file
log_path = "log/log.txt"

# load training data
data = h5py.File("Dataset.h5","r")

# keep track of best dice score
best_MSE = 9999
f = open(log_path,"w");f.write("mean: "+str(np.mean(data["train"]))+"\t std: "+str(np.std(data["train"])));f.close()

for n in range(1000):
	
	f = open(log_path,"a");f.write("-"*75+" Epoch "+str(n+1)+"\n");f.close()
	
	# online augmentation
	train_image,train_label,train_mean,train_sd = online_augmentation(data["train"],data["train_lab"])
	
	# run 1 epoch and evaluate current performance
	model.fit(train_image,train_label,n_epoch=1,show_metric=True,batch_size=32,shuffle=True)
	MSEs = evaluate(data["test"],data["test_lab"],model,log_path,train_mean,train_sd)
	
	# if the model is currently the best
	if np.mean(MSEs) < best_MSE:
		best_MSE = np.mean(MSEs)
		scipy.io.savemat("log/all_MSE_scores.mat",mdict={"dat":MSEs})