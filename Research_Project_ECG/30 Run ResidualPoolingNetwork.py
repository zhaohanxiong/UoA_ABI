import numpy as np
import scipy.io
from sklearn.metrics import f1_score
import tensorflow as tf

def load_graph(frozen_graph_filename):

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,input_map=None,return_elements=None,name="prefix",op_dict=None,producer_op_list=None)
    return graph

# loading graph
graph = load_graph('C:/Users/Administrator/Desktop/0.839 0.832.pb')

x = graph.get_tensor_by_name('prefix/InputData/X:0')
y = graph.get_tensor_by_name('prefix/FullyConnected/Softmax:0')

sess = tf.Session(graph=graph)

# testing
def prediction(pred_data,pred_label,input_layer,output_layer,DNN_model_session):
	
	pred,answer,pred_raw = [],[],[]
	
	for n in range(pred_data.shape[0]):
		print(n)
		Sxx = (pred_data[n][0,:] - train_mean)/train_sd

		Sxx_all = np.array([Sxx[i:(i+steps)] for i in range(0,len(Sxx)-steps,300)]) # *** make predictions in parrallel to increase effeciency (tune the step size)
		Sxx_all = np.expand_dims(Sxx_all,axis=2)
		
		pred_prob = np.array(DNN_model_session.run(output_layer,feed_dict={input_layer:Sxx_all}))

		pred.append(np.argmax(np.multiply.reduce(pred_prob,0)))
		answer.append(np.argmax(pred_label[n]))
		
		pred_raw.append(pred_prob)
	
	f1 = f1_score(pred,answer,average=None)
	
	f = open("log.txt","a")
	f.write("\n  Accuracy: "+str(np.round(np.sum(np.array(pred)==np.array(answer))/len(pred),3)))
	f.write("\n  F1 Score: "+str(np.round(np.mean(f1[:3]),3))+"   F1: "+str(f1)+"\n\n")
	f.close()
	
	return(np.array(pred_raw),np.array(answer))

temp = scipy.io.loadmat("Dev-Test")
data,labels,test_data,test_lab = temp["train"][0],temp["train_lab"],temp["test"][0],temp["test_lab"]
train_mean = 7.51190822991  ; train_sd = 235.404723927; steps = 1500

pred1,answer1 = prediction(data,labels,x,y,sess)
pred2,answer2 = prediction(test_data,test_lab,x,y,sess)

scipy.io.savemat("C:/Users/Administrator/Desktop/predictions2",mdict={"train_pred":pred1,"train_label":answer1,"test_pred":pred2,"test_label":answer2})