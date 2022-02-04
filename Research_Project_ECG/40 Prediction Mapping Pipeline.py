import numpy as np
import tensorflow as tf
import scipy.io
from sklearn.metrics import f1_score

def load_graph(frozen_graph_filename):

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,input_map=None,return_elements=None,name="prefix",op_dict=None,producer_op_list=None)

    sess_out = tf.Session(graph=graph)
    x_out = graph.get_tensor_by_name('prefix/InputData/X:0')
    y_out = graph.get_tensor_by_name('prefix/FullyConnected/Softmax:0')
	
    return(sess_out,x_out,y_out)



def prediction(pred_data,pred_label):
	
	pred,answer = [],[]
	
	for n in range(len(pred_data)):
		
		Sxx = (pred_data[n][0,:]-7.51190822991)/235.404723927
		Sxx_all = np.array([Sxx[i:(i+1500)] for i in range(0,len(Sxx)-1500+1,300)])[:,:,None]
		pred_prob = np.array(sess.run(y,{x:Sxx_all}))
		Sxx_all = np.pad(pred_prob,((0,58-len(pred_prob)),(0,0)),mode="constant")
		
		p1,p2,p3,p4,p5,p6,p7 = sess1.run(y1,{x1:[Sxx_all]}),sess2.run(y2,{x2:[Sxx_all]}),sess3.run(y3,{x3:[Sxx_all]}),sess4.run(y4,{x4:[Sxx_all]}),sess5.run(y5,{x5:[Sxx_all]}),sess6.run(y6,{x6:[Sxx_all]}),sess7.run(y7,{x7:[Sxx_all]})
		index = np.argmax(p1+p2+p3+p4+p5+p6+p7)
		
		pred.append(index)
		answer.append(np.argmax(pred_label[n]))
		
	f1 = f1_score(pred,answer,average=None)  
	
	f = open("log.txt","a")
	f.write("\n  Accuracy: "+str(np.round(np.sum(np.array(pred)==np.array(answer))/len(pred),3)))
	f.write("\n  F1 Score: "+str(np.round(np.mean(f1[:3]),3))+"   F1: "+str(f1)+"\n\n")
	f.close()
	
	return(np.array(pred),np.array(answer))



sess ,x ,y  = load_graph('entry/frozen_graph.pb.min')
sess1,x1,y1 = load_graph('entry/frozen_graph21.pb')
sess2,x2,y2 = load_graph('entry/frozen_graph22.pb')
sess3,x3,y3 = load_graph('entry/frozen_graph23.pb')
sess4,x4,y4 = load_graph('entry/frozen_graph24.pb')

sess5,x5,y5 = load_graph('entry/frozen_graph11.pb')
sess6,x6,y6 = load_graph('entry/frozen_graph12.pb')
sess7,x7,y7 = load_graph('entry/frozen_graph13.pb')

temp = scipy.io.loadmat("Dev-Test")
test_data,test_lab = temp["test"][0],temp["test_lab"]

pred,ans = prediction(test_data,test_lab)










# sess1,x1,y1 = load_graph('C:/Users/Administrator/Desktop/log/good/seq mapping/frozen_graph15.pb')
# sess2,x2,y2 = load_graph('C:/Users/Administrator/Desktop/log/good/seq mapping/frozen_graph35.pb')
# sess3,x3,y3 = load_graph('C:/Users/Administrator/Desktop/log/good/seq mapping/frozen_graph36.pb')
# sess4,x4,y4 = load_graph('C:/Users/Administrator/Desktop/log/good/seq mapping/frozen_graph40.pb')

# sess5,x5,y5 = load_graph('C:/Users/Administrator/Desktop/log/good/seq mapping/frozen_graph11.pb')
# sess6,x6,y6 = load_graph('C:/Users/Administrator/Desktop/log/good/seq mapping/frozen_graph14.pb')
# sess7,x7,y7 = load_graph('C:/Users/Administrator/Desktop/log/good/seq mapping/frozen_graph16.pb')

# def prediction2(pred_data,pred_label):
	# pred,answer = [],[]
	# for n in range(len(pred_data)):
		# Sxx_all = np.pad(pred_data[n],((0,max_len-len(pred_data[n])),(0,0)),mode="constant")
		
		# p1 = sess1.run(y1,feed_dict={x1:[Sxx_all]})
		# p2 = sess2.run(y2,feed_dict={x2:[Sxx_all]})
		# p3 = sess3.run(y3,feed_dict={x3:[Sxx_all]})
		# p4 = sess4.run(y4,feed_dict={x4:[Sxx_all]})
		
		# p5 = sess5.run(y5,feed_dict={x5:[Sxx_all]})
		# p6 = sess6.run(y6,feed_dict={x6:[Sxx_all]})
		# p7 = sess7.run(y7,feed_dict={x7:[Sxx_all]})
		
		# if np.argmax( p5+p6+p7 ) == 1:
			# pred.append(1)
		# else:
			# pred.append(np.argmax( p1+p2+p3+p4 + p8+p9+p10))
		
		# answer.append(pred_label[n])
	
	# f1 = f1_score(pred,answer,average=None)
	# f = open("log.txt","a")
	# f.write("\n  Accuracy: "+str(np.round(np.sum(np.array(pred)==np.array(answer))/len(pred),3)))
	# f.write("\n  F1 Score: "+str(np.round(np.mean(f1[:3]),3))+"   F1: "+str(f1)+"\n\n")
	# f. close()

# temp = scipy.io.loadmat("C:/Users/Administrator/Desktop/predictions1");max_len=58
# Test_X,Test_Y = temp["test_pred"][0],temp["test_label"][0]
# prediction2(Test_X,Test_Y)

# sys