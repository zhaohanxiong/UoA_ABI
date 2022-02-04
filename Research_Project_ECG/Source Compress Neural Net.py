'''
https://stackoverflow.com/questions/38947658/tensorflow-saving-into-loading-a-graph-from-a-file
https://gist.github.com/tokestermw/795cc1fd6d0c9069b20204cbd133e36b

https://github.com/tomasreimers/tensorflow-graph-compression
https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
'''

import numpy as np
import scipy.io
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.python.framework import graph_util

def prediction(pred_data,pred_label,input_layer,output_layer,DNN_model_session):
	
	pred,answer = [],[]
	
	for n in range(pred_data.shape[0]):
	
		Sxx = (pred_data[n][0,:] - train_mean)/train_sd

		Sxx_all = np.array([Sxx[i:(i+steps)] for i in range(0,len(Sxx)-steps,300)]) # *** make predictions in parrallel to increase effeciency (tune the step size)
		Sxx_all = np.expand_dims(Sxx_all,axis=2)
		
		pred_prob = np.array(DNN_model_session.run(output_layer,feed_dict={input_layer:Sxx_all}))
		
		pred.append(np.argmax(np.multiply.reduce(pred_prob,0)))		#pred.append(np.argmax(pred_prob.mean(0)))
		answer.append(np.argmax(pred_label[n]))

	f1 = f1_score(pred,answer,average=None)
	
	f = open("log2.txt","a")
	f.write("\n  Accuracy: "+str(np.round(np.sum(np.array(pred)==np.array(answer))/len(pred),3)))
	f.write("\n  F1 Score: "+str(np.round(np.mean(f1[:3]),3))+"   F1: "+str(f1)+"\n\n")
	f.close()

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph

## freezing graph to protobuf and loading it
# from original script (run the original script and build the graph from scratch, then apply these operations):
#model.load("C:/Users/Administrator/Desktop/model/ckpt")
#with model.session.graph.as_default():
	#del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
#model.save("C:/Users/Administrator/Desktop/model/ckpt")
#temp = scipy.io.loadmat("Dev-Test")
#test_data = temp["test"][0]; test_lab = temp["test_lab"]; train_mean = 7.543; train_sd = 240.133; steps = 1500
#prediction(test_data,test_lab,model)

def pred_temp(sess,x,y):

	temp = scipy.io.loadmat("Dev-Test")
	test_data = temp["test"][0]; train_mean = 7.543  ; train_sd = 240.133 ; steps = 1500
	temp = (test_data[1][0,:]-train_mean)/train_sd
	
	Sxx_all = np.array([temp[i:(i+steps)] for i in range(0,len(temp)-steps,300)])
	Sxx_all = np.expand_dims(Sxx_all,axis=2)
	pred_prob = sess.run(y,feed_dict={x:Sxx_all})
	
	out = np.multiply.reduce(pred_prob,0)
	
	f = open("log2.txt","a")
	f.write(str(out)+"\n\n")
	f.close()

# load model
#input_checkpoint = "C:/Users/zxio506/Desktop/model/ckpt"
#sess = tf.Session()
#saver = tf.train.import_meta_graph(input_checkpoint+'.meta',clear_devices=True)
#saver.restore(sess,input_checkpoint)

#graph = sess.graph
#graph = tf.get_default_graph()
#input_graph_def = graph.as_graph_def()

sess = model.session
graph = model.session.graph
x = graph.get_tensor_by_name('InputData/X:0')
y = graph.get_tensor_by_name('FullyConnected/Softmax:0')

pred_temp(sess,x,y);sys.exit()

input_graph_def = graph.as_graph_def()

# fix batch norm bug in nodes
for node in input_graph_def.node:
  if node.op == 'RefSwitch':
    node.op = 'Switch'
    for index in range(len(node.input)):
      if 'moving_' in node.input[index]:
        node.input[index] = node.input[index] + '/read'
  elif node.op == 'AssignSub':
    node.op = 'Sub'
    if 'use_locking' in node.attr: del node.attr['use_locking']

# freezing graph
#temp = graph_util.convert_variables_to_constants(sess,input_graph_def,['output_data/Softmax'])
#tf.train.write_graph(temp,'','C:/Users/Administrator/Desktop/frozen_graph.pb',False)
output_graph = 'C:/Users/zxio506/Desktop/frozen_graph.pb'
output_graph_def = graph_util.convert_variables_to_constants(sess,input_graph_def,['output_data/Softmax'])
tf.gfile.GFile(output_graph, "wb").write(output_graph_def.SerializeToString())
print("%d ops in the final graph." % len(output_graph_def.node))

# compression
#...

# loading graph
graph = load_graph('C:/Users/zxio506/Desktop/frozen_graph.pb')

x = graph.get_tensor_by_name('prefix/input_data/X:0')
y = graph.get_tensor_by_name('prefix/output_data/Softmax:0')

sess = tf.Session(graph=graph)

# testing
temp = scipy.io.loadmat("Dev-Test")
test_data = temp["test"][0]; test_lab = temp["test_lab"]; train_mean = 7.543  ; train_sd = 240.133 ; steps = 1500

prediction(test_data,test_lab,x,y,sess)