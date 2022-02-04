#!/usr/bin/python3

import numpy as np
import scipy.io
import tensorflow as tf
import sys

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

def func(samples):
	
	# # Your classification algorithm goes here...
	Sxx = (samples-7.51190822991)/235.404723927
	Sxx_all = np.array([Sxx[i:(i+1500)] for i in range(0,len(Sxx)-1500+1,300)])[:,:,None]
	pred_prob = np.array(sess.run(y,{x:Sxx_all}))
	Sxx_all = np.pad(pred_prob,((0,58-len(pred_prob)),(0,0)),mode="constant")

	p1 = sess1.run(y1,{x1:[Sxx_all]})

	index = np.argmax(p1)
	return(index)


sess ,x ,y  = load_graph('entry/frozen_graph.pb.min')
sess1,x1,y1 = load_graph('entry/frozen_graph21.pb')


temp = scipy.io.loadmat("Dev-Test")

data		= temp["train"][0]
labels		= temp["train_lab"]             
test_data	= temp["test"][0]
test_lab	= temp["test_lab"]

pred = [] 
for i in range(len(test_lab)):
    pred.append(func(test_data[i][0,:]))
    print(i)

pred = np.array(pred)

real = np.array([np.argmax(test_lab[i]) for i in range(len(test_lab))])
