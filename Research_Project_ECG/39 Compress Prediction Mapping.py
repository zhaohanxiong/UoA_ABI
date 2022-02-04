
'''
steps for converting tflearn model to frozen tensorflow protobuf file:

- define the computation graph (tflearn)
- load model from checkpoint (3 or 4 files)
- run model once to initialise the variables
- fix bug #1
- transfer graph and and values to a tensorflow session
- fix bug #2
- convert all variables to constants
- write graph to output
'''

from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.conv import conv_1d
from tflearn.layers.estimator import regression
from tflearn.models.dnn import DNN
from tflearn.data_utils import to_categorical
import tflearn

from wavenet_utils import dilated_conv1d

import tensorflow as tf
import numpy as np
import scipy.io
from tensorflow.python.framework import graph_util

## --- Defining Computational Graph ##################################################################################
import numpy as np
from sklearn.metrics import f1_score
import scipy.io
import time
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

n_classes,max_len = 4,58

net = tflearn.input_data([None,max_len,n_classes])
net = tflearn.lstm(net,128,return_seq=True,dynamic=True)
net = tflearn.lstm(net,128,return_seq=True)
net = tflearn.lstm(net,128)

net = tflearn.fully_connected(net,n_classes,activation='softmax')
net = tflearn.regression(net,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy')
model = tflearn.DNN(net)
#####################################################################################################################

## --- Loading trained model from checkpoint
model.load("C:/Users/Administrator/Desktop/log/0.865")
_=model.predict(np.zeros([1,max_len,n_classes])) # have to make a prediction after loading the model to initialise the vars

## --- fix "Adam not defined" bug
with model.session.as_default():
	del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]

## --- Transferring tflearn graph to tensorflow session
sess = model.session
input_graph_def = sess.graph_def

## --- fix "bach norm" bug
for node in input_graph_def.node:
  if node.op == 'RefSwitch':
    node.op = 'Switch'
    for index in range(len(node.input)):
      if 'moving_' in node.input[index]:
        node.input[index] = node.input[index] + '/read'
  elif node.op == 'AssignSub':
    node.op = 'Sub'
    if 'use_locking' in node.attr: del node.attr['use_locking']

## --- freezing graph
output_graph = 'C:/Users/Administrator/Desktop/frozen_graph41.pb'
output_graph_def = graph_util.convert_variables_to_constants(sess,input_graph_def,['FullyConnected/Softmax'])
tf.gfile.GFile(output_graph, "wb").write(output_graph_def.SerializeToString())