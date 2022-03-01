import os
import cv2
import tflearn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

os.chdir("C:/Users/zxio506/Desktop/VT Data")

files = os.listdir()
train_files = [s for s in files if "train" in s]
test_files = [s for s in files if "test" in s]

# training data
X_train, Y_train = [], []

for i in range(len(train_files)):
	
	d_mat = scipy.io.loadmat(train_files[i])
	img1, img2, img3 = d_mat["img1"], d_mat["img2"], d_mat["img3"]
	X_train_i, Y_train_i = d_mat["X_train"], d_mat["Y_train"]

	X_train.append(img1[:,:,:(img1.shape[2]-2)])
	X_train.append(img2[:,:,:(img2.shape[2]-2)])
	X_train.append(img3[:,:,:(img3.shape[2]-2)])
	
	img1_lab = img1[:,:,-1]
	img2_lab = img2[:,:,-1]
	img3_lab = img3[:,:,-1]
	
	img1_lab = cv2.dilate(img1_lab,np.ones((3,3),np.uint8),iterations=1)
	img2_lab = cv2.dilate(img2_lab,np.ones((3,3),np.uint8),iterations=1)
	img3_lab = cv2.dilate(img3_lab,np.ones((3,3),np.uint8),iterations=1)

	Y_train.append(img1_lab)
	Y_train.append(img2_lab)
	Y_train.append(img3_lab)
	
X_train, Y_train = np.array(X_train), np.array(Y_train)

temp = np.zeros([Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],2])
temp[:,:,:,0] = 1 - Y_train
temp[:,:,:,1] = Y_train
Y_train = np.copy(temp)

scipy.io.savemat("../train.mat", 
				 mdict={"X_train": X_train, "Y_train": Y_train})
				 
# testing data
X_test, Y_test = [], []

for i in range(len(test_files)):
	
	d_mat = scipy.io.loadmat(test_files[i])
	img1, img2, img3 = d_mat["img1"], d_mat["img2"], d_mat["img3"]
	X_train_i, Y_train_i = d_mat["X_test"], d_mat["Y_test"]

	X_test.append(img1[:,:,:(img1.shape[2]-2)])
	X_test.append(img2[:,:,:(img2.shape[2]-2)])
	X_test.append(img3[:,:,:(img3.shape[2]-2)])
	
	img1_lab = img1[:,:,-1]
	img2_lab = img2[:,:,-1]
	img3_lab = img3[:,:,-1]
	
	img1_lab = cv2.dilate(img1_lab,np.ones((3,3),np.uint8),iterations=1)
	img2_lab = cv2.dilate(img2_lab,np.ones((3,3),np.uint8),iterations=1)
	img3_lab = cv2.dilate(img3_lab,np.ones((3,3),np.uint8),iterations=1)

	Y_test.append(img1_lab)
	Y_test.append(img2_lab)
	Y_test.append(img3_lab)

X_test, Y_test = np.array(X_test), np.array(Y_test)

scipy.io.savemat("../test.mat", 
				 mdict={"X_test": X_test, "Y_test": Y_test})
