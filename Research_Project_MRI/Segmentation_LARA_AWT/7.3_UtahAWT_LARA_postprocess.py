import os
import cv2
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# set directory location
os.chdir("C:/Users/zxio506/Desktop/WaikatoAWT Test Set")

# list files
files = os.listdir("Prediction Sample")

# make files
#os.mkdir("Image Stack")
#os.mkdir("AWT Distribution")

# store output scores
column_headers = ["MSE","MSE_masked","awt_true_mean","awt_pred_mean","awt_err","awt_err_per","value_err"]

test_patients  = os.listdir("Prediction Sample")

RAawt_score  = np.zeros([len(test_patients),len(column_headers)])
LAawt_score  = np.zeros([len(test_patients),len(column_headers)])

# loop through all files
for n in range(len(files)):
	
	print("Processing "+files[n])
	
	# read prediction/ground truth
	dat = scipy.io.loadmat("Prediction Sample/"+files[n])
	
	true_awt,pred_awt = dat["true"].astype(np.int8),dat["pred"].astype(np.int8)

	LARA_wall = dat["input_seg"].astype(np.int8)
	LARA_wall[LARA_wall>2] = 0
	
	# seperate left and right walls
	RA_true_awt,LA_true_awt = np.copy(true_awt),np.copy(true_awt)
	RA_pred_awt,LA_pred_awt = np.copy(pred_awt),np.copy(pred_awt)

	RA_true_awt[LARA_wall != 1] = 0
	LA_true_awt[LARA_wall != 2] = 0
	
	RA_pred_awt[LARA_wall != 1] = 0
	LA_pred_awt[LARA_wall != 2] = 0

	if True:
	
		RA_true,RA_pred = RA_true_awt[RA_true_awt>0].flatten()*0.625,RA_pred_awt[RA_pred_awt>0].flatten()*0.625
		LA_true,LA_pred = LA_true_awt[LA_true_awt>0].flatten()*0.625,LA_pred_awt[LA_pred_awt>0].flatten()*0.625
		
		RA_true,RA_pred = RA_true[RA_true<10],RA_pred[RA_pred<10]
		LA_true,LA_pred = LA_true[LA_true<7],LA_pred[LA_pred<7]
		
		# initialize figure
		plt.figure(figsize=(6,4))
		
		plt.subplot(121)
		plt.title("RA Wall Thickness Distribution")
		plt.hist(RA_true,15,alpha=0.75,edgecolor='black',linewidth=0.5,color="deepskyblue")
		plt.hist(RA_pred,15,alpha=0.5,edgecolor='black',linewidth=0.5,color="tomato")
		plt.yticks([])

		plt.subplot(122)
		plt.title("LA Wall Thickness Distribution")
		plt.hist(LA_true,7,alpha=0.75,edgecolor='black',linewidth=0.5,color="deepskyblue")
		plt.hist(LA_pred,7,alpha=0.5,edgecolor='black',linewidth=0.5,color="tomato")
		plt.yticks([])
		#plt.show();sys.exit()
		
		plt.savefig("AWT Distribution/"+files[n]+".png")
		plt.close()

	if False: # True False
	
		# write to image stack
		os.mkdir("Image Stack/"+test_patients[n])
		os.mkdir("Image Stack/"+test_patients[n]+"/true_awt")
		os.mkdir("Image Stack/"+test_patients[n]+"/pred_awt")
		os.mkdir("Image Stack/"+test_patients[n]+"/seg")
		
		for i in range(true_awt.shape[2]):
	
			temp1 = cv2.dilate(true_awt[:,:,i],np.ones((3,3),np.uint8),iterations = 1)
			temp2 = cv2.dilate(pred_awt[:,:,i],np.ones((3,3),np.uint8),iterations = 1)
			
			img_name = "{0:03}".format(i+1)+".tif"
			cv2.imwrite("Image Stack/"+test_patients[n]+"/true_awt/"+img_name,temp1)
			cv2.imwrite("Image Stack/"+test_patients[n]+"/pred_awt/"+img_name,temp2)
			cv2.imwrite("Image Stack/"+test_patients[n]+"/seg/"     +img_name,LARA_wall[:,:,i])
	
	if True: # True False

		## ----------------------------------------------------------------------------------------------- RA AWT
	
		# mean square distance error
		RAawt_score[n,0] = np.mean((RA_true_awt - RA_pred_awt)**2) * 0.625
		
		# mean square distance error for only the atrial wall pixels
		RAawt_score[n,1] = np.mean((RA_true_awt[RA_true_awt>0] - RA_pred_awt[RA_true_awt>0])**2) * 0.625
		
		# mean thickness errors
		RAawt_score[n,2] = np.mean(RA_true_awt[RA_true_awt>0]) * 0.625
		RAawt_score[n,3] = np.mean(RA_pred_awt[RA_pred_awt>0]) * 0.625
		RAawt_score[n,4] = np.abs(RAawt_score[n,2] - RAawt_score[n,3])
		RAawt_score[n,5] = np.abs(RAawt_score[n,2] - RAawt_score[n,3])/RAawt_score[n,2]*100
		
		# average error between values
		RAawt_score[n,6] = np.mean(np.abs(RA_true_awt[RA_true_awt>0] - RA_pred_awt[RA_true_awt>0])) * 0.625

		## ----------------------------------------------------------------------------------------------- LA AWT

		# mean square distance error
		LAawt_score[n,0] = np.mean((LA_true_awt - LA_pred_awt)**2) * 0.625
	
		# mean square distance error for only the atrial wall pixels
		LAawt_score[n,1] = np.mean((LA_true_awt[LA_true_awt>0] - LA_pred_awt[LA_true_awt>0])**2) * 0.625
		
		# mean thickness errors
		LAawt_score[n,2] = np.mean(LA_true_awt[LA_true_awt>0]) * 0.625
		LAawt_score[n,3] = np.mean(LA_pred_awt[LA_pred_awt>0]) * 0.625
		LAawt_score[n,4] = np.abs(LAawt_score[n,2] - LAawt_score[n,3])
		LAawt_score[n,5] = np.abs(LAawt_score[n,2] - LAawt_score[n,3])/LAawt_score[n,2]*100
		
		# average error between values
		LAawt_score[n,6] = np.mean(np.abs(LA_true_awt[LA_true_awt>0] - LA_pred_awt[LA_true_awt>0])) * 0.625

# Save data to csv output
pd.DataFrame(RAawt_score).to_csv("RA_awt.csv",header=column_headers,index=False)
pd.DataFrame(LAawt_score).to_csv("LA_awt.csv",header=column_headers,index=False)

# summarize
print("\n\n---------------------- RA AWT Evaluation")
print("   MSE Masked: "+str(np.mean(RAawt_score[:,1])))
print("    AWT Error: "+str(np.mean(RAawt_score[:,4])))
print("  AWT Error %: "+str(np.mean(RAawt_score[:,5])))
print("Average Error: "+str(np.mean(RAawt_score[:,6])))

print("\n\n---------------------- LA AWT Evaluation")
print("   MSE Masked: "+str(np.mean(LAawt_score[:,1])))
print("    AWT Error: "+str(np.mean(LAawt_score[:,4])))
print("  AWT Error %: "+str(np.mean(LAawt_score[:,5])))
print("Average Error: "+str(np.mean(LAawt_score[:,6])))
