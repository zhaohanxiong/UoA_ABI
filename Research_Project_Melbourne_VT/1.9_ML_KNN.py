import os
import tflearn
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split

os.chdir("C:/Users/zxio506/Desktop/VT Data New Feature")

files = os.listdir()

# training data
X_train, Y_train = pd.DataFrame([]), pd.DataFrame([])
X_test, Y_test = [],[]

# set up training set
for i in range(len(files)):

	# load Data 
	# columns: 'X.Position.Coordinate', 'Y.Position.Coordinate', 'Z.Position.Coordination', 
	#				'Alpha.Angular.Coordinate', 'Beta.Angular.Coordinate', 'Gamma.Angular.Coordinate', 
	#              'Unipolar.Value', 'Bipolar.Value', 'Wavefront.Slope.Value', 'Unipolar.Voltage.Local', 
	#              'Bipolar.Voltage.Local', 'LAT.Earliest', 'LAT.Peak', 'LAT.Latest', 'LAT.dVdT', 'EGM.Duration', 
	#              'DP.TAG', 'FS.TAG', 'LAVA.TAG', 'LP.TAG', 'FLP.TAG', '5.mm.IN', '10.mm.IN', 'xy', 'xz', 'yz'
	
	dat = pd.read_csv(files[i])

	# split into data label
	X = dat.drop([#'X.Position.Coordinate', 'Y.Position.Coordinate', 'Z.Position.Coordination', 
                        'Alpha.Angular.Coordinate', 'Beta.Angular.Coordinate', 'Gamma.Angular.Coordinate',
			#'Unipolar.Value', 'Bipolar.Value',
			#'Unipolar.Voltage.Local',
                        'Bipolar.Voltage.Local',
			'Wavefront.Slope.Value', 'EGM.Duration', 
			'LAT.Earliest', 'LAT.Peak', 'LAT.Latest', 'LAT.dVdT',
                        #'FS.TAG', 'LP.TAG',
                        'LAVA.TAG', 'FLP.TAG', 'DP.TAG',
			'xy','xz','yz',
			'5.mm.IN',
                        '10.mm.IN',
			], 1)
	#Y = dat["'5.mm.IN"]
	Y = dat["10.mm.IN"]

	# append
	if "train" in files[i]:
		X_train = pd.concat([X_train, X])
		Y_train = pd.concat([Y_train, Y])
	elif "test" in files[i]:
		X_test.append(X)
		Y_test.append(Y)
	else:
		assert False, "ERROR DATA"

# create model
clf = neighbors.KNeighborsClassifier(n_neighbors=50,weights="distance") # distance uniform
clf.fit(X_train, np.array(Y_train)[:,0])

# evaluate
if True: # True False

	f1, sensi, speci = [],[],[]
	
	for i in range(len(X_test)):
		
		# predict for test set
		#pred = clf.predict(X_test[i])
		#t,p = np.array(Y_test[i])==1,pred==1

		pred = clf.predict_proba(X_test[i])[:,1]
		t,p = np.array(Y_test[i])==1,pred>0.0

		tp,fp = np.sum((p == 1) * (p == t)),np.sum((p == 1) * (p != t))
		fn,tn = np.sum((p == 0) * (p != t)),np.sum((p == 0) * (p == t))

		f1_score    = 2 * np.sum(p * t) / (np.sum(p) + np.sum(t))
		sensitivity = tp / (tp + fn)
		specificity = tn / (tn + fp)

		f1.append(f1_score)
		sensi.append(sensitivity)
		speci.append(specificity)
	
	f1, sensi, speci = np.array(f1),np.array(sensi),np.array(speci)
	
	print("------------------------------- Individual Sensitivities")
	print(np.round(f1*100,2))
	print("\n ---------------------------- Overall Evaluation Metrics")
	print("F1 Score = " + str(np.round(np.mean(f1*100),2)))
	print("Sensitivity = " + str(np.round(np.mean(sensi*100),2)))
	print("Specificity = " + str(np.round(np.mean(speci*100),2)))
