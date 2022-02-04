import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import gzip

os.chdir("C:/Users/Administrator/Desktop")
os.chdir("G:/icentia11k")

# load data from one patient
f_str = "00020"
f_dat = gzip.open(f_str+"_batched.pkl.gz","rb")
f_lab = gzip.open(f_str+"_batched_lbls.pkl.gz","rb")

#dat = pickle.load(open("12147"+"_batched.pkl","rb"))
#lab = pickle.load(open("12147"+"_batched_lbls.pkl","rb"))
dat = pickle.load(f_dat)
lab = pickle.load(f_lab)


# print data dimensions
print("\n")
print("The ECG data has a shape of ",str(dat.shape))
print("The Label data has a list length of ",str(len(lab))," with dictionary keys ",str(lab[0].keys()))
print("-"*100)
print("Label data with key \"btype\"(Beat Type) has a list length of ",str(len(lab[0]["btype"])))
for i in range(len(lab[0]["btype"])):
      print("\tElement "+str(i)+" has shape "+str(lab[0]["btype"][i].shape))
print("\n")
print("Label data with key \"rtype\"(Rhythm Type) has a list length of ",str(len(lab[0]["rtype"])))
for i in range(len(lab[0]["rtype"])):
      print("\tElement "+str(i)+" has shape "+str(lab[0]["rtype"][i].shape))

# visualize data
#plt.plot(dat[24,990000:1000000]);plt.show()

