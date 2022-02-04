import numpy as np
import wfdb
import scipy.io
import matplotlib.pyplot as plt

i = 5
name = "Copy of compare_data_0"+str(i)+"\ECGRec_202003_C839524_1590542073855-1590897133847"

dat = wfdb.rdsamp(name)
scipy.io.savemat("ecg"+str(i)+".mat",mdict={"ecg":np.array(dat[0])[:,0]})

f = open("info"+str(i)+".txt","w")
f.write(str(dat[1]))
f.close()
