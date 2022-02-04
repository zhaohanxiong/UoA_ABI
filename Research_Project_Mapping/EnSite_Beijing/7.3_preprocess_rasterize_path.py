from utils import *

# set directories
os.chdir("C:/Users/zxio506/Desktop/beijing utah concave hull projected/beijing utah path")
utah_path = "C:/Users/zxio506/Desktop/Atria_Data/Utah"

# define individual patients for utah
utah_patients = []
for s in os.listdir(utah_path):
	pat = os.listdir(utah_path+"/"+s)
	for s2 in pat:
		temp = s+"/"+s2
		temp = temp.replace("CARMA","")
		temp = temp.replace("/","")
		utah_patients.append(temp)
utah_patients = np.array(utah_patients)
utah_train,utah_test = utah_patients[:124],utah_patients[124:]

# define files
files = os.listdir()
files = [s for s in files if ".mat" in s]

# define output files
train_dat,train_lab,test_dat,test_lab = [],[],[],[]

# loop through all beijing utah path files
count = 1
for f in range(len(files)): # sample randomly somehow to decrease number of samples

	# print(str(f+1)+" - "+files[f])

	# load 3D LA
	f_mri = files[f].split("_")[1]
	f_mri = f_mri.replace(".mat","")
	
	# laendo = load_nrrd(utah_path+"/CARMA"+f_mri[:4]+"/"+f_mri[4:]+"/laendo.nrrd")
	# laendo = np.rollaxis(laendo,0,3)
	# laendo[laendo>1] = 1

	# # load path data
	# dat = scipy.io.loadmat(files[f])["path_interpolated"]
	# dat[:,2] = dat[:,2] - 1
	
	# # rasterize path
	# mask = np.zeros_like(laendo)
	
	# for i in range(dat.shape[0]):
		# mask[int(dat[i,0]),int(dat[i,1]),int(dat[i,2])] += 1

	# # invert values to enhance boundary values
	# val             = np.max(mask) + 1
	# mask            = val - mask
	# mask[mask==val] = 0
	
	# # normalize mask to 0-1
	# mask = mask/np.max(mask)
	
	# # crop images to size
	# com     = center_of_mass(laendo)
	# n11,n12 = int(com[0] - 128//2),int(com[0] + 128//2)
	# n21,n22 = int(com[1] - 208//2),int(com[1] + 208//2)
	
	# laendo = laendo[n11:n12,n21:n22,:]
	# mask   = mask[n11:n12,n21:n22,:]

	# # add feature map dimension to input and output
	# label          = np.zeros([laendo.shape[0],laendo.shape[1],laendo.shape[2],2])
	# label[:,:,:,0] = 1 - laendo
	# label[:,:,:,1] = laendo
	
	# mask           = mask[:,:,:,None]
	
	# append to train/test
	if np.any(f_mri == utah_train) and random.choice([True,False]):
		#train_dat.append(mask)
		#train_lab.append(label)
		x = 1
	else:
		#test_dat.append(mask)
		#test_lab.append(label)
		
		print(str(count) + " - " + files[f])
		count = count + 1

train_dat = np.array(train_dat)
train_lab = np.array(train_lab)
test_dat  = np.array(test_dat)
test_lab  = np.array(test_lab)

# save data
create_folder("Catheter Path Test Set")
create_folder("Catheter Path Test Set/log")
create_folder("Catheter Path Test Set/Prediction Sample")

h5f = h5py.File('Catheter Path Test Set/beijing utah path.h5','w')
h5f.create_dataset("train.data",  data=train_dat)
h5f.create_dataset("train.label", data=train_lab)
h5f.create_dataset("test.data",   data=test_dat)
h5f.create_dataset("test.label",  data=test_lab)
h5f.close()