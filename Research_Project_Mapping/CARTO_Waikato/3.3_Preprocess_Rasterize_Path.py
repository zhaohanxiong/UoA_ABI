from utils import *

# patient folders
waikato_patient = ["00003","00004","00005","000011"]

# patient folders
lgemri_path = "C:/Users/zxio506/Desktop/Atria_Data/Waikato/"
mesh_path   = "C:/Users/zxio506/Desktop/beijing utah concave hull projected/Waikato Data/"

# define output files
test_dat,test_lab = [],[]

for n in range(len(waikato_patient)):

	print("Processing "+waikato_patient[n]) 
	
	# load 3D LA
	laendo = np.zeros([640,640,44])
	f      = os.listdir(lgemri_path+waikato_patient[n]+"/label")

	for i in range(laendo.shape[2]):
		laendo[:,:,i] = cv2.imread(lgemri_path+waikato_patient[n]+"/label/"+f[i],cv2.IMREAD_GRAYSCALE)
		
	laendo = (laendo==4).astype(np.int8)

	# load path data
	dat = scipy.io.loadmat(mesh_path+waikato_patient[n]+"_projected_path.mat")["path_interpolated"]
	dat[:,2] = dat[:,2] - 1

	# rasterize path
	mask = np.zeros_like(laendo)

	for i in range(dat.shape[0]):
		mask[int(dat[i,0]),int(dat[i,1]),int(dat[i,2])] += 1

	# invert values to enhance boundary values
	val             = np.max(mask) + 1
	mask            = val - mask
	mask[mask==val] = 0

	# normalize mask to 0-1
	mask = mask/np.max(mask)

	# crop images to size
	com     = center_of_mass(laendo)
	n11,n12 = int(com[0] - 128//2),int(com[0] + 128//2)
	n21,n22 = int(com[1] - 208//2),int(com[1] + 208//2)

	laendo = laendo[n11:n12,n21:n22,:]
	mask   = mask[n11:n12,n21:n22,:]

	# add feature map dimension to input and output
	label          = np.zeros([laendo.shape[0],laendo.shape[1],laendo.shape[2],2])
	label[:,:,:,0] = 1 - laendo
	label[:,:,:,1] = laendo

	mask = mask[:,:,:,None]
	
	# append to test
	test_dat.append(mask)
	test_lab.append(label)

test_dat,train_lab = np.array(test_dat),np.array(test_lab)

# save data
h5f = h5py.File(mesh_path+'waikato path.h5','w')
h5f.create_dataset("test.data",   data=test_dat)
h5f.create_dataset("test.label",  data=test_lab)
h5f.close()

# write files to image stacks
if True:

	os.mkdir(mesh_path+"Image Stack")
	
	# loop through all files
	for n in range(test_dat.shape[0]):
		
		print("Writing "+waikato_patient[n])

		# write to image stack
		os.mkdir(mesh_path+"Image Stack/"+waikato_patient[n])
		os.mkdir(mesh_path+"Image Stack/"+waikato_patient[n]+"/input")
		os.mkdir(mesh_path+"Image Stack/"+waikato_patient[n]+"/true")
		
		for i in range(test_dat[n].shape[2]):
			img_name = "{0:03}".format(i+1)+".tif"
			cv2.imwrite(mesh_path+"Image Stack/"+waikato_patient[n]+"/input/"+img_name,test_dat[n,:,:,i,0]*255)
			cv2.imwrite(mesh_path+"Image Stack/"+waikato_patient[n]+ "/true/"+img_name,np.argmax(train_lab[n,:,:,i,:],2))
