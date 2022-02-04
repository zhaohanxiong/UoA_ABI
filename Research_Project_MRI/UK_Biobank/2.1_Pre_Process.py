from utils import *

nxy              = 144   # image size (x by y)
train_test_split = 0.87  # tran-test-split
d                = 1     # which data to extract (0 = LA 2 chamber, 1 = LA 4 chamber, 2 = Ventricle)

### INITIALIZATION -------------------------------------------------------------------------------------------------------------------------------------------
if d == 1:
	n1,n2 = 80,96
elif d == 0:
	n1,n2 = 64,80
	
#os.chdir("C:/Users/Administrator/Desktop/UK_Biobank_Labelled")
os.chdir("/hpc/zxio506/UK_Biobank_Labelled")

# define the data types in UK Biobank with labels
data_types = ["la_2ch", "la_4ch", "sa"]

# create an output folder for the images if it wasnt already created
create_folder(data_types[d]+" Test")
create_folder(data_types[d]+" Test/log")
create_folder(data_types[d]+" Test/Prediction Sample")

# list all the files in dataset, and filter and arrange names
files = os.listdir()

remove_list = []
for f in files:
	if f in [s+" Test" for s in data_types]:
		remove_list.append(f)
for f in remove_list:
	files.remove(f)

files = sorted(files)

### DATA EXTRACTION ------------------------------------------------------------------------------------------------------------------------------------------

# initialise the image and label arrays
Image,Image_Test,Label,Label_Test = [],[],[],[]

# loop through all training patients
for i in range(len(files)):

	print(str(i+1)+" Processing: "+files[i])
	
	# only process if data exists in patient folder
	if "label_"+data_types[d]+".nii.gz" in os.listdir(files[i]):
		
		# read in MRI/annotation data and convert to integer 8 numpy array
		mri = nifti_to_array(os.path.join(files[i],data_types[d]+".nii.gz"))
		lab = nifti_to_array(os.path.join(files[i],"label_"+data_types[d]+".nii.gz"),is_label=True)

		# check data shapes to make sure they match
		assert np.all(mri.shape == lab.shape), 			"Error! Image and Label Shape Don't Match for Patient "+files[i]
		assert len(np.unique(lab)) <= np.max(lab) + 1, 	"Error! Label has Incorrect Values Annotated for Patient "+files[i]
		
		# ventricles is 4D data
		if data_types[d] == "sa":
			
			# flatten the data out into 3D format, with 3rd axis as slices
			temp1,temp2 = [],[]
			for n1 in range(mri.shape[2]):
				for n2 in range(mri.shape[3]):
					temp1.append(mri[:,:,n1,n2])
					temp2.append(lab[:,:,n1,n2])
			mri = np.rollaxis(np.array(temp1),0,3)
			lab = np.rollaxis(np.array(temp2),0,3)
			
		# pad data if too small in x/y directions
		if mri.shape[0] < nxy:
			pad_width = int(np.ceil((nxy - mri.shape[0])/2))
			mri       = np.pad(mri,((pad_width,pad_width),(0,0),(0,0)),"constant",constant_values=0)
			lab       = np.pad(lab,((pad_width,pad_width),(0,0),(0,0)),"constant",constant_values=0)
		if mri.shape[1] < nxy:
			pad_width = int(np.ceil((nxy - mri.shape[1])/2))
			mri       = np.pad(mri,((0,0),(pad_width,pad_width),(0,0)),"constant",constant_values=0)
			lab       = np.pad(lab,((0,0),(pad_width,pad_width),(0,0)),"constant",constant_values=0)
			
		# compute which slice is annotated
		n_lab = np.unique(np.where(lab > 0)[2])
		
		# loop through all slices with labels
		for n in n_lab:
			
			# extract individual slices
			temp_mri = mri[:,:,n]
			temp_lab = lab[:,:,n]
			
			# find midpoint of label
			midpoint= np.array(ndimage.measurements.center_of_mass(temp_lab > 0)).astype(np.int32)
			
			# extract the patches from the midpoint
			n11,n12 = int(midpoint[0] - n1//2),int(midpoint[0] + n1//2)
			n21,n22 = int(midpoint[1] - n2//2),int(midpoint[1] + n2//2)

			# if the number of files is less than train-test-split, then data is in training set, otherwise its in testing set
			if i <= int(train_test_split*len(files)):
				Image.append(temp_mri[n11:n12,n21:n22])
				Label.append(temp_lab[n11:n12,n21:n22])
			else:
				Image_Test.append(temp_mri[n11:n12,n21:n22])
				Label_Test.append(temp_lab[n11:n12,n21:n22])
			
Image,Image_Test,Label,Label_Test = np.array(Image),np.array(Image_Test),np.array(Label),np.array(Label_Test)

# for 2ch, filter out labels with value of 2
if data_types[d] == "la_2ch":
	Label[Label > 1] = 1

# for training data encoding label to neural network output format
temp = np.empty(shape=[Label.shape[0],nxy,nxy,len(np.unique(Label))])
for i in range(len(np.unique(Label))):	
	x = Label == i
	temp[:,:,:,i] = x

Image,Label = np.reshape(Image,newshape=[-1,n1,n2,1]),np.reshape(temp,newshape=[-1,n1,n2,len(np.unique(Label))])

# create a HDF5 dataset
h5f = h5py.File(data_types[d]+' Test/Dataset.h5','w')
h5f.create_dataset("train",		data=Image)
h5f.create_dataset("train_lab",	data=Label)
h5f.create_dataset("test",		data=Image_Test)
h5f.create_dataset("test_lab",	data=Label_Test)
h5f.close()