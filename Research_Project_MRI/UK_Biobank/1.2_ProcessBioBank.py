from utils import *

os.chdir("/unifiles/jzha319/TARGET_LOCAL_DIRNAME/imaging_by_participant")

# create folder to save files into
create_folder("UK_Biobank_Labelled")

# list all group files
data_files = os.listdir()
data_files.remove("identifier_conflicts")
data_files.remove("Healthy_804_feid.csv")
data_files.remove("inventory_heart_MRI_plus_values_r9a.csv")
data_files.remove("index_and_data_extract_unprocessed_r9a.csv")

# search through each group file
for sub_path in sorted(data_files):
	
	# For each subject in the  group file
	for eid in sorted(os.listdir(sub_path)):
	
		data_dir = os.path.join(sub_path, eid)

		# Only convert data if there is manual annotation, i.e. cvi42 files, or if the data has not been converted, or if its not an error file
		if os.path.exists(os.path.join(data_dir, '{0}_cvi42.zip'.format(eid))) and not eid in os.listdir("UK_Biobank_Labelled") and not eid in ["3291007","5354483"]:

			# Decompress the zip files in this directory
			files = glob.glob('{0}/{1}_*.zip'.format(data_dir, eid))
			dicom_dir = os.path.join(data_dir, 'dicom')
			
			# create directory for dicom dictory
			create_folder(dicom_dir)

			for f in files:
			
				if os.path.basename(f) == '{0}_cvi42.zip'.format(eid):
				
					# extract contents of zip file to folder
					#zipfile.ZipFile(f, 'r').extractall(data_dir)         # FIX!!! catch errors with the zip file being corrupt
					os.system('unzip -o {0} -d {1}'.format(f, data_dir))  # only works on linux
					
				else:
					
					# extract contents of zip file to folder
					#zipfile.ZipFile(f, 'r').extractall(dicom_dir)        # FIX!!! catch errors with the zip file being corrupt
					os.system('unzip -o {0} -d {1}'.format(f, dicom_dir)) # only works on linux
					
					# Process the manifest file
					process_manifest(os.path.join(dicom_dir, 'manifest.csv'),os.path.join(dicom_dir, 'manifest2.csv'))
					df2 = pd.read_csv(os.path.join(dicom_dir, 'manifest2.csv'), error_bad_lines=False)

					# Organise the dicom files, group the files into subdirectories for each imaging series
					for series_name, series_df in df2.groupby('series discription'):
						series_dir = os.path.join(dicom_dir, series_name)
						create_folder(series_dir)
						series_files = [os.path.join(dicom_dir, x) for x in series_df['filename']]
						#for ff in series_files:
						#	shutil.move(ff, series_dir)           # FIX!!! if theres files already there
						os.system('mv {0} {1}'.format(' '.join(series_files), series_dir)) # only works on linux

			# Parse cvi42 xml file
			cvi42_contours_dir = os.path.join(data_dir, 'cvi42_contours')
			create_folder(cvi42_contours_dir)
			xml_name = os.path.join(data_dir, '{0}_cvi42.cvi42wsx'.format(eid))
			txt_name = os.path.join(data_dir, '{0}_cvi42.txt'.format(eid))
			parseFile(xml_name, cvi42_contours_dir)

			# Rare cases when no dicom file exists
			# e.g. 12xxxxx/1270299
			if not os.listdir(dicom_dir):
				print('Warning: empty dicom directory! Skip this one.')
			else:

				# Convert dicom files and annotations into nifti images, and save them as nifti files
				dset = Biobank_Dataset(dicom_dir, cvi42_contours_dir)
				dset.read_dicom_images()
				dset.convert_dicom_to_nifti(data_dir)

				# Remove intermediate dicom files
				shutil.rmtree(dicom_dir)
				shutil.rmtree(cvi42_contours_dir)
				os.remove(xml_name)
				os.remove(txt_name)
				
				# create new folder for saving data outputs
				create_folder(os.path.join("UK_Biobank_Labelled",eid))
				
				# move extracted files to new folder
				for ff in [s for s in os.listdir(data_dir) if "nii.gz" in s]:
					move_folder(os.path.join(data_dir,ff), os.path.join("UK_Biobank_Labelled",eid))