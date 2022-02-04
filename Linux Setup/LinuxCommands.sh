pscp -r "C:\Users\zxio506\Desktop\Atria_Data.zip" zxio506@BN356574.uoa.auckland.ac.nz:/hpc/zxio506
pscp -r zxio506@BN356574.uoa.auckland.ac.nz:/hpc/zxio506/icentia11k/10999_batched_lbls.pkl.gz "C:\Users\Administrator\Desktop"

pscp -r "C:\Users\zxio506\Desktop\Atria_Data" zxio506@BN356574.uoa.auckland.ac.nz:/hpc/zxio506

# UK biobank folder
cd /unifiles/jzha319/TARGET_LOCAL_DIRNAME/imaging_by_participant


##########################################################################
# Titan V Computer
##########################################################################

# putty address to access uoa network IP address: 130.216.208.10 or 130.216.208.20
zxio506@bioeng253.bioeng.auckland.ac.nz
zxio506@bioeng20.bioeng.auckland.ac.nz

jzha319@bioeng20.bioeng.auckland.ac.nz

# ssh into titan V computer
ssh -X zxio506@BN356574.uoa.auckland.ac.nz

### titan v computer commands ###
# activate virtual environment

source /home/zxio506/Virtual_ENV/TitanV/bin/activate
cd /home/zxio506

# when im at uni - directly from titan V local PC to my desktop
pscp -r "C:\Users\zxio506\Desktop\UTAH_MICCAI\UTAH Test set" zxio506@BN356574.uoa.auckland.ac.nz:/home/zxio506/UTAH_MICCAI
pscp -r C:\Users\zxio506\Desktop\CNN.py "zxio506@BN356574.uoa.auckland.ac.nz:/home/zxio506/UTAH_MICCAI"

# run the scripts
nohup python3 /home/zxio506/CNN.py

# grab outputs
pscp -r "zxio506@BN356574.uoa.auckland.ac.nz:/home/zxio506/UTAH_MICCAI/UTAH Test set/log" C:\Users\zxio506\Desktop\

# when im at home - copy to hpc first, then copy to my desktop from hpc
pscp -r C:\Users\Administrator\Desktop\CNN.py zxio506@bioeng253.bioeng.auckland.ac.nz:/hpc/zxio506
scp -r /hpc/zxio506/CNN.py "/home/zxio506/UTAH_MICCAI"





scp -r "/home/zxio506/la_2ch Test/Dataset.h5" /hpc/zxio506
pscp -r "zxio506@bioeng20.bioeng.auckland.ac.nz:/hpc/zxio506/Dataset.h5" C:\Users\Administrator\Desktop\

scp -r "/home/zxio506/UtahWaikato Test Set/Prediction Sample" /hpc/zxio506
pscp -r "zxio506@bioeng20.bioeng.auckland.ac.nz:/hpc/zxio506/Prediction Sample" C:\Users\Administrator\Desktop\





##########################################################################
# ABI HPC
##########################################################################

# putty address to access uoa network IP address: 130.216.208.10 or 130.216.208.20
ssh -X zxio506@bioeng253.bioeng.auckland.ac.nz

# activate virtual environment
source /hpc/zxio506/Virtual_ENV/hpc_virt/bin/activate
source /hpc/zxio506/Virtual_ENV/hpc_virt/bin/activate.csh

# go to my folder
cd /hpc/zxio506

# copy file from windows to HPC, then copy it from hpc to titan V computer*/
pscp -r C:\Users\Administrator\Desktop\3_VNet3D.py zxio506@bioeng253.bioeng.auckland.ac.nz:/hpc/zxio506/LAsegmentation
scp -r /hpc/zxio506/LAsegmentation/3_VNet3D.py /home/zxio506/LAsegmentation/

# copy folder from windows to HPC, then copy it from hpc to titan V computer*/
pscp -r C:\Users\Administrator\Desktop\UTAH_MICCAI zxio506@bioeng253.bioeng.auckland.ac.nz:/hpc/zxio506/LAsegmentation
pscp -r "C:\Users\zxio506\Desktop\Atria_Data.zip" zxio506@bioeng253.bioeng.auckland.ac.nz:/hpc/zxio506
scp -r /hpc/zxio506/LAsegmentation/UTAH_MICCAI/ /home/zxio506/LAsegmentation

# copy folder from HPC to windows
pscp -r zxio506@bioeng253.bioeng.auckland.ac.nz:/home/zxio506/UTAH_MICCAI/UTAH Test set/log/log.txt C:\Users\zxio506\Desktop\

# get all the data from windows to Titan V computer
pscp -r C:\Users\zxio506\Desktop\UTAH_MICCAI zxio506@BN356574.uoa.auckland.ac.nz:/home/zxio506

# set up virtual environment locally (in my folder /home/zxio506)
mkdir Virtual_ENV
cd Virtual_ENV
virtualenv -p /usr/bin/python3.6 Python3virtual
pip install -r requirements.txt

# activate virtual environment
source /home/zxio506/Virtual_ENV/TitanV/bin/activate

# deactivate virtual environment
deactivate

# run file in virtual environment
python3 temp.py

# check GPU usage
nvidia-smi



##########################################################################
# NeSI
##########################################################################

password: fuckNESI2019?

# login sequentially
ssh -Y zxio506@lander02.nesi.org.nz		# first connect to nesi node
ssh -Y login.mahuika.nesi.org.nz		# then into hpc cluster

# first factor is ur password
# second factor is ur google authenticator

# navigate to my directory
cd /nesi/project/nesi00385/zxio506

# keep track of resource use
squeue -u zxio506

# manage jobs
sacct					# view status of all jobs
vi <slurm43425.out>		# view status of specific job (contains error messages)
scancel <JOBID>			# cancel a job

# jumping the node (needed for transferring files from NeSI). setup in ABI hpc
mkdir -p ~/.ssh/sockets
vi ~/.ssh/config # then add the text (big chunk) with ur UPI from "https://nesi.github.io/hpc_training/lessons/maui-and-mahuika/connecting#jumping-across-the-lander-node"
chmod 600 ~/.ssh/config

# now transferring files (executed while on ABI hpc)
scp -r mahuika:/nesi/project/nesi00385/zxio506/zhaohanPart4/Test .
pscp -r zxio506@bioeng253.bioeng.auckland.ac.nz:/hpc/zxio506/Test C:\Users\zxio506\Desktop\


#----------- CPU STUFF -----------
# load library
module load OpenMPI/2.0.2-GCC-6.3.0

# create executable
mpicc /nesi/project/nesi00385/zxio506/zhaohanPart4/FK_3current.c -mcmodel=large -lm -o Test/2DSim.out

# run executable
sbatch job.sl


#----------- GPU STUFF -----------
# tutorial for python
#https://support.nesi.org.nz/hc/en-gb/articles/207782537

# load python 3/other modules etc
module load Python/3.6.3-gimkl-2017a
module load TensorFlow/1.10.1-gimkl-2017a-Python-3.6.3

# install packages
pip install <package name> --user

# run executable
sbatch jobGPU.sl

# run python in terminal (only use CPU, no GPU compute)
python3 example.py




##########################################################################
# General Commands
##########################################################################
# list files
ls -l --color

# create new directory
mkdir temp

# delete directory
rm temp

# rename directory
mv oldname newname

# rename file
cp old.sl new.sl

# view file script
vi file_temp.txt

# exit view mode in "vi" file viewing
:q (save)
:q! (dont save)

# edit file in "vi" file viewing, click esc to exit
i

# save file in "vi" file viewing
:w

# remove all permission
chmod go= <filename>

# enable all permission
chmod 777 <filename>   

# enable all subfolder permission
chmod -R 777 <filename>

# compute memory size of directory
du -h <filename>

# count the number of files in folder
ls <filename> | wc -l

# zip directory
zip -r <output name> <directory name>

##########################################################################
# Windows Powershell Python Commands
##########################################################################

# install python package
py -m pip install <package> --user