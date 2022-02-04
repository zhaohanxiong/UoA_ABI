##########################################################################
# Windows commands
# (do everything in command window on windows computer)
##########################################################################
/*send files from windows PC to Nesi*/
pscp C:\Users\zxio506\OneDrive\Part4\part4project\ComputerModelCode\FK_3current.c zxio506@login.uoa.nesi.org.nz:/projects/nesi00385/zhaohanPart4/

/*send folder from windows PC to Nesi*/
pscp -r C:\Users\zxio506\OneDrive\Part4\part4project\ComputerModelCode\LoadRotor\StableRotor500x500 zxio506@login.uoa.nesi.org.nz:/projects/nesi00385/zhaohanPart4/

/*copy files from Nesi to windows PC*/
pscp zxio506@login.uoa.nesi.org.nz:/projects/nesi00385/zhaohanPart4/Test/2DSim.out C:\Users\zxio506\Desktop\

#copy folder from Nesi to windows PC*/
pscp -r zxio506@login.uoa.nesi.org.nz:/projects/nesi00385/zhaohanPart4/Test C:\Users\zxio506\Desktop\

#copy folder from windows to ABI HPC PC*/
pscp -r C:\Users\zxio506\Desktop\ECP_Final.zip zxio506@bioeng253.bioeng.auckland.ac.nz:/hpc/zxio506/

pscp -r C:\Users\zxio506\Desktop\UTAH_MICCAI.zip zxio506@bioeng253.bioeng.auckland.ac.nz:/hpc/zxio506/

##########################################################################
# NESI commands zhaohan
##########################################################################

#log on to nesi zhaohan*/
ssh -X zxio506@login.uoa.nesi.org.nz

#create the shell environment*/
ssh build-sb

#go to project files*/
cd /projects/nesi00385/zhaohanPart4

#copy files from HPC to nesi*/
scp zxio506@bioeng253.bioeng.auckland.ac.nz:/hpc_utoz/zxio506/ZhaohanPart4/Atrium2DSim_25_6_2018.c .

#load packages*/
module load OpenMPI/1.6.5-GCC-4.8.2

#create executable*/
mpicc /projects/nesi00385/zhaohanPart4/FK_3current.c -mcmodel=large -lm -o ./Test/2DSim.out

#run job*/
sbatch job.sl

#check job status*/
squ

#cancel job (find the JOBID first) then*/
scancel JOBID

##########################################################################
# general commands
##########################################################################
/*open a new tab in terminal*/
CTRL+SHIFT+T

/*get current directory*/
pwd

/*list files colored*/
ls -l --color

/*list the with sl extension on the end*/
ls *.sl

/*create directory*/
mkdir /hpc_utoz/zxio506/folder

/*delete file*/
rm /hpc_utoz/zxio506/folder

/*delete directory*/
rm -rf /hpc_utoz/zxio506/folder/Test

/*rename a file/folder*/
cp SAN3D.sl job.sl

/*copy folders*/
cp -a /hpc_htom/jzha319/ZhaohanPart4/hpc_utoz/zxio506

/*rename folder*/
mv oldname newname

/*view file script*/
vi file_temp.txt

/*exit view mode in "vi" file viewing*/
:q (save)
:q! (dont save)

/*edit file in "vi" file viewing, click esc to exit*/
i

/*save file in "vi" file viewing*/
:w

/*run matlab*/
matlab file.m

/*run python3*/
python3 file.py



##########################################################################
# NESI commands from Jichao
# https://wiki.auckland.ac.nz/display/CER/Slurm+User+Guide
##########################################################################

/*login to NeSi*/
zxio506@bn364363:/hpc_htom/jzha319$ ssh -X jzha319@login.uoa.nesi.org.nz

/*current working directory*/
[jzha319@login-01 ~]$ pwd

/*our project files nesi00385 or uoa00193*/
[jzha319@login-01 projects]$ cd nesi00385

/*copy files from local to Nesi*/
scp jzha319@bioeng253.bioeng.auckland.ac.nz:/hpc/jzha319/Vadim/Oct31SAN_Modelling2017/2013-31_10HumanHeart/ComputerModel/SANModel30um3110/MPI_SAN_FK_JZ_30June_2017.c .
scp -r jzha319@bioeng253.bioeng.auckland.ac.nz:/hpc/jzha319/2015-05-18_MRI_Heart15_MicroCTed/Heart15WholeAtriaModelling/WholeAtriaDownsizedJournal/Figure6Inducibility/Mid3RedPacing_RerunShort.c .

/*copy files from nesi to local*/
scp jzha319@login.uoa.nesi.org.nz:/projects/uoa00193/H15Modelling/slurm-28969405.out .

/*compile run*/
ssh build-sb
module load OpenMPI/1.6.5-GCC-4.8.2
sbatch Mid3V2pacing.sl
squ

##########################################################################
# hpc commands
###############################################e##########################

/*log in to HPC from virtual box*/
ssh -X zxio506@bioeng253.bioeng.auckland.ac.nz
ssh hpc3

/*access hpc and go to hpc my folder*/
cd /hpc_utoz/zxio506/

/*get file from nesi*/
scp zxio506@login.uoa.nesi.org.nz:/projects/nesi00385/zhaohanPart4/Atrium2DSim_25_6_2018.c .

/*get a folder or directory from nesi*/
scp -r zxio506@login.uoa.nesi.org.nz:/projects/nesi00385/zhaohanPart4/Test/ .

/*compile and run C code*/
/hpc/jzha319/mpich2Local/bin/mpicc Atrium2DSim_25_6_2018.c -mcmodel=large -lm
/hpc/jzha319/mpich2Local/bin/mpiexec -n 20 ./2DSim.out

/*view current processes on HPC, quit and kill with Proejct ID*/
top
q
kill [PID]