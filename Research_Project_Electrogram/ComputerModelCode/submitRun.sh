#remember leading underscoreS
type="_Thin"
output="fa"
stim=0
if [ $stim -eq 1 ]; then
	filename="MPI_CRN_2DSAN_17April_2018_Ellipse_Stim.c"
	outLNum=1590
else
	outLNum=1582
	filename="MPI_CRN_2DSAN_17April_2018_Ellipse.c"
fi
sed -i "634 c        in = fopen (\"SAN_Ellipse$type\_RoundedEntry_Apr18.dat\", \"r\");" $filename
sed -i "697 c        in = fopen (\"SAN_Ellipse_Fibres$type\_RoundedEntry_Apr18.dat\",\"r\");" $filename
sed -i "$outLNum c        sprintf (str, \"Test9Regions/$output%05d.vtk\", cnt); //output" $filename
mpicc $filename -mcmodel=large -lm -o $output.out
sed -i "12 c\srun /projects/uoa00193/SAN/EllipseModel/$output.out" SANEllip.sl
sbatch SANEllip.sl
echo $output