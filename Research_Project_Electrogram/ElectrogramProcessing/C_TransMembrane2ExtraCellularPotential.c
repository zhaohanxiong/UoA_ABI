#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define X 		240
#define Y 		240
#define Z 		5

#define N_E  	240


/*void ConvertTransmembraneActionPotential2ExtracellularPotential(float *Vm){
	
	// input: 	pointer to Vm, the action potential (transmembrane potential), shape = [X][Y][Z][T]
	// output:	Phi, the extracellular (unipolar) potential, shape = [N_E][N_E][T]

	float N_E 		= 240.0;		// number of virtual electrodes (N_E x N_E)
	float d 		= 0.2;		// distance of electrodes above cell
	float spacing 	= 0.5;		// discritization space
	float K 		= 0.33;		// ratio of conductivities

	float Vm_fx[X][Y][Z][T];	// gradient of Vm in x direction
	float Vm_fy[X][Y][Z][T];	// gradient of Vm in y direction

	float r_inv[X][Y][Z];		// r, the inverse distance matrix from the cell to the electrodes
	float r_inv_fx[X][Y][Z];	// gradient of r in the x direction
	float r_inv_fy[X][Y][Z];	// gradient of r in the y direction
	
	float Phi[X][Y][T];		// extra cellular action potential
	
	int i,j,k,t,xx,yy;					// counters
	float x_dash,y_dash,z_dash;	// positional values
	
	// calculate gradient of Vm
	for(t = 0; t < T; t++){
		for(k = 0; k < Z; k++){
			for(i = 0; i < X; i++){
				for(j = 1; j < X-1; j++){				
					Vm_fx[i][j][k][t] = (Vm_fx[i+1][j][k][t] - Vm_fx[i-1][j][k][t])/(2*spacing);
					Vm_fy[i][j][k][t] = (Vm_fy[j][i+1][k][t] - Vm_fy[j][i-1][k][t])/(2*spacing);
				}
				Vm_fx[0][i][k][t] = (Vm_fx[1][i][k][t] - Vm_fx[0][i][k][t])/spacing;
				Vm_fy[i][0][k][t] = (Vm_fy[i][1][k][t] - Vm_fy[i][0][k][t])/spacing;
				Vm_fx[X][i][k][t] = (Vm_fx[X][j][k][t] - Vm_fx[X-1][i][k][t])/spacing;
				Vm_fy[i][Y][k][t] = (Vm_fy[i][Y][k][t] - Vm_fy[i][Y-1][k][t])/spacing;
			}
		}
	}

	for(xx = 0; xx < N_E; xx++){
		for(yy = 0; yy < N_E; yy++){
			
			// x,y,z positions of virtual electrodes
			x_dash = spacing*(X/N_E/2.0 + X/N_E*xx);
			y_dash = spacing*(Y/N_E/2.0 + Y/N_E*yy);
			z_dash = spacing*(Z + d);
			
			// calculate inverse distance matrix
			for(i = 0; i < X; i++){
				for(j = 0; j < Y; j++){
					for(k = 0; k < Z; k++){			
						r_inv[i][j][k] = 1/sqrt(pow(x_dash-i*spacing,2)+pow(y_dash-j*spacing,2)+pow(z_dash-k*spacing,2)); 
					}
				}
			}
			
			// calculate gradient of r
			for(k = 0; k < Z; k++){
				for(i = 0; i < X; i++){
					for(j = 1; j < X-1; j++){
						r_inv_fx[j][i][k] = (r_inv[j+1][i][k] - r_inv[j-1][i][k])/(2*spacing);
						r_inv_fy[i][j][k] = (r_inv[i][j+1][k] - r_inv[i][j-1][k])/(2*spacing);
					}
					r_inv_fx[0][i][k] = (r_inv[1][i][k] - r_inv[0][i][k])/spacing;
					r_inv_fy[i][0][k] = (r_inv[i][1][k] - r_inv[i][0][k])/spacing;
					r_inv_fx[X][i][k] = (r_inv[X][i][k] - r_inv[X-1][i][k])/spacing;
					r_inv_fy[i][Y][k] = (r_inv[i][Y][k] - r_inv[i][Y-1][k])/spacing;
				}
			}
			
			// calculate Phi, perform dot product then 3D integral (sum over X/Y/Z)
			for(t = 0; t < T; t++){
				Phi[xx][yy][t] = 0.0;
				for(i = 0; i < X; i++){
					for(j = 0; j < Y; j++){
						for(k = 0; k < Z; k++){
							Phi[xx][yy][t] += Vm_fx[i][j][k][t]*r_inv_fx[i][j][k] + Vm_fy[i][j][k][t]*r_inv_fy[i][j][k];
						}
					}
				}
			}

		}
	}
	
}*/

int main(int argc, char **argv){
	
	float K 		= 0.0033;// ratio of conductivities
	float spacing 	= 0.5;	 // discritization space
	
	float r_inv[X][Y][Z];		// r, the inverse distance matrix from the cell to the electrodes
	float r_inv_fx[X][Y][Z];	// gradient of r in the x direction
	float r_inv_fy[X][Y][Z];	// gradient of r in the y direction
	
	float Vm[X][Y][Z];		// action potential
	float Vm_fx[X][Y][Z];	// gradient of Vm in x direction
	float Vm_fy[X][Y][Z];	// gradient of Vm in y direction
	
	float Phi[N_E][N_E];	// extra cellular action potential
	
	float T,temp;
	int x,y,z,t,i,j,k;
	float x_dash,y_dash,z_dash,d;
	
	FILE *out,*in;
	char *str,*str1;
	
	in = fopen("AP/Vm1.dat","r");
	fscanf(in,"%f ",&T);
	
	// write Phi for each time step
	for(t = 0; t < T; t++){
		
		// read file
		for(z = 0; z < Z; z++){
			for(y = 0; y < Y; y++){
				for(x = 0; x < X; x++){
					fscanf(in,"%f ",&temp);
					Vm[x][y][z] = temp;
				}
			}
		}
		
		// calculate gradient of Vm
		for(k = 0; k < Z; k++){
			for(i = 0; i < X; i++){
				for(j = 1; j < X-1; j++){
					Vm_fx[j][i][k] = (Vm[j+1][i][k] - Vm[j-1][i][k])/(2*spacing);
					Vm_fy[i][j][k] = (Vm[i][j+1][k] - Vm[i][j-1][k])/(2*spacing);
				}
				Vm_fx[0][i][k] = (Vm[1][i][k] - Vm[0][i][k])/spacing;
				Vm_fy[i][0][k] = (Vm[i][1][k] - Vm[i][0][k])/spacing;
				Vm_fx[X-1][i][k] = (Vm[X-1][i][k] - Vm[X-2][i][k])/spacing;
				Vm_fy[i][Y-1][k] = (Vm[i][Y-1][k] - Vm[i][Y-2][k])/spacing;
			}
		}

		// Save Phi
		str = malloc(40*sizeof(char));
		sprintf(str,"Test/pp%04d.vtk",t);
		out = fopen(str,"wt");
		
		// calculate Phi, perform dot product then 3D integral (sum over X/Y/Z)
		for(x = 0; x < N_E; x++){
			for(y = 0; y < N_E; y++){

				// x,y,z,d positions of virtual electrodes
				d = 0.2; // distance of electrodes above cell
				x_dash = spacing*(X/N_E/2.0 + X/N_E*x);
				y_dash = spacing*(Y/N_E/2.0 + Y/N_E*y);
				z_dash = spacing*(Z + d);
				
				// calculate inverse distance matrix
				for(i = 0; i < X; i++){
					for(j = 0; j < Y; j++){
						for(k = 0; k < Z; k++){
							r_inv[i][j][k] = 1/sqrt(pow(x_dash-i*spacing,2)+pow(y_dash-j*spacing,2)+pow(z_dash-k*spacing,2)); 
						}
					}
				}

				// calculate gradient of r
				for(k = 0; k < Z; k++){
					for(i = 0; i < X; i++){
						for(j = 1; j < X-1; j++){
							r_inv_fx[j][i][k] = (r_inv[j+1][i][k] - r_inv[j-1][i][k])/(2*spacing);
							r_inv_fy[i][j][k] = (r_inv[i][j+1][k] - r_inv[i][j-1][k])/(2*spacing);
						}
						r_inv_fx[0][i][k] = (r_inv[1][i][k] - r_inv[0][i][k])/spacing;
						r_inv_fy[i][0][k] = (r_inv[i][1][k] - r_inv[i][0][k])/spacing;
						r_inv_fx[X-1][i][k] = (r_inv[X-1][i][k] - r_inv[X-2][i][k])/spacing;
						r_inv_fy[i][Y-1][k] = (r_inv[i][Y-1][k] - r_inv[i][Y-2][k])/spacing;
					}
				}
				
				// calculate phi
				Phi[x][y] = 0.0;
				for(i = 0; i < X; i++){
					for(j = 0; j < Y; j++){
						for(k = 0; k < Z; k++){
							Phi[x][y] += Vm_fx[i][j][k]*r_inv_fx[i][j][k] + Vm_fy[i][j][k]*r_inv_fy[i][j][k];
						}
					}
				}
				Phi[x][y] = Phi[x][y]*-1.0;
				fprintf(out,"%3.1f ",Phi[x][y]);
			}
			fprintf(out,"\n");
		}
		fclose(out);
		free(str);
		
	}
	
	fclose(in);
	
	return 0;
}