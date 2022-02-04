#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define BLOCK_LOW(id,p,n) ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id+1),p,n)-1)
#define BLOCK_SIZE(id,p,n) (BLOCK_LOW((id+1),p,n)-BLOCK_LOW(id,p,n))

#define X    240 // 240
#define Y    240 // 240
#define Z    5
#define N_E  8

#define D1   0.025 //0.3//1.2//3.3 
#define D2   0.025 //0.06//1.2//0.33
#define DD   0.0 //0.24//2.97   

#define dt   0.0025 //0.0025   // ms
#define dx   0.15 //0.04       // mm

#define R     8314
#define T     308
#define F     96487
#define RTOnF (T*0.08554)

// 4-current FK model solver
float global_u[X+1][Y+1][Z+1];
float global_v[X+1][Y+1][Z+1];
float global_w[X+1][Y+1][Z+1];
float global_s[X + 1][Y + 1][Z + 1];
float new_u[X+1][Y+1][Z+1];

float xx[X+1][Y+1][Z+1];
float yy[X+1][Y+1][Z+1];
float zz[X+1][Y+1][Z+1];  

int g[X+1][Y+1][Z+1];
int h[X+1][Y+1][Z+1];

float dc[X+1][Y+1][Z+1][10];
float df[X+1][Y+1][Z+1][10];

// Gating variable
float u[X + 1][Y + 1][Z + 1];
float v[X + 1][Y + 1][Z + 1];
float w[X + 1][Y + 1][Z + 1];
float s[X + 1][Y + 1][Z + 1];

// 4-current FK model paramters
float tau_v1m 	= 16.3;
float tau_v2m 	= 1150;
float tau_vp 	= 1.703;
float tau_w1m 	= 79.963;	// change AP shape
float tau_w2m 	= 28.136;	// change AP shape
float tau_wp	= 213.55;	// change AP shape
float tau_fi 	= 0.084;	// change conduction (drifting rotor)
float tau_o1 	= 250.03;
float tau_o2 	= 16.632;
float tau_so1 	= 73.675;
float tau_so2 	= 6.554;
float tau_s1 	= 9.876; 	// change AP shape
float tau_s2 	= 4.203; 	// change AP shape
float tau_si 	= 10.699;
float tau_w_inf = 0.223;

float theta_v 	= 0.3;
float theta_w 	= 0.1817;
float theta_vm 	= 0.1007;
float theta_o	= 0.0155;

float u_wm		= 0.01;
float u_0 		= 0;
float u_u 		= 1.0089;
float u_so 		= 0.5921;
float u_s 		= 0.8157;

float k_wm 		= 60.219;
float k_so 		= 2.975;
float k_s 		= 2.227;

float w_inf_star= 0.902;
float V_mu 		= 85.7;
float V_0 		= -83.91;

// extracellular (unipolar) potential parameters
float global_dVmdx[X][Y][Z];		// parallel gradient Vm_x
float global_dVmdy[X][Y][Z];		// parallel gradient Vm_y
float global_Phi[N_E][N_E];			// parallel Phi

// Heaviside step function
float H(float x) {
	if (x > 0.00000000001){
		return(1.0);
	} else if (x < -0.00000000001){
		return(0.0);
	} else {
		return(0.5);
	}
}

// 4 current fenton-karma model
float fk (int x, int y, int z) {
	
	// tau's
	float tau_vm,tau_wm,tau_so,tau_s,tau_o;
	
	tau_vm = tau_v1m*(1-H(u[x][y][z]-theta_vm)) + tau_v2m*H(u[x][y][z]-theta_vm);
	
	tau_wm = tau_w1m + 0.5*(tau_w2m-tau_w1m)*(1+tanh(k_wm*(u[x][y][z]-u_wm)));
	tau_so = tau_so1 + 0.5*(tau_so2-tau_so1)*(1+tanh(k_so*(u[x][y][z]-u_so)));
	
	tau_s = tau_s1*(1-H(u[x][y][z]-theta_w)) + tau_s2*(H(u[x][y][z]-theta_w));
	tau_o = tau_o1*(1-H(u[x][y][z]-theta_o)) + tau_o2*(H(u[x][y][z]-theta_o));
	
	// J's
	float J_fi,J_so,J_si;
	
	J_fi = -(v[x][y][z]*(u[x][y][z]-theta_v)*(u_u-u[x][y][z]))*H(u[x][y][z]-theta_v)/tau_fi;
	J_so = (u[x][y][z]-u_0)*(1-H(u[x][y][z]-theta_w))/tau_o + H(u[x][y][z]-theta_w)/tau_so;
	J_si = -s[x][y][z]*w[x][y][z]*H(u[x][y][z]-theta_w)/tau_si;

	// infinity's
	float v_inf,w_inf;
	
	v_inf = H(theta_vm-u[x][y][z]);
	w_inf = (1-u[x][y][z]/tau_w_inf)*(1-H(u[x][y][z]-theta_o)) + w_inf_star*H(u[x][y][z]-theta_o);
	
	// gradients
	float du,dv,dw,ds;
	
	du = -(J_fi + J_so + J_si);
	
	dv = (v_inf-v[x][y][z])*(1-H(u[x][y][z]-theta_v))/tau_vm - v[x][y][z]*H(u[x][y][z]-theta_v)/tau_vp;
	dw = (w_inf-w[x][y][z])*(1-H(u[x][y][z]-theta_w))/tau_wm - w[x][y][z]*H(u[x][y][z]-theta_w)/tau_wp;
	
	ds = ((1+tanh(k_s*(u[x][y][z]-u_s)))/2 - s[x][y][z])/tau_s;
	
	// update parameters (except u)
	v[x][y][z] = v[x][y][z] + dt*dv;
	w[x][y][z] = w[x][y][z] + dt*dw;
	s[x][y][z] = s[x][y][z] + dt*ds;
	
	return (du);
}

/*// convert transmembrane action potential to extra cellular unipolar potential
void ConvertTransmembrane2ExtracellularPotential(float Vm[X+1][Y+1][Z+1]){
	
	// input: 	pointer to Vm, the action potential (transmembrane potential), shape = [X][Y][Z]
	// output:	Phi, the extracellular (unipolar) potential, shape = [N_E][N_E]
	// run time: ~25min on Nesi for 1000 time steps
	

	int i,j,k,t,x,y;	// counters
	float K = 0.0033;	// ratio of conductivities
	float spacing = 0.5;// discritization space
	
	float Vm_fx[X][Y][Z];	// gradient of Vm in x direction
	float Vm_fy[X][Y][Z];	// gradient of Vm in y direction
	
	// calculate gradient of Vm
	for(k = 0; k < Z; k++){ //for(k = BLOCK_LOW(rank,size,Z); k <= BLOCK_HIGH(rank,size,Z); k++){
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

	// calculate Phi, perform dot product then 3D integral (sum over X/Y/Z)
 	for(x = 0; x < N_E; x++){
		for(y = 0; y < N_E; y++){
			Phi[x][y] = 0.0;
			for(i = 0; i < X; i++){
				for(j = 0; j < Y; j++){
					for(k = 0; k < Z; k++){
						Phi[x][y] += Vm_fx[i][j][k]*r_inv_fx[i][j][k][x][y] + Vm_fy[i][j][k]*r_inv_fy[i][j][k][x][y];
					}
				}
			}
			Phi[x][y] = Phi[x][y]*-1.0*K;
		}
	}
}*/

int main(int argc, char **argv) {
	
	float K 		= 0.0033;// ratio of conductivities
	float spacing 	= 0.5;	 // discritization space
	float r_inv[X][Y][Z];				// r, the inverse distance matrix from the cell to the electrodes
	float r_inv_fx[X][Y][Z][N_E][N_E];	// gradient of r in the x direction
	float r_inv_fy[X][Y][Z][N_E][N_E];	// gradient of r in the y direction
	float Vm_fx[X][Y][Z];	// gradient of Vm in x direction
	float Vm_fy[X][Y][Z];	// gradient of Vm in y direction
	float Phi[N_E][N_E];	// extra cellular action potential
	
	

	int convert_to_extra_cellular = 0; // 0 if no, 1 if yes
	
	int x,y,z,i,j,k;
	
	int cnt = 0;
	int snp = 0;
	char c;
	
	FILE *in;
	FILE *out;
	char *str;

	float dudx2, dudy2, dudz2;
	float dudxdy, dudxdz, dudydz;
	float dudx, dudy, dudz;

	float gx, gy, gz;
	float du, fu;

	float t = 0.0;
	int tstim = 500;
	int delt = 50;
	int counter = 500;

	int isbound;
	int num = 0;
	int gg[27];
	int flag;

	float root2 = sqrt(2.0);
	float root3 = sqrt(3.0);
	float ic, ir, il, imax;
	float tflt;
	float temp; 

	int rank = 0;
	int size = 1;

	int *recv_cnts, *recv_disp ;
	MPI_Status status ;

	MPI_Init(&argc, &argv) ;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	recv_cnts = calloc (size, sizeof(int)) ;
	recv_disp = calloc (size, sizeof(int)) ;

	for (i = 0 ; i < size ; i++) {
		recv_cnts[i] = BLOCK_SIZE(i, size, X+1)*(Y+1)*(Z+1);
		recv_disp[i] = BLOCK_LOW(i, size, X+1)*(Y+1)*(Z+1);
	} 

	/* Pre-Computing distance matrix for convert transmembrane to extra cellular potential*/
	if (convert_to_extra_cellular == 1) {
		
		float x_dash,y_dash,z_dash,d;
		
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
							r_inv_fx[j][i][k][x][y] = (r_inv[j+1][i][k] - r_inv[j-1][i][k])/(2*spacing);
							r_inv_fy[i][j][k][x][y] = (r_inv[i][j+1][k] - r_inv[i][j-1][k])/(2*spacing);
						}
						r_inv_fx[0][i][k][x][y] = (r_inv[1][i][k] - r_inv[0][i][k])/spacing;
						r_inv_fy[i][0][k][x][y] = (r_inv[i][1][k] - r_inv[i][0][k])/spacing;
						r_inv_fx[X-1][i][k][x][y] = (r_inv[X-1][i][k] - r_inv[X-2][i][k])/spacing;
						r_inv_fy[i][Y-1][k][x][y] = (r_inv[i][Y-1][k] - r_inv[i][Y-2][k])/spacing;
					}
				}
				
			}
		}
		
	}
	
	if (!rank){
		for (z = 0; z < Z; z++) { 
			for (y = 0; y < Y; y++) { 
				for (x = 0; x < X; x++) {

					g[x][y][z] = 1; 
					h[x][y][z] = 1;

					if (z < 2 || z > Z-2){
						g[x][y][z] = 0;
						h[x][y][z] = 0;
					}
					if (x < 2 || x > X-2){
						g[x][y][z] = 0;
						h[x][y][z] = 0;
					}
					if (y < 2 || y > Y-2){
						g[x][y][z] = 0;
						h[x][y][z] = 0;
					}
				}
			}
		}

		for (z = 0; z < Z; z++) 
		  for (y = 0; y < Y; y++)  
			  for (x = 0; x < X; x++) {
				  xx[x][y][z] = 0.0; //gx;
				  yy[x][y][z] = 0.0; //gy;
				  zz[x][y][z] = 0.0; //gz;
			  }

		num = 1;      
		for (x = 1; x < X; x++) 
		  for (y = 1; y < Y; y++) 
			for (z = 1; z < Z; z++)
			  if (g[x][y][z] > 0) {
				g[x][y][z] = num;
				num++;
			  }

		for (x = 1; x < X; x++) 
		 for (y = 1; y < Y; y++) 
		  for (z = 1; z < Z; z++) {
			gg[1] = g[x - 1][y - 1][z - 1];
			gg[2] = g[x - 1][y - 1][z];
			gg[3] = g[x - 1][y - 1][z+1];
			gg[4] = g[x - 1][y][z - 1];
			gg[5] = g[x - 1][y][z];
			gg[6] = g[x - 1][y][z + 1];
			gg[7] = g[x - 1][y + 1][z - 1];
			gg[8] = g[x - 1][y + 1][z];
			gg[9] = g[x - 1][y + 1][z + 1];

			gg[10] = g[x][y - 1][z - 1];
			gg[11] = g[x][y - 1][z];
			gg[12] = g[x][y - 1][z + 1];
			gg[13] = g[x][y][z - 1];
			gg[14] = g[x][y][z + 1];
			gg[15] = g[x][y + 1][z - 1];
			gg[16] = g[x][y + 1][z];
			gg[17] = g[x][y + 1][z + 1];

			gg[18] = g[x + 1][y - 1][z - 1];
			gg[19] = g[x + 1][y - 1][z];
			gg[20] = g[x + 1][y - 1][z + 1];
			gg[21] = g[x + 1][y][z - 1];
			gg[22] = g[x + 1][y][z];
			gg[23] = g[x + 1][y][z + 1];
			gg[24] = g[x + 1][y + 1][z - 1];
			gg[25] = g[x + 1][y + 1][z];
			gg[26] = g[x + 1][y + 1][z + 1];

			isbound = 0;
			for(i = 1; i <= 26; i++) { 
			  if (gg[i] > 0) {gg[i] = 1; isbound++;} 
			  else gg[i] = 0;
			}

			if (g[x][y][z] == 0 && isbound > 0) {
			  ic = (gg[3]/root3) - (gg[1]/root3) + (gg[6]/root2) +
				   (gg[9]/root3) - (gg[7]/root3) - (gg[4]/root2) +
				   (gg[12]/root2) - (gg[10]/root2) + gg[14] +
				   (gg[17]/root2) - (gg[15]/root2) - gg[13] +
				   (gg[20]/root3) - (gg[18]/root3) + (gg[23]/root2) +
				   (gg[26]/root3) - (gg[24]/root3) - (gg[21]/root2);

			  ir = (gg[9]/root3) - (gg[2]/root2) - (gg[3]/root3) - 
				   (gg[1]/root3) + (gg[8]/root2) + (gg[7]/root3) +
				   (gg[17]/root2) - gg[11] - (gg[12]/root2) -
				   (gg[10]/root2) + gg[16] + (gg[15]/root2) +
				   (gg[26]/root3) - (gg[19]/root2) - (gg[20]/root3) -
				   (gg[18]/root3) + (gg[25]/root2) + (gg[24]/root3);

			  il = (gg[18]/root3) + (gg[19]/root2) + (gg[20]/root3) +
				   (gg[21]/root2) + gg[22] + (gg[23]/root2) +
				   (gg[24]/root3) + (gg[25]/root2) + (gg[26]/root3) -
				   (gg[1]/root3) - (gg[2]/root2) - (gg[3]/root3) -
				   (gg[4]/root2) - gg[5] - (gg[6]/root2) - 
				   (gg[7]/root3) - (gg[8]/root2) - (gg[9]/root3);

			  imax = fabs(ic);
			  if (fabs(ir) > imax) imax = fabs(ir);
			  if (fabs(il) > imax) imax = fabs(il);

			  i = 0; j = 0; k = 0;

			  tflt = ir / fabs(imax);
			  if (tflt <= 0.5 && tflt >= -0.5) i = 0;
			  else if (tflt > 0.5) i = 1;
			  else if (tflt < -0.5) i = -1;

			  tflt = ic / fabs(imax);
			  if (tflt <= 0.5 && tflt >= -0.5) j = 0;
			  else if (tflt > 0.5) j = 1;
			  else if (tflt < -0.5) j = -1;

			  tflt = il / fabs(imax);
			  if (tflt <= 0.5 && tflt >= -0.5) k = 0;
			  else if (tflt > 0.5) k = 1;
			  else if (tflt < -0.5) k = -1;

			  if (imax == 0) { i = 0; j = 0; k = 0; }

			  if (g[x + k][y + i][z + j] > 0)    
				g[x][y][z] = -1 * g[x + k][y + i][z + j];   
			  else
				g[x][y][z] = g[x + k][y + i][z + j];  
			}
		  }
	}

	MPI_Bcast (g, (X+1)*(Y+1)*(Z+1), MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (h, (X+1)*(Y+1)*(Z+1), MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (xx, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast (yy, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast (zz, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);

	for (x = 1; x < X; x++)
		for (y = 1; y < Y; y++) 
			for (z = 1; z < Z; z++) 
				if (g[x][y][z] > 0) {
					dc[x][y][z][1] = D2 + (DD * xx[x][y][z] * xx[x][y][z]);
					dc[x][y][z][2] = DD * xx[x][y][z] * yy[x][y][z];
					dc[x][y][z][3] = DD * xx[x][y][z] * zz[x][y][z];
					dc[x][y][z][4] = DD * yy[x][y][z] * xx[x][y][z];
					dc[x][y][z][5] = D2 + (DD * yy[x][y][z] * yy[x][y][z]);
					dc[x][y][z][6] = DD * yy[x][y][z] * zz[x][y][z];
					dc[x][y][z][7] = DD * zz[x][y][z] * xx[x][y][z];
					dc[x][y][z][8] = DD * zz[x][y][z] * yy[x][y][z];
					dc[x][y][z][9] = D2 + (DD * zz[x][y][z] * zz[x][y][z]);

					df[x][y][z][1] = (xx[x + 1][y][z] - xx[x - 1][y][z]) / (2*dx);
					df[x][y][z][2] = (xx[x][y + 1][z] - xx[x][y - 1][z]) / (2*dx);
					df[x][y][z][3] = (xx[x][y][z + 1] - xx[x][y][z - 1]) / (2*dx);       
					df[x][y][z][4] = (yy[x + 1][y][z] - yy[x - 1][y][z]) / (2*dx);
					df[x][y][z][5] = (yy[x][y + 1][z] - yy[x][y - 1][z]) / (2*dx);
					df[x][y][z][6] = (yy[x][y][z + 1] - yy[x][y][z - 1]) / (2*dx);       
					df[x][y][z][7] = (zz[x + 1][y][z] - zz[x - 1][y][z]) / (2*dx);
					df[x][y][z][8] = (zz[x][y + 1][z] - zz[x][y - 1][z]) / (2*dx);
					df[x][y][z][9] = (zz[x][y][z + 1] - zz[x][y][z - 1]) / (2*dx);       
				}

	for (x = 0; x <= X; x++)
		for (y = 0; y <= Y; y++) 
			for (z = 0; z <= Z; z++) {
				u[x][y][z] = 1.18*0.00000001;
				v[x][y][z] = 1.0; // 1 = closed
				w[x][y][z] = 1.0; // 1 = closed
				s[x][y][z] = 0.02587;
			}

	// Computer simulation
	while (t < 1000) {
		
		// S1-S2 protocol
		// (D1D2=0.025/t=160/240x240/1.0Stimulus@square1:120;1:120, stable rotor)
		// (D1D2=0.025/t=210/500x500/1.0Stimulus@square1:250;1:250, stable rotor)
		// (D1D2=0.050/t=150/500x500/tau_fi=0.008(min), drifting rotor. tau_fi > 0.05 similar to tau_fi==0.0834,D=0.05,t=200)
		// (slight breakup at tau_fi = 0.008,tau_fi > 0.08 more stable but weird activation and timing)
		if ((t>=0) && (t<(0 + 0.05))){
			for (x = 0; x < 240; x++)
				for (y = 0; y < 10; y++)
					for (z = 0; z < 5; z++)
							{if (h[x][y][z] > 0)
									u[x][y][z] = 2.0;}
		}
		if ((t>=160) && (t<(160+0.01))){
			for (x = 0; x < 120; x++)
				for (y = 0; y < 120; y++)
					for (z = 0; z < 5; z++)
						{u[x][y][z] = 1.0;}
		}
		
		/*// S1-S2 (2 rotor)
		// (D1D2=0.025/t=190/240x240/1.0Stimulus@70:170;50:120, 2 stable rotors)
		// (D1D2=0.025/t=280/500x500/1.0Stimulus@100:400;100:250, 2 stable rotors)
		// adjust x-limits to change the distance between rotor centroids
		if ((t>=0) && (t<(0 + 0.05))){
			for (x = 0; x < 500; x++)
				for (y = 0; y < 10; y++)
					for (z = 0; z < 5; z++)
							{if (h[x][y][z] > 0)
									u[x][y][z] = 2;}
		}
		if ((t>=190) && (t<(190+0.01))){
			for (x = 70; x < 170; x++)
				for (y = 50; y < 120; y++)
					for (z = 0; z < 5; z++)
						{u[x][y][z] = 1.0;}
		}*/
		
		
		for (x = BLOCK_LOW(rank, size, X+1); x <= BLOCK_HIGH(rank, size, X+1); x++) {
		  if (x == 0 || x == X) continue;
			  for (y = 1; y < Y; y++) 
				  for (z = 1; z < Z; z++) 
					  if (g[x][y][z] < 0)
						  for (i = -1; i <= 1; i++)
							  for (j = -1; j <= 1; j++)
								  for (k = -1; k <= 1; k++)
									  if (g[x][y][z] == -g[x + i][y + j][z + k]) 
										  u[x][y][z] = u[x + i][y + j][z + k];
		}
	  
		for (x = BLOCK_LOW(rank, size, X+1); x <= BLOCK_HIGH(rank, size, X+1); x++) {
		  if (x == 0 || x == X) continue;
			  for (y = 1; y < Y; y++)
				for (z = 1; z < Z; z++) {
				  if (h[x][y][z] > 0) {

					dudx2 = (u[x - 1][y][z] + u[x + 1][y][z] - 2 * u[x][y][z]) / (dx*dx);          
					dudy2 = (u[x][y - 1][z] + u[x][y + 1][z] - 2 * u[x][y][z]) / (dx*dx);
					dudz2 = (u[x][y][z - 1] + u[x][y][z + 1] - 2 * u[x][y][z]) / (dx*dx);

					dudxdy = (u[x + 1][y + 1][z] + u[x - 1][y - 1][z] - u[x + 1][y - 1][z] - u[x - 1][y + 1][z])/(4*dx*dx);  
					dudxdz = (u[x + 1][y][z + 1] + u[x - 1][y][z - 1] - u[x + 1][y][z - 1] - u[x - 1][y][z + 1])/(4*dx*dx);
					dudydz = (u[x][y + 1][z + 1] + u[x][y - 1][z - 1] - u[x][y + 1][z - 1] - u[x][y - 1][z + 1])/(4*dx*dx);

					dudx = (u[x + 1][y][z] - u[x - 1][y][z])/(2*dx);  
					dudy = (u[x][y + 1][z] - u[x][y - 1][z])/(2*dx);
					dudz = (u[x][y][z + 1] - u[x][y][z - 1])/(2*dx);

					du = (dc[x][y][z][1]*dudx2)  + (dudx * (DD*(xx[x][y][z]*df[x][y][z][1] + xx[x][y][z]*df[x][y][z][1]))) +
						 (dc[x][y][z][2]*dudxdy) + (dudy * (DD*(xx[x][y][z]*df[x][y][z][4] + yy[x][y][z]*df[x][y][z][1]))) +
						 (dc[x][y][z][3]*dudxdz) + (dudz * (DD*(xx[x][y][z]*df[x][y][z][7] + zz[x][y][z]*df[x][y][z][1]))) +

						 (dc[x][y][z][4]*dudxdy) + (dudx * (DD*(yy[x][y][z]*df[x][y][z][2] + xx[x][y][z]*df[x][y][z][5]))) +
						 (dc[x][y][z][5]*dudy2)  + (dudy * (DD*(yy[x][y][z]*df[x][y][z][5] + yy[x][y][z]*df[x][y][z][5]))) +
						 (dc[x][y][z][6]*dudydz) + (dudz * (DD*(yy[x][y][z]*df[x][y][z][8] + zz[x][y][z]*df[x][y][z][5]))) +

						 (dc[x][y][z][7]*dudxdz) + (dudx * (DD*(zz[x][y][z]*df[x][y][z][3] + xx[x][y][z]*df[x][y][z][9]))) +
						 (dc[x][y][z][8]*dudydz) + (dudy * (DD*(zz[x][y][z]*df[x][y][z][6] + yy[x][y][z]*df[x][y][z][9]))) +
						 (dc[x][y][z][9]*dudz2) + (dudz * (DD*(zz[x][y][z]*df[x][y][z][9] + zz[x][y][z]*df[x][y][z][9])));
						 
					fu = fk (x, y, z);
					new_u[x][y][z] = u[x][y][z] + dt*(du+fu);
				  }
				}
		}
		
		for (x = BLOCK_LOW(rank, size, X+1); x <= BLOCK_HIGH(rank, size, X+1); x++) 
		  for (y = 0; y <= Y; y++) 
			for (z = 0; z <= Z; z++) 
			  if (h[x][y][z] > 0)
				u[x][y][z] = new_u[x][y][z];
		
		// send u
		if (rank < (size-1))
		  MPI_Send(u[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
		if (rank > 0)
		  MPI_Recv(u[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
		if (rank > 0)
		  MPI_Send(u[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
		if (rank < (size-1))
		  MPI_Recv(u[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);
		
		// send v
		if (rank < (size-1))
		  MPI_Send(v[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
		if (rank > 0)
		  MPI_Recv(v[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
		if (rank > 0)
		  MPI_Send(v[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
		if (rank < (size-1))
		  MPI_Recv(v[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);
		
		// send w
		if (rank < (size-1))
		  MPI_Send(w[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
		if (rank > 0)
		  MPI_Recv(w[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
		if (rank > 0)
		  MPI_Send(w[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
		if (rank < (size-1))
		  MPI_Recv(w[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);
		
		// send s
		if (rank < (size-1))
		  MPI_Send(s[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
		if (rank > 0)
		  MPI_Recv(s[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
		if (rank > 0)
		  MPI_Send(s[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
		if (rank < (size-1))
		  MPI_Recv(s[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);
		
		// collect variables u,v,w,s
		if (snp % 400 == 0) {  
			MPI_Gatherv (&(((float *)u)[recv_disp[rank]]), recv_cnts[rank], MPI_FLOAT, &global_u[0][0][0], recv_cnts, recv_disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
			MPI_Gatherv (&(((float *)v)[recv_disp[rank]]), recv_cnts[rank], MPI_FLOAT, &global_v[0][0][0], recv_cnts, recv_disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
			MPI_Gatherv (&(((float *)w)[recv_disp[rank]]), recv_cnts[rank], MPI_FLOAT, &global_w[0][0][0], recv_cnts, recv_disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
			MPI_Gatherv (&(((float *)s)[recv_disp[rank]]), recv_cnts[rank], MPI_FLOAT, &global_s[0][0][0], recv_cnts, recv_disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}
		
		if (!rank) {
		  if (snp % 400 == 0) {

			cnt++;

			// Save phi
			if (convert_to_extra_cellular == 1){
				
				if (t>=0){

					// calculate gradient of Vm
					for(k = 0; k < Z; k++){//for(k = BLOCK_LOW(rank,size,Z); k < BLOCK_HIGH(rank,size,Z); k++){
						for(i = 0; i < X; i++){
							for(j = 1; j < X-1; j++){
								Vm_fx[j][i][k] = (global_u[j+1][i][k] - global_u[j-1][i][k])/(2*spacing);
								Vm_fy[i][j][k] = (global_u[i][j+1][k] - global_u[i][j-1][k])/(2*spacing);
							}
							Vm_fx[0][i][k] = (global_u[1][i][k] - global_u[0][i][k])/spacing;
							Vm_fy[i][0][k] = (global_u[i][1][k] - global_u[i][0][k])/spacing;
							Vm_fx[X-1][i][k] = (global_u[X-1][i][k] - global_u[X-2][i][k])/spacing;
							Vm_fy[i][Y-1][k] = (global_u[i][Y-1][k] - global_u[i][Y-2][k])/spacing;
						}
					}
					//MPI_Gatherv (&(((float *)Vm_fx)[recv_disp[rank]]),recv_cnts[rank],MPI_FLOAT,&global_dVmdx[0][0][0],recv_cnts,recv_disp,MPI_FLOAT,0,MPI_COMM_WORLD);
					//MPI_Gatherv (&(((float *)Vm_fy)[recv_disp[rank]]),recv_cnts[rank],MPI_FLOAT,&global_dVmdy[0][0][0],recv_cnts,recv_disp,MPI_FLOAT,0,MPI_COMM_WORLD);
					
					// calculate Phi, perform dot product then 3D integral (sum over X/Y/Z)
					for(x = 0; x < N_E; x++){//for (x = BLOCK_LOW(rank,size,N_E); x < BLOCK_HIGH(rank,size,N_E); x++){
						for(y = 0; y < N_E; y++){
							Phi[x][y] = 0.0;
							for(i = 0; i < X; i++){
								for(j = 0; j < Y; j++){
									for(k = 0; k < Z; k++){
										Phi[x][y] += Vm_fx[i][j][k]*r_inv_fx[i][j][k][x][y] + Vm_fy[i][j][k]*r_inv_fy[i][j][k][x][y];
									}
								}
							}
							Phi[x][y] = Phi[x][y]*-1.0*K;
						}
					}
					//MPI_Gatherv (&(((float *)Phi)[recv_disp[rank]]),recv_cnts[rank],MPI_FLOAT,&global_Phi[0][0],recv_cnts,recv_disp,MPI_FLOAT,0,MPI_COMM_WORLD);
					
					// Save Phi
					str = malloc (40*sizeof(char));
					sprintf (str, "Test/pp%04d.vtk", cnt);
					out = fopen (str, "wt");
					for(x = 0; x < N_E; x++){
						for(y = 0; y < N_E; y++)
							fprintf (out, "%3.1f ", 100*Phi[x][y]);
						fprintf (out, "\n");
					}
					fclose (out);
					free (str);
				}
					
			} else {
				
				// Save u
				str = malloc (40*sizeof(char));
				sprintf (str, "Test1/aa%04d.vtk", cnt);
				out = fopen (str, "wt");
				for (z = 0; z < Z; z++) {
				  for (y = 0; y < Y; y++) {
					for (x = 0; x < X; x++)
					  fprintf (out, "%3.1f ", 100*global_u[x][y][z]);
					fprintf (out, "\n");
				  }
				  fprintf (out, "\n");
				}          	
				fclose (out);
				free (str);
				/*
				// Save v
				str = malloc (40*sizeof(char));
				sprintf (str, "Test/DataV%04d.vtk", cnt);
				out = fopen (str, "wt");
				for (z = 0; z < Z; z++) {
				  for (y = 0; y < Y; y++) {
					for (x = 0; x < X; x++)
					  fprintf (out, "%3.1f ", 100*global_v[x][y][z]);
					fprintf (out, "\n");
				  }
				  fprintf (out, "\n");
				}
				fclose (out);
				free (str);
				
				// Save w
				str = malloc (40*sizeof(char));
				sprintf (str, "Test/DataW%04d.vtk", cnt);
				out = fopen (str, "wt");
				for (z = 0; z < Z; z++) {
				  for (y = 0; y < Y; y++) {
					for (x = 0; x < X; x++)
					  fprintf (out, "%3.1f ", 100*global_w[x][y][z]);
					fprintf (out, "\n");
				  }
				  fprintf (out, "\n");
				}
				fclose (out);
				free (str);
				
				// Save s
				str = malloc (40*sizeof(char));
				sprintf (str, "Test/DataS%04d.vtk", cnt);
				out = fopen (str, "wt");
				for (z = 0; z < Z; z++) {
				  for (y = 0; y < Y; y++) {
					for (x = 0; x < X; x++)
					  fprintf (out, "%3.1f ", 100*global_s[x][y][z]);
					fprintf (out, "\n");
				  }
				  fprintf (out, "\n");
				}
				fclose (out);
				free (str);
				*/
			}

		  }     
		}
		t += dt;
		snp++;
	}
 
  if (recv_cnts) free (recv_cnts) ;
  if (recv_disp) free (recv_disp) ;
  MPI_Finalize() ;
 
  return 0 ;
}