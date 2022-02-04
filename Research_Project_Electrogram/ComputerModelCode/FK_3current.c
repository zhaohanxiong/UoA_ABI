#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define BLOCK_LOW(id,p,n) ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id+1),p,n)-1)
#define BLOCK_SIZE(id,p,n) (BLOCK_LOW((id+1),p,n)-BLOCK_LOW(id,p,n))

#define X    500 
#define Y    500 
#define Z    5
#define D1   0.05
#define D2   0.05
#define DD   0.00

#define dt   0.0025
#define dx   0.15

#define R     8314
#define T     308
#define F     96487
#define RTOnF (T*0.08554)

float global_u[X+1][Y+1][Z+1];
float global_v[X+1][Y+1][Z+1];
float global_w[X+1][Y+1][Z+1];

float u[X+1][Y+1][Z+1];
float v[X+1][Y+1][Z+1];
float w[X+1][Y+1][Z+1];
float new_u[X+1][Y+1][Z+1];

float xx[X+1][Y+1][Z+1];
float yy[X+1][Y+1][Z+1];
float zz[X+1][Y+1][Z+1];  
float dc[X+1][Y+1][Z+1][10];
float df[X+1][Y+1][Z+1][10];

int g[X+1][Y+1][Z+1];
int h[X+1][Y+1][Z+1];

// 3 current Fenton Karma model paramters
//
// Parameter sets obtained from the paper: 
//		Fenton, Cherry, Hastings, Evans: "Multiple mechanisms of spiral wave break up in a model of the cardiac eletrical activity"
//
//						cAF			Set1		Set2		Set3		Set4		Set6
float tau_vp  = 5.75; 	//5.75		3.33		10.0		3.33		3.33		3.33
float tau_v1m = 82.5; 	//82.5		19.6		10.0		19.6		15.6		9.0
float tau_v2m = 60.0; 	//60.0		1000.0		10.0		1250.0		5.0			8.0
float tau_wp  = 300.0; 	//300.0		667.0		300.0		870.0		350.0		250.0
float tau_wm  = 400.0; 	//400.0		11.0		400.0		41.0		80.0		60.0
float tau_d   = 0.125; 	//0.125		0.25		0.25		0.25		0.407		0.395
float tau_o   = 32.5; 	//32.5		8.3			10.0		12.5		9.0			9.0
float tau_r   = 70.0; 	//70.0		50.0		190.0		33.33		34.0		33.33
float tau_si  = 114.0; 	//114.0		45.0		114.0		29.0		26.5		29.0

float kk      = 10.0; 	//10.0		10.0		10.0		10.0		15.0		15.0

float u_csi   = 0.85; 	//0.85;		0.85		0.85		0.85		0.45		0.5
float u_c     = 0.16; 	//0.16;		0.13		0.13		0.13		0.15		0.13
float u_v     = 0.04; 	//0.04;		0.055		0.04		0.04		0.04		0.04

float V_fi    = 3.6;		
float V_0     = -83.91;

// Heaviside step function
float H(float x) {
	if (x > 0.00001){
		return(1.0);
	} else if (x < -0.00001){
		return(0.0);
	} else {
		return(0.5);
	}
}

// 3 current fenton-karma model
float fk (int x, int y, int z) {
	
	// J's
	float J_fi,J_so,J_si;
	
	J_fi = -(v[x][y][z]/tau_d)*H(u[x][y][z]-u_c)*(1.0-u[x][y][z])*(u[x][y][z]-u_c);	// Na+
	J_so = (u[x][y][z]/tau_o)*H(u_c-u[x][y][z]) + (1.0/tau_r)*H(u[x][y][z]-u_c);	// K+
	J_si = -(w[x][y][z]/(2*tau_si))*(1.0+tanh(kk*(u[x][y][z]-u_csi)));				// Ca+
	
	// tau
	float tau_vm;
	
	tau_vm = H(u[x][y][z]-u_v)*tau_v1m + H(u_v-u[x][y][z])*tau_v2m;
	
	// gradients
	float du,dv,dw;
	
	du = -(J_fi + J_so + J_si);
	dv = H(u_c-u[x][y][z])*(1.0-v[x][y][z])/tau_vm - H(u[x][y][z]-u_c)*v[x][y][z]/tau_vp;
	dw = H(u_c-u[x][y][z])*(1.0-w[x][y][z])/tau_wm - H(u[x][y][z]-u_c)*w[x][y][z]/tau_wp;
	
	// update parameters (except u)
	v[x][y][z] = v[x][y][z] + dt*dv;
	w[x][y][z] = w[x][y][z] + dt*dw;

	return (du);
	
}

int main(int argc, char **argv) {

	int x, y, z;
	long int cnt = 0;
	long int snp = 0;
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
	int i, j, k;
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
			  for (x = 0; x < X; x++){
				  xx[x][y][z] = 0.0;
				  yy[x][y][z] = 0.0;
				  zz[x][y][z] = 0.0;
			  }

		num = 1;      
		for (x = 1; x < X; x++) 
		  for (y = 1; y < Y; y++) 
			for (z = 1; z < Z; z++)
			  if (g[x][y][z] > 0){
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
				v[x][y][z] = 1.0;
				w[x][y][z] = 1.0; 
			}

	// Computer simulation
	while (t < 250){ // 10000 == 10s is ideal
		
		// S1-S2 protocol (1 rotor)
		// for 240 x 240, activation at t = 125, D1/D2 = 0.05
		// for 500 x 500, activation at t = 165, D1/D2 = 0.05, tau_d = 0.125
		// for 500 x 500, activation at t = 155, D1/D2 = 0.05, tau_d = 0.075
		// for 500 x 500, activation at t = 130, D1/D2 = 0.05, tau_d = 0.025
		if ((t>=0) && (t<(0 + 0.05))){
			for (x = 0; x < 500; x++)
				for (y = 0; y < 10; y++)
					for (z = 0; z < 5; z++)
							{if (h[x][y][z] > 0)
								u[x][y][z] = 2.0;}
		}
		// parameter set 1: t=180,
		if ((t>=155) && (t<(155+0.01))){
			for (x = 0; x < 250; x++)
				for (y = 0; y < 250; y++)
					for (z = 0; z < 5; z++)
						{u[x][y][z] = 1.0;}
		}
		
		if ((t>=0) && (t<(0 + 0.05))){
			for (x = 245; x < 255; x++)
				for (y = 245; y < 255; y++)
					for (z = 0; z < 5; z++)
							{if (h[x][y][z] > 0)
								u[x][y][z] = 2.0;}
		}

		
		/* stable rotor (no break up) */
		// small oscillation 			- parameter cAF, t=125,tau_d=0.125
		// medium oscillation 			- parameter cAF, t=120,tau_d=0.075
		// large oscillation 			- parameter cAF, t=120,tau_d=0.050
		// larger oscillation			- parameter cAF, t=105,tau_d=0.025
		// large oscillation break@1.5s	- parameter cAF, t=105,tau_d=0.015
		
		/* wavelets: */
		// break up after  1 cycle  - parameter set 1, t=180, tau_d=0.15
		// break up after ~5 cycles - parameter set 1, t=180, tau_d=0.125
		// tau_d = 0.1 no rotor
		
		/* mother rotor */
		// small central rotor - parameter set 1 with tau_d=0.175, t=180
		//tau_d=0.20 no rotor
		
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
					u[x][y][z] = u[x][y][z] + dt*(du+fu);
				  }
				}
		}
		
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
		
		// collect variables u,v,w
		if (snp % 400 == 0){  
			MPI_Gatherv (&(((float *)u)[recv_disp[rank]]), recv_cnts[rank], MPI_FLOAT, &global_u[0][0][0], recv_cnts, recv_disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
			MPI_Gatherv (&(((float *)v)[recv_disp[rank]]), recv_cnts[rank], MPI_FLOAT, &global_v[0][0][0], recv_cnts, recv_disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
			MPI_Gatherv (&(((float *)w)[recv_disp[rank]]), recv_cnts[rank], MPI_FLOAT, &global_w[0][0][0], recv_cnts, recv_disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}
		
		if (!rank){
			if (snp % 400 == 0){
				cnt++;
				if (t >= 0){
					// Save u
					str = malloc (40*sizeof(char));
					sprintf (str, "Test1/aa%04d.vtk", cnt);
					out = fopen (str, "wt");
					for (z = 0; z <= Z; z++) {
					  for (y = 0; y <= Y; y++) {
						for (x = 0; x <= X; x++)
						  fprintf (out, "%3.1f ", 100*global_u[x][y][z]);
						fprintf (out, "\n");
					  }
					  fprintf (out, "\n");
					}          	
					fclose (out);
					free (str);
					
					
					// Save v
					str = malloc (40*sizeof(char));
					sprintf (str, "Test/DataV%04d.vtk", cnt);
					out = fopen (str, "wt");
					for (z = 0; z <= Z; z++) {
					  for (y = 0; y <= Y; y++) {
						for (x = 0; x <= X; x++)
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
					for (z = 0; z <= Z; z++) {
					  for (y = 0; y <= Y; y++) {
						for (x = 0; x <= X; x++)
						  fprintf (out, "%3.1f ", 100*global_w[x][y][z]);
						fprintf (out, "\n");
					  }
					  fprintf (out, "\n");
					}
					fclose (out);
					free (str);
					
				}
			}
		}
		t += dt;
		snp++;
	}
 
	if (recv_cnts) free(recv_cnts);
	if (recv_disp) free(recv_disp);
	MPI_Finalize();

	return 0;
}