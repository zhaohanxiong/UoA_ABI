// writes rotors
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define BLOCK_LOW(id,p,n) ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id+1),p,n)-1)
#define BLOCK_SIZE(id,p,n) (BLOCK_LOW((id+1),p,n)-BLOCK_LOW(id,p,n))
#define X    500 // 240
#define Y    500 // 240
#define Z    5

/*
#define D1   3.2 
#define D2   0.32 
#define DD   2.88  //0.0025/(0.15)^2*(DD)^2<1, thus DD<3.0 
*/
#define D1   0.05 //0.3//1.2//3.3 
#define D2   0.05 //0.06//1.2//0.33
#define DD   0.0 //0.24//2.97   

//below is used to test in T2 with D1   0.5, D2   0.5 
#define dt   0.0025 //0.0025   // ms
#define dx   0.15 //0.04       // mm

#define R     8314
#define T     308
#define F     96487
#define RTOnF (T*0.08554)
//#define	tau_d (1.0 / gfi)

float global_u[X+1][Y+1][Z+1];
float new_u[X+1][Y+1][Z+1];
float global_v[X+1][Y+1][Z+1];
float global_w[X+1][Y+1][Z+1];

float xx[X+1][Y+1][Z+1];
float yy[X+1][Y+1][Z+1];
float zz[X+1][Y+1][Z+1];  

int g[X+1][Y+1][Z+1];
int h[X+1][Y+1][Z+1];

float dc[X+1][Y+1][Z+1][10];
float df[X+1][Y+1][Z+1][10];

//float gfi     = 4.0;
//float tau_r   = 66.0;
//float tau_si  = 45.0;
//float tau_o   = 8.3;
//float tau_vp  = 3.33;
//float tau_vm1 = 1000.0;
//float tau_vm2 = 19.2;
//float tau_wp  = 667.0; 
//float tau_wm  = 11.0;
//float u_c     = 0.13;
//float u_v     = 0.055;
//float usi_c   = 0.85;
//float kk      = 10.0;

// for PV sleeves  
float tau_d = 0.125;
float tau_r   = 125.0;
float tau_si  = 114.0;
float tau_o   = 32.5;
float tau_vp  = 5.75;
float tau_vm1 = 82.5;
float tau_vm2 = 60.0;
float tau_wp  = 300.0;
float tau_wm  = 90.0;
float u_c     = 0.16;
float u_v     = 0.04;
float usi_c   = 0.85;
float kk      = 10.0;
/*
// fk = SAN
float tau_d = 0.38; //0.38
float tau_r   = 135.0;
float tau_si  = 400; //300 -> 150ms; with increasing this one will decrease APDs; 
float tau_o   = 12.5;
float tau_vp  = 3.33; //
float tau_vm1 = 10.0;
float tau_vm2 = 18.2;
float tau_wp  = 1020.0; //
float tau_wm  = 80.0;
float u_c     = 0.13; //
float u_v     = 0.13;
float usi_c   = 0.13; //
float kk      = 10;
//float gsi     = 0.52;
*/
float u[X + 1][Y + 1][Z + 1];
float v[X + 1][Y + 1][Z + 1];
float w[X + 1][Y + 1][Z + 1];

float fk (int x, int y, int z)
{
float dv, dw;
float tau_vm;
float Jfi, Jso, Jsi;

  if (u[x][y][z] < u_c) {
    Jfi = 0.0;
    Jso = u[x][y][z] / tau_o;
    if (u[x][y][z] > u_v)
      dv = (1.0 - v[x][y][z]) / tau_vm1;
    else dv = (1.0 - v[x][y][z]) / tau_vm2;
    dw = (1.0 - w[x][y][z]) / tau_wm;
  }
  else {
    Jfi = -(v[x][y][z] / tau_d) * (1.0 - u[x][y][z]) * (u[x][y][z] - u_c);
    Jso = 1.0 / tau_r;
    dv = -v[x][y][z] / tau_vp;
    dw = -w[x][y][z] / tau_wp;
  }
  Jsi = -0.5 * (w[x][y][z] / tau_si) * (1 + tanh(kk * (u[x][y][z] - usi_c)));
  v[x][y][z] = v[x][y][z] + dt * dv;
  w[x][y][z] = w[x][y][z] + dt * dw;
  return (-(Jfi + Jso + Jsi)); 
}

int main(int argc, char **argv)
//int main()
{ 
	int x, y, z;
	long int cnt = 0;
	long int snp = 0;
	char c;
	int dd;

	FILE *in;
	FILE *out;
	char *str;

	float dudx2, dudy2, dudz2;
	float dudxdy, dudxdz, dudydz;
	float dudx, dudy, dudz;

	float gx, gy, gz;
	float du, fu;

	float t = 0.0;
	//float tstim = 0.1;
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
	//float global_u[X+1][Y+1][Z+1];
	//float new_u[X+1][Y+1][Z+1];

	int rank = 0;
	int size = 1;
        //int long TN = (X+1)*(Y+1)*(Z+1)*(10);
        //long int TN = 898li*541li*487li*10li;
        // (X+1) ul*(Y+1) ul*(Z+1) ul*(10) ul
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
	/*
	   ENa = RTOnF*log(Nao/Nai);
	   ECa = 0.5*RTOnF*log(Cao/Cai);
	   EK  = RTOnF*log(Ko/Ki);
	   EKs = RTOnF*log((Ko+0.003*Nao)/(Ki+0.03*Nai));
	   ECl=RTOnF*exp(30./132.); 
	   EbCl=ECl-0.49*(ECl+30.59);
	 */  
	if (!rank) {   
		//in = fopen ("Input2Datrium.dat", "r");
		for (z = 0; z < Z; z++) { 
			for (y = 0; y < Y; y++) { 
				for (x = 0; x < X; x++) {
					//fscanf(in, "%c", &c);

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
				//fscanf(in, "\n"); 
			} 
        //fscanf(in, "\n"); 
		}
    //fclose (in);

	for (z = 0; z < Z; z++) 
      for (y = 0; y < Y; y++)  
	      for (x = 0; x < X; x++) {
		      xx[x][y][z] = 0.0; //gx;
		      yy[x][y][z] = 0.0; //gy;
		      zz[x][y][z] = 0.0; //gz;
	      }

	/*
	   for (x = 1; x < X; x++) 
      for (y = 1; y < Y; y++) 
	    for (z = 1; z < Z; z++) 
          if (g[x][y][z] != 2)
		    g[x][y][z] = 0;
*/
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
      u[x][y][z] = 0.0;
      v[x][y][z] = 1.0; // 1 = closed
      w[x][y][z] = 1.0; // 1 = closed
    }

/////////////////////////////////////////////////// for reload to restart  
// also need to comment out fabs (t) < 1 
/*
in = fopen ("Pacing/T3/restart.vtk", "r");
  for (z = 0; z < Z; z++) { 
    for (y = 0; y < Y; y++) { 
          for (x = 0; x < X; x++)
                 {
                     fscanf(in, "%f ", &temp);
                     u[x][y][z] = temp/100; 
                     }
                    fscanf(in, "\n");
                  } 
                  fscanf(in, "\n");
  }
fclose (in);
*/



while (t < 10000) {


/******************************************************************************************************************************/
	if ((t>=0) && (t<(0 + 0.05))){
		// if (!rank) {
			// out = fopen ("./para2D.dat", "a");
			// fprintf (out, "%d %d %d %f\n", delt, counter, tstim, t);
			// fclose (out);
		// }

		// set the initial stimulus (S1)
		for (x = 0; x < 500; x++) /* 85 - 90*/
			for (y = 0; y < 10; y++) /* 85 - 90*/
				for (z = 0; z < 5; z++)
						{if (h[x][y][z] > 0)
								u[x][y][z] = 2.0;}

		if (counter <= 500){
			delt = 50;
		} 
		if (counter <= 200){
			delt = 10;
		}
		counter = counter - delt;
		tstim = tstim + counter;
	}
	
	// S2 protocol 
	// (D1D2=0.05/t=220/500x500/1.0Stimulus/square1:250;1:250, pretty much rotor)
	// (0.1/220/rectangle, worse rotation) (0.1/220/line, even worse rotation)
	if ((t>=220) && (t<(220+0.01))){
		for (x = 0; x < 250; x++)
			for (y = 0; y < 250; y++)
				for (z = 0; z < 5; z++)
					{u[x][y][z] = 1.0;}
	}
	
/******************************************************************************************************************************/
	
	
	for (x = BLOCK_LOW(rank, size, X+1); x <= BLOCK_HIGH(rank, size, X+1); x++) {
      if (x == 0 || x == X) continue;
	  // for (x = 1; x < X; x++)
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

    //for (x = 1; x < X; x++) {
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
				 
			//du=D*(dudx2 + dudy2 + dudz2) + fk(x,y,z); 
			fu = fk (x, y, z);
            //new_u[x][y][z] = u[x][y][z] + dt * du;
            new_u[x][y][z] = u[x][y][z] + dt * (du + fu);
          }
        }
    //}
}
    for (x = BLOCK_LOW(rank, size, X+1); x <= BLOCK_HIGH(rank, size, X+1); x++) 
    // for (x = 0; x <= X; x++)
      for (y = 0; y <= Y; y++) 
        for (z = 0; z <= Z; z++) 
          if (h[x][y][z] > 0)
	    u[x][y][z] = new_u[x][y][z];

    if (rank < (size-1))
      MPI_Send(u[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(u[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(u[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(u[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

    if (rank < (size-1))
      MPI_Send(v[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(v[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(v[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(v[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

    if (rank < (size-1))
      MPI_Send(w[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(w[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(w[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(w[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

    if (snp % 400 == 0) {  
      MPI_Gatherv (&(((float *)u)[recv_disp[rank]]), recv_cnts[rank], MPI_FLOAT, &global_u[0][0][0], recv_cnts, recv_disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gatherv (&(((float *)v)[recv_disp[rank]]), recv_cnts[rank], MPI_FLOAT, &global_v[0][0][0], recv_cnts, recv_disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gatherv (&(((float *)w)[recv_disp[rank]]), recv_cnts[rank], MPI_FLOAT, &global_w[0][0][0], recv_cnts, recv_disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    
    if (!rank) {
      if (snp % 400 == 0) {

        str = malloc (40*sizeof(char));
        cnt++;
        sprintf (str, "Test/aa%05d.vtk", cnt);
        out = fopen (str, "wt");
        for (z = 0; z < Z; z++) {
          for (y = 0; y < Y; y++) {
            for (x = 0; x < X; x++)
              fprintf (out, "%3.1f ", 100*global_u[x][y][z]);
            //fprintf (out, "%5.3f ", 100*u[x][y][z]);
            fprintf (out, "\n");
          }
          fprintf (out, "\n");
	    }          	
        fclose (out);
        free (str);

        /* // Start: the othe two variables
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
        // End: the othe two variables */
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