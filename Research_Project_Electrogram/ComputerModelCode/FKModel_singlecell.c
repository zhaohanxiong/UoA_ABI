#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define X    1 // 240
#define Y    1 // 240
#define Z    1 // 5
#define dt   0.005

// Gating variable
float u[X + 1][Y + 1][Z + 1];
float v[X + 1][Y + 1][Z + 1];
float w[X + 1][Y + 1][Z + 1];
float s[X + 1][Y + 1][Z + 1];
float new_u[X+1][Y+1][Z+1];

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

// 3 current FK model
float tau_d   = 0.125;
float tau_r   = 70.0;
float tau_si  = 114.0;
float tau_o   = 32.5;
float tau_vp  = 5.75;
float tau_v1m = 82.5;
float tau_v2m = 60.0;
float tau_wp  = 300.0;
float tau_wm  = 400.0;

float u_c     = 0.16;
float u_v     = 0.04;
float u_csi   = 0.85;

float kk      = 10.0;

float V_mu    = 87.51;
float V_0     = -83.91;

float fk (int x, int y, int z) {
	
	// J's
	float J_fi,J_so,J_si;
	
	J_fi = -(v[x][y][z]/tau_d)*H(u[x][y][z]-u_c)*(1.0-u[x][y][z])*(u[x][y][z]-u_c);
	J_so = (u[x][y][z]/tau_o)*H(u_c-u[x][y][z]) + (1.0/tau_r)*H(u[x][y][z]-u_c);
	J_si = -(w[x][y][z]/(2*tau_si))*(1.0+tanh(kk*(u[x][y][z]-u_csi)));
	
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

/*// 4 current FK model
float tau_v1m 	= 16.3;
float tau_v2m 	= 1150;
float tau_vp 	= 1.703;
float tau_w1m 	= 79.963;
float tau_w2m 	= 28.136;
float tau_wp	= 213.55;
float tau_fi 	= 0.084;
float tau_o1 	= 250.03;
float tau_o2 	= 16.632;
float tau_so1 	= 73.675;
float tau_so2 	= 6.554;
float tau_s1 	= 9.876;
float tau_s2 	= 4.203;
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
	
}*/

int main(int argc, char **argv) {

	int x, y, z;
	int cnt = 0,snp = 0;
	float t = 0.0;
	float J_stim;
	
	FILE *in;
	FILE *out;
	char *str;

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
		
		// set the initial stimulus (S1)
		if ((t>=50) && (t<52)){
			J_stim = 0.2;
		} else {
			J_stim = 0.0;
		}

		// calculate the action potential
		for (x = 0; x < X; x++)
			for (y = 0; y < Y; y++)
				for (z = 0; z < Z; z++) {
					u[x][y][z] = u[x][y][z] + dt*(fk(x,y,z)+J_stim);
					new_u[x][y][z] = u[x][y][z]*V_mu+V_0;
				}

		if (snp % 200 == 0) {
			// Save u
			str = malloc (40*sizeof(char));
			cnt++;
			sprintf (str, "Test/aa%04d.vtk", cnt);
			out = fopen (str, "wt");
			for (z = 0; z < Z; z++) {
			  for (y = 0; y < Y; y++) {
				for (x = 0; x < X; x++)
				  fprintf (out, "%3.1f ", new_u[x][y][z]);
				fprintf (out, "\n");
			  }
			  fprintf (out, "\n");
			}          	
			fclose (out);
			free (str);
		}
		t += dt;
		snp++;
	}

  return 0 ;
}