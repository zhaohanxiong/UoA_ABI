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
//#define D    0.6
//#define D1   1.4 
//#define D2   0.14 
//#define DD   1.26
#define D1   0.05 // conduction velocity
#define D2   0.05 // conduction velocity
#define DD   0.0 //

//below is used to test in T2 with D1   0.5, D2   0.5 
#define dt   0.0025 //0.000005   // ms
#define dx   0.15 //0.04       // mm
//#define R     8314
//#define T     308
//#define F     96487
//#define RTOnF (T*0.08554)
//#define	tau_d (1.0 / gfi)

float global_u[X+1][Y+1][Z+1];
float new_u[X+1][Y+1][Z+1];
float global_ina[X+1][Y+1][Z+1];
float global_ilca[X+1][Y+1][Z+1];

float xx[X+1][Y+1][Z+1];
float yy[X+1][Y+1][Z+1];
float zz[X+1][Y+1][Z+1];  

int g[X+1][Y+1][Z+1];
int h[X+1][Y+1][Z+1];

float dc[X+1][Y+1][Z+1][10];
float df[X+1][Y+1][Z+1][10];
// for PV sleeves  
/*float tau_d = 0.125;
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
*/
float u[X + 1][Y + 1][Z + 1];
float hh[X+1][Y+1][Z+1];
float m[X+1][Y+1][Z+1];
float jj[X+1][Y+1][Z+1];
float u[X+1][Y+1][Z+1];
float ina[X+1][Y+1][Z+1];
float nai[X+1][Y+1][Z+1];
float cai[X+1][Y+1][Z+1];
float d[X+1][Y+1][Z+1];
float f[X+1][Y+1][Z+1];
float fca[X+1][Y+1][Z+1];
float ilca[X+1][Y+1][Z+1];
float xr[X+1][Y+1][Z+1];
float ikr[X+1][Y+1][Z+1];
float xs[X+1][Y+1][Z+1];
float iks[X+1][Y+1][Z+1];
float iki[X+1][Y+1][Z+1];
float yach[X+1][Y+1][Z+1];
float ikach[X+1][Y+1][Z+1];
float uakur[X+1][Y+1][Z+1];
float uikur[X+1][Y+1][Z+1];
float ikur[X+1][Y+1][Z+1];
float ato[X+1][Y+1][Z+1];
float iito[X+1][Y+1][Z+1];
float ito[X+1][Y+1][Z+1];
float inaca[X+1][Y+1][Z+1];
float inak[X+1][Y+1][Z+1];
float ipca[X+1][Y+1][Z+1];
float icab[X+1][Y+1][Z+1];
float inab[X+1][Y+1][Z+1];
float ki[X+1][Y+1][Z+1];
float nsr[X+1][Y+1][Z+1];
float urel[X+1][Y+1][Z+1];
float vrel[X+1][Y+1][Z+1];
float wrel[X+1][Y+1][Z+1];
float jsr[X+1][Y+1][Z+1];
float itr[X+1][Y+1][Z+1];

//float v[X + 1][Y + 1][Z + 1];
//float w[X + 1][Y + 1][Z + 1];
///////////////////     Courtemanche  cell model parameters ///////////////////
/* The Courtemanche Model of the Human Atrial Myocyte 
   This code requires a C++ compiler 
   Acetylcholine-activated K channel current (GIRK3.1) was incoporated 
   Am. J. Physiol. 1998:275 H301-H321
   Modified to C by HJ LAI 2006/08/01 */
   /* List of variables and paramaters (this code uses all global variables) */
        /* Creation of Data File */
        FILE *ap;
        FILE *fmaxs;
        FILE *fpara;
        void prttofile ();      
        int printdata;  
        int printval;   

        /* Cell Geometry */
        float l = 0.01;       /* Length of the cell (cm) */
        float a = 0.0008;     /* Radius of the cell (cm) */
        float pi = 3.141592;  /* Pi */
        float vcell;   /* Cell volume (uL) */
        float ageo;    /* Geometric membrane area (cm^2) */
        float acap;    /* Capacity */
        float vmyo;    /* Myoplasm volume (uL) */
        float vmito;   /* Mitochondria volume (uL) */
        float vsr;     /* SR volume (uL) */
        float vnsr;    /* NSR volume (uL) */
        float vjsr;    /* JSR volume (uL) */
        
        /* Voltage */
        //float v;       /* Membrane voltage (mV) */
        //float vnew;    /* New Voltage (mV) */
        //float dvdt;    /* Change in Voltage / Change in Time (mV/ms) */
        //float dvdtnew; /* New dv/dt (mV/ms) */
        //float boolien; /* Boolien condition to test for dvdtmax */
        
        /* Time Step */
        //float dt;      /* Time step (ms) */
        //float t;       /* Time (ms) */
        //float udt;     /* Universal Time Step */
        //int utsc;       /* Universal Time Step Counter */
        //int nxstep;     /* Interval Between Calculating Ion Currents */
        //int steps;      /* Number of Steps */
        //int increment;  /* Loop Control Variable */ 
        
        /* Action Potential Duration and Max. Info */
        //float vmax[beats+1];           /* Max. Voltage (mV) */
        //float dvdtmax[beats+1];        /* Max. dv/dt (mV/ms) */
        //float apd[beats+1];            /* Action Potential Duration */
        //float toneapd[beats+1];        /* Time of dv/dt Max. */
        //float ttwoapd[beats+1];        /* Time of 90% Repolarization */
        //float trep[beats+1];           /* Time of Full Repolarization */
        //float di[beats+1];             /* Diastolic Interval */
        //float rmbp[beats+1];           /* Resting Membrane Potential */
        //float nair[beats+1];           /* Intracellular Na At Rest */
        //float cair[beats+1];           /* Intracellular Ca At Rest */
        //float caimax[beats+1];         /* Peak Intracellular Ca */
        //int i;                        /* Stimulus Counter */
        
        /* Total Current and Stimulus */
        //float st;       /* Constant Stimulus (uA/cm^2) */
        //float tstim;    /* Time Stimulus is Applied (ms) */
        //float stimtime; /* Time period during which stimulus is applied (ms) */
        float it;       /* Total current (uA/cm^2) */

        /* Terms for Solution of Conductance and Reversal Potential */
        float R = 8314;      /* Universal Gas Constant (J/kmol*K) */
        float frdy = 96485;  /* Faraday's Constant (C/mol) */
        float temp = 310;    /* Temperature (K) */

        /* Ion Valences */
        float zna = 1;  /* Na valence */
        float zk = 1;   /* K valence */
        float zca = 2;  /* Ca valence */

        /* Ion Concentrations */
        //float nai;    /* Intracellular Na Concentration (mM) */
        float nao;    /* Extracellular Na Concentration (mM) */
        //float ki;     /* Intracellular K Concentration (mM) */
        float ko;     /* Extracellular K Concentration (mM) */
        //float cai;    /* Intracellular Ca Concentration (mM) */
        float cao;    /* Extracellular Ca Concentration (mM) */
        float cmdn;   /* Calmodulin Buffered Ca Concentration (mM) */
        float trpn;   /* Troponin Buffered Ca Concentration (mM) */
        //float nsr;    /* NSR Ca Concentration (mM) */
        //float jsr;    /* JSR Ca Concentration (mM) */
        float csqn;   /* Calsequestrin Buffered Ca Concentration (mM) */

        /* Myoplasmic Na Ion Concentration Changes */
        float naiont;  /* Total Na Ion Flow (mM/ms) */
        float dnai;    /* Change in Intracellular Na Concentration (mM) */

        /* Myoplasmic K Ion Concentration Changes */
        float kiont; /* Total K Ion Flow (mM/ms) */
        float dki;   /* Change in Intracellular K Concentration (mM) */

        /* NSR Ca Ion Concentration Changes */
        float dnsr;   /* Change in [Ca] in the NSR (mM) */
        float iup;    /* Ca uptake from myo. to NSR (mM/ms) */
        float ileak;  /* Ca leakage from NSR to myo. (mM/ms) */
        float kleak;  /* Rate constant of Ca leakage from NSR (ms^-1) */
        float kmup = 0.00092;    /* Half-saturation concentration of iup (mM) */
        float iupbar = 0.005;  /* Max. current through iup channel (mM/ms) */
        float nsrbar = 15;       /* Max. [Ca] in NSR (mM) */
        
        /* JSR Ca Ion Concentration Changes */
        float djsr;                   /* Change in [Ca] in the JSR (mM) */
        //float urel;                   /* Activation gate u of Ca release from jsr*/

	float urelss;                 /* Steady state of activation gate u*/
	float tauurel;                /* Time constant of activation gate u*/
	//float vrel;                   /* Activation gate v of Ca release from jsr*/
	float vrelss;                 /* Steady state of activation gate v*/
	float tauvrel;                /* Time constant of activation gate v*/
	//float wrel;                   /* Inactivation gate w of Ca release from jsr*/
	float wrelss;                 /* Steady state of inactivation gate w*/
	float tauwrel;                /* Time constant of inactivation gate w*/
	float fn;
	float grelbarjsrol = 30; /* Rate constant of Ca release from JSR due to overload (ms^-1)*/
        float greljsrol;               /* Rate constant of Ca release from JSR due to CICR (ms^-1)*/
        float ireljsrol;               /* Ca release from JSR to myo. due to JSR overload (mM/ms)*/
        float csqnbar = 10;      /* Max. [Ca] buffered in CSQN (mM)*/
        float kmcsqn = 0.8;      /* Equalibrium constant of buffering for CSQN (mM)*/
        float bjsr;                    /* b Variable for analytical computation of [Ca] in JSR (mM)*/
        float cjsr;                    /* c Variable for analytical computation of [Ca] in JSR (mM)*/
        float on;                      /* Time constant of activation of Ca release from JSR (ms)*/
        float off;                     /* Time constant of deactivation of Ca release from JSR (ms)*/
        float magrel;                  /* Magnitude of Ca release*/

        /* Translocation of Ca Ions from NSR to JSR */
        //float itr;                /* Translocation current of Ca ions from NSR to JSR (mM/ms)*/
        float tautr = 180;  /* Time constant of Ca transfer from NSR to JSR (ms)*/
        
        /* Myoplasmic Ca Ion Concentration Changes */
        float caiont;  /* Total Ca Ion Flow (mM/ms) */
        float dcai;    /* Change in myoplasmic Ca concentration (mM) */
        float b1cai;

	float b2cai;

	float cmdnbar = 0.050;   /* Max. [Ca] buffered in CMDN (mM) */
        float trpnbar = 0.070;   /* Max. [Ca] buffered in TRPN (mM) */
        float kmcmdn = 0.00238;  /* Equalibrium constant of buffering for CMDN (mM) */
        float kmtrpn = 0.0005;   /* Equalibrium constant of buffering for TRPN (mM) */

        /* Fast Sodium Current (time dependant) */
        //float ina;    /* Fast Na Current (uA/uF) */
        float gna;    /* Max. Conductance of the Na Channel (mS/uF) */
        float ena;    /* Reversal Potential of Na (mV) */
        float ah;     /* Na alpha-h rate constant (ms^-1) */
        float bh;     /* Na beta-h rate constant (ms^-1) */
        float aj;     /* Na alpha-j rate constant (ms^-1) */
        float bj;     /* Na beta-j rate constant (ms^-1) */
        float am;     /* Na alpha-m rate constant (ms^-1) */
        float bm;     /* Na beta-m rate constant (ms^-1) */
        //float h;      /* Na activation */
        //float j;      /* Na inactivation */
        //float m;      /* Na inactivation */
        float gB;
	
        /* Current through L-type Ca Channel */
        //float ilca;    /* Ca current through L-type Ca channel (uA/uF) */
        //float ilcatot; /* Total current through the L-type Ca channel (uA/uF) */
        float ibarca;  /* Max. Ca current through Ca channel (uA/uF) */
        //float d;       /* Voltage dependant activation gate */
        float dss;     /* Steady-state value of activation gate d  */
        float taud;    /* Time constant of gate d (ms^-1) */
        //float f;       /* Voltage dependant inactivation gate */
        float fss;     /* Steady-state value of inactivation gate f */
        float tauf;    /* Time constant of gate f (ms^-1) */
        //float fca;     /* Ca dependant inactivation gate */
        float taufca;  /* Time constant of gate fca (ms^-1) */

	float fcass;   /* Steady-state value of activation gate fca  */
	float gcalbar = 0.1238;
    	/* Acetylcholine-Activated Potassium Current */
	/* modified from Matsuoka et al., Jap J Physiol 2003;53:105-123 */
	//float ikach; /* Acetylcholine-activated K current (uA/uF) */
	float gkach; /* Channel conductance of acetylcholine-activated K current (mS/uF) */
        float ekach; /* Reversal potential of acetylcholine-activated K current (mV) */
    	float alphayach; /* Alpha rate constant (ms^-1) */
        float betayach; /* Beta rate constant (ms^-1) */
        float tauyach; /* Time constant (ms) */
        float yachss; /* Steady-state value */
        //float yach;
        float ach = 0.0; /* Acetylcholine concentration */

        /* Ultra-Rapidly Activating Potassium Current */
        //float ikur;   /* Ultra-rapidly activating K current (uA/uF) */
        float gkur;   /* Channel conductance of ultra-rapidly activating K current (mS/uF) */
        float ekur;   /* Reversal potential of ultra-rapidly activating K current (mV) */
        //float uakur;    /* Ultra-rapidly activating K activation gate ua */
        float uakurss;  /* Steady-state value of activation gate ua */
        float tauuakur; /* Time constant of gate ua (ms^-1) */
        float alphauakur; /* Alpha rate constant of activation gate ua (ms^-1) */
	float betauakur;  /* Beta rate constant of activation gate ua (ms^-1) */
	//float uikur;    /* Ultra-rapidly activating K activation gate ui*/
        float uikurss;  /* Steady-state value of activation gate ui */
        float tauuikur; /* Time constant of gate ui (ms) */
	float alphauikur; /* Alpha rate constant of activation gate ui (ms^-1) */
	float betauikur; /* Beta rate constant of activation gate ui (ms^-1) */

	/* Rapidly Activating Potassium Current */
        //float ikr;   /* Rapidly activating K current (uA/uF) */
        float gkr;   /* Channel conductance of rapidly activating K current (mS/uF) */
        float ekr;   /* Reversal potential of rapidly activating K current (mV) */
        //float xr;    /* Rapidly activating K time-dependant activation */
        float xrss;  /* Steady-state value of inactivation gate xr */
        float tauxr; /* Time constant of gate xr (ms^-1) */
        float r;     /* K time-independant inactivation */
        
        /* Slowly Activating Potassium Current */
        //float iks;   /* Slowly activating K current (uA/uF) */
        float gks;   /* Channel conductance of slowly activating K current (mS/uF) */
        float eks;   /* Reversal potential of slowly activating K current (mV) */
        //float xs;    /* Slowly activating potassium current activation gate*/
        float xsss;  /* Steady-state value of activation gate xs */
        float tauxs; /* Time constant of gate xs (ms^-1) */
	float prnak = 0.01833;  /* Na/K Permiability Ratio */
        
        /* Time-Independent Potassium Current */
		/*Partly modified from Matsuoka, et al, Jap J Physiol,2003:53:105-123*/
        //float iki;    /* Time-independant K current (uA/uF) */
        float gki;    /* Channel conductance of time independant K current (mS/uF) */
        float eki;    /* Reversal potential of time independant K current (mV) */
        float kin;    /* K inactivation */
	float iku ;	/*Attaching rate constant of Magnesium to iki*/
	float ikl ;    /*Detaching rate constant of Magnesium to iki*/
	float ikay ;   /*Attaching rate constant of spermine to iki*/
	float ikby ;   /*Detaching rate constant of spermine to iki*/
	float tauiky ; /*Time constant of spermine attachment*/
	float ikyss ;  /*Steady state of spermine attachment*/
	float iky ;    /*Spermine attachment*/
	float foiki ;  /*Fraction of channel free from attachment of Magnesium*/
	float fbiki ;  /*Fraction of channel with attachment of Magnesium*/
        
        /* Transient Outward Potassium Current */
        //float ito;       /* Transient outward current */
        float gito;      /* Maximum conductance of Ito */
        float erevto;    /* Reversal potential of Ito */
        //float ato;       /* Ito activation */
        float alphaato;  /* Ito alpha-a rate constant */
        float betaato;   /* Ito beta-a rate constant */
        float tauato;    /* Time constant of a gate */
        float atoss;     /* Steady-state value of a gate */
        //float iito;      /* Ito inactivation */
        float alphaiito; /* Ito alpha-i rate constant */
        float betaiito;  /* Ito beta-i rate constant */
        float tauiito;   /* Time constant of i gate */
        float iitoss;    /* Steady-state value of i gate */
                
        /* Sodium-Calcium Exchanger */
        //float inaca;               /* NaCa exchanger current (uA/uF) */
	float kmnancx = 87.5;  /* Na saturation constant for NaCa exchanger */
	float ksatncx = 0.1;   /* Saturation factor for NaCa exchanger */
	float kmcancx = 1.38;  /* Ca saturation factor for NaCa exchanger */
        float gammas = 0.35;  /* Position of energy barrier controlling voltage dependance of inaca */

        /* Sodium-Potassium Pump */
        //float inak;    /* NaK pump current (uA/uF) */
        float fnak;    /* Voltage-dependance parameter of inak */
        float sigma;   /* [Na]o dependance factor of fnak */
        float ibarnak = 1.0933;   /* Max. current through Na-K pump (uA/uF) */
        float kmnai = 10;    /* Half-saturation concentration of NaK pump (mM) */
        float kmko = 1.5;    /* Half-saturation concentration of NaK pump (mM) */
        
        /* Sarcolemmal Ca Pump */
        //float ipca;                 /* Sarcolemmal Ca pump current (uA/uF) */
        float ibarpca = 0.275; /* Max. Ca current through sarcolemmal Ca pump (uA/uF) */
        float kmpca = 0.0005; /* Half-saturation concentration of sarcolemmal Ca pump (mM) */
        
        /* Ca Background Current */
        //float icab;  /* Ca background current (uA/uF) */
        float gcab;  /* Max. conductance of Ca background (mS/uF) */
        float ecan;  /* Nernst potential for Ca (mV) */

        /* Na Background Current */
        //float inab;  /* Na background current (uA/uF) */
        float gnab;  /* Max. conductance of Na background (mS/uF) */
        float enan;  /* Nernst potential for Na (mV) */
      
        /* Total Ca current */
        float icatot;

        /* Ion Current Functions */     
        float comp_ina (int x, int y, int z);    /* Calculates Fast Na Current */
        float comp_ical (int x, int y, int z);   /* Calculates Currents through L-Type Ca Channel */
        float comp_ikr (int x, int y, int z);    /* Calculates Rapidly Activating K Current */
        float comp_iks (int x, int y, int z);    /* Calculates Slowly Activating K Current */
        float comp_iki (int x, int y, int z);    /* Calculates Time-Independant K Current */
	float comp_ikach(int x, int y, int z);   /* Calculates Acetylcholine-sensitive potassium*/
	float comp_ikur (int x, int y, int z);   /* Calculates Ultra-Rapidly activation K Current*/
        float comp_ito (int x, int y, int z);    /* Calculates Transient Outward Current */
        float comp_inaca (int x, int y, int z);  /* Calculates Na-Ca Exchanger Current */
        float comp_inak (int x, int y, int z);   /* Calculates Na-K Pump Current */
        float comp_ipca (int x, int y, int z);   /* Calculates Sarcolemmal Ca Pump Current */
        float comp_icab (int x, int y, int z);   /* Calculates Ca Background Current */
        float comp_inab (int x, int y, int z);   /* Calculates Na Background Current */
        float comp_it (int x, int y, int z);     /* Calculates Total Current */

        /* Ion Concentration Functions */
        float conc_nai (int x, int y, int z);    /* Calculates new myoplasmic Na ion concentration */
        float conc_ki (int x, int y, int z);     /* Calculates new myoplasmic K ion concentration */
        float conc_nsr (int x, int y, int z);    /* Calculates new NSR Ca ion concentration */
        float conc_jsr (int x, int y, int z);    /* Calculates new JSR Ca ion concentration */
        float calc_itr (int x, int y, int z);    /* Calculates Translocation of Ca from NSR to JSR */
        float conc_cai (int x, int y, int z);    /* Calculates new myoplasmic Ca ion concentration */
//////////////////      Courtemanche  cell model parameters ///////////////////

float court (int x, int y, int z)
{
        /* Cell Geometry */
        vcell = 1000*pi*a*a*l;     /*   3.801e-5 uL */
        ageo = 2*pi*a*a+2*pi*a*l;

	acap = ageo*2;             /*   1.534e-4 cm^2 */
        vmyo = vcell*0.68;
        vmito = vcell*0.26;
        vsr = vcell*0.06;
        vnsr = vcell*0.0552;
        vjsr = vcell*0.0048;
        /* Time Loop Conditions */
        //t = 0;           /* Time (ms) */
        //udt = 0.01;     /* Time step (ms) */
        //steps = (S2 + bcl*beats)/udt; /* Number of ms */
        //st = -200;        /* Stimulus */
        //tstim = 10;       /* Time to begin stimulus */
        //stimtime = 10;   /* Initial Condition for Stimulus */
        //v = -81.2;       /* Initial Voltage (mv) */

        /* Beginning Ion Concentrations */
        //nai = 11.2;       /* Initial Intracellular Na (mM) */
        nao = 140;      /* Initial Extracellular Na (mM) */
        //ki = 139;       /* Initial Intracellular K (mM) */
        ko = 4.5;       /* Initial Extracellular K (mM) */
        //cai = 0.000102;  /* Initial Intracellular Ca (mM) */
        cao = 1.8;      /* Initial Extracellular Ca (mM) */

        /* Initial Gate Conditions */
        //m = 0.00291;
        //d = 0.000137;
        //f = 0.999837;
	//xs = 0.0187;
        //xr = 0.0000329;
	//ato = 0.0304;
	//iito = 0.999;
	//uakur = 0.00496;
	//uikur = 0.999;
	//fca = 0.775;
	ireljsrol=0;
        
        /* Initial Conditions */
        //jsr = 1.49;
        //nsr = 1.49;
	trpn = 0.0118;
        cmdn = 0.00205;
        csqn = 6.51;
        //boolien = 1;
        //dt = udt;
        //utsc = 50;
	//urel = 0.00;
	//vrel = 1.00;
	//wrel = 0.999;
	//yach = 2.54e-2;
        iky = 0.6;
	//i=-1;
        /* List of functions called for each timestep, currents commented out are only used when modeling pathological conditions */
        comp_ina (x, y, z);
        comp_ical (x, y, z);
        comp_ikr (x, y, z);
        comp_iks (x, y, z);
        comp_iki (x, y, z);

	comp_ikach (x, y, z);
        comp_ikur (x, y, z);
        comp_ito (x, y, z); 
        comp_inaca (x, y, z);
        comp_inak (x, y, z);
        comp_ipca (x, y, z);
        comp_icab (x, y, z);
        comp_inab (x, y, z);
        comp_it (x, y, z);
                
        conc_nai (x, y, z);
        conc_ki (x, y, z);
        conc_nsr (x, y, z);
        conc_jsr (x, y, z);
        calc_itr (x, y, z);
        conc_cai (x, y, z);
/*      utsc = 0;
        dt = udt;
        vnew = v-it*udt;
        dvdtnew = (vnew-v)/udt;
                if (vnew>vmax[i])
                        vmax[i] = vnew;
                if (cai[x][y][z]>caimax[i])
                        caimax[i] = cai[x][y][z];
                if (dvdtnew>dvdtmax[i])
                        {dvdtmax[i] = dvdtnew;
                        toneapd[i] = t;}
                if (vnew>=(vmax[i]-0.9*(vmax[i]-rmbp[i])))
                        ttwoapd[i] = t;
                 if (vnew>=(vmax[i]-0.98*(vmax[i]-rmbp[i])))
                        trep[i] = t;
*/

 //       v = vnew;
 //       utsc = utsc+1;      
 //       t = t+udt;     
       return (-(it));
}

int main(int argc, char **argv)
//int main()
{ 
	int x, y, z;
	int cnt = 0;
	int snp = 0;
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
        int tstim = 0;
        int delt = 10;
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
        //float temp; 
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
                                        //c = 1; 
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
/*
     in = fopen ("heterogeneity", "r");	
	for (z = 0; z < Z; z++) { 
      for (y = 0; y < Y; y++) { 
		for (x = 0; x < X; x++) {
		  fscanf(in, "%d ", &dd);

            g[x][y][z] = dd;     
	        if (z < 3 || z > Z-3)
             g[x][y][z] = 0;
		}
        fscanf(in, "\n"); 
      } 
      fscanf(in, "\n"); 
    }
    fclose (in);
*/
//       in = fopen ("Fibers.dat", "r");	
//	for (z = 0; z < Z; z++) 
 //     for (y = 0; y < Y; y++)  
//	      for (x = 0; x < X; x++) {
//		      fscanf(in, "%f %f %f\n", &gx, &gy, &gz);
//		      xx[x][y][z] = gx;
//		      yy[x][y][z] = gy;
//		      zz[x][y][z] = gz;
//	      }
//	fclose (in);
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
for (x = 0; x <= X; x++)
    for (y = 0; y <= Y; y++) 
    	for (z = 0; z <= Z; z++) {
      u[x][y][z] = -81.2;
      hh[x][y][z] = 0.965;
      jj[x][y][z] = 0.978;
      m[x][y][z] = 0.00291;
      d[x][y][z] = 0.000137;
      f[x][y][z] = 0.999837;
      ina[x][y][z] = 0.0;
      nai[x][y][z] = 11.2;
      cai[x][y][z] = 0.000102;
      fca[x][y][z] = 0.775;
      ilca[x][y][z] = 0.0;
      xr[x][y][z] = 0.0000329;
      xs[x][y][z] = 0.0187;
      ikr[x][y][z] = 0.0;
      iks[x][y][z] = 0.0;
      iki[x][y][z] = 0.0;
      yach[x][y][z] = 2.54e-2;
      ikach[x][y][z] = 0.0;
      uakur[x][y][z] = 0.00496;
      uikur[x][y][z] = 0.999;
      ikur[x][y][z] = 0.0;
      ato[x][y][z] = 0.0304;
      iito[x][y][z] = 0.999;
      ito[x][y][z] = 0.0;
      inaca[x][y][z] = 0.0;
      inak[x][y][z] = 0.0;
      ipca[x][y][z] = 0.0;
      icab[x][y][z] = 0.0;
      inab[x][y][z] = 0.0;
      ki[x][y][z] = 139;
      nsr[x][y][z] = 1.49;
      urel[x][y][z] = 0.00;
      vrel[x][y][z] = 1.00;
      wrel[x][y][z] = 0.999; 
      jsr[x][y][z] = 1.49;
      itr[x][y][z] = 0.0;
    }
/*
in = fopen ("Results/SaveU218.vtk", "r");
for (z = 0; z < Z; z++) {
    for (y = 0; y < Y; y++) {
        for (x = 0; x < X; x++)
                {
                     fscanf(in, "%f ", &temp);
                     u[x][y][z] = temp/100.0;
                     }
                    fscanf(in, "\n");
                  }
                  fscanf(in, "\n");
}
fclose (in);
in = fopen ("Results/SaveV218.vtk", "r");
for (x = 0; x <= X; x++) {
    for (y = 0; y <= Y; y++) {
        for (z = 0; z < Z; z++)
                {
                     fscanf(in, "%f ", &temp);
                     v[x][y][z] = temp;
                     }
                    fscanf(in, "\n");
                  }
                  fscanf(in, "\n");
}
fclose (in);
in = fopen ("Results/SaveW218.vtk", "r");
for (x = 0; x <= X; x++) {
    for (y = 0; y <= Y; y++) {
          for (z = 0; z <= Z; z++)
                 {
                     fscanf(in, "%f ", &temp);
                     w[x][y][z] = temp;
                     }
                    fscanf(in, "\n");
                  }
                  fscanf(in, "\n");
}
fclose (in);
*/
}

 MPI_Bcast (u, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (g, (X+1)*(Y+1)*(Z+1), MPI_INT, 0, MPI_COMM_WORLD);
 MPI_Bcast (h, (X+1)*(Y+1)*(Z+1), MPI_INT, 0, MPI_COMM_WORLD);
 MPI_Bcast (xx, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (yy, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (zz, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (hh, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (m, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (jj, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (ina, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (nai, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (cai, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (d, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (f, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (fca, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (ilca, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (xr, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (ikr, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (xs, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (iks, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (iki, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (yach, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (ikach, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (uakur, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (uikur, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (ikur, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (ato, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (iito, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (ito, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (inaca, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (inak, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (ipca, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (icab, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (inab, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (nai, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (ki, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (nsr, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (urel, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (vrel, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (wrel, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (jsr, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
 MPI_Bcast (itr, (X+1)*(Y+1)*(Z+1), MPI_FLOAT, 0, MPI_COMM_WORLD);

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

while (t < 1000){

/********************************************************* Set the initial Simuli *********************************************/
//////t=0;tstim=0;rank=0;dt=0.0025;counter=500;t+=dt;u=-81.2;
	
	// apply stimulus at certain times
	//if ((t>=tstim) && (t<(tstim + 0.05))) {
	if ((t>=0) && (t<(0 + 0.05))) {
		
		//if ((t>=tstim) && (t<(tstim + dt)))  {
		//	if (!rank) {
		//		out = fopen ("./paraStage.dat", "a");
		//		fprintf (out, "%d %d %d %f\n", delt, counter, tstim, t);
		//		fclose (out);
		//	}
		//}

		/*//set focal source
		for (x = 30; x <= 40; x++)
			for (y = 100; y <= 120; y++)
				for (z = 1; z <= 5; z++)
					{if (h[x][y][z] > 0)
						u[x][y][z] = 20;}*/
		
		// set wave front
		for (x = 0; x < 500; x++)
			for (y = 0; y < 10; y++)
				for (z = 0; z < 5; z++)
					{if (h[x][y][z] > 0)
						u[x][y][z] = 20;}
		
		// set the time intervals to apply the stimulus after a few timesteps after the first stimulus is applied
		if ((t>=(tstim + 0.05 - dt)) && (t<(tstim + 0.05))){
			if (counter <= 500){
				delt = 50;
			}
			if (counter <= 300){
				delt = 10;
			}
			counter = counter - delt;
			tstim = tstim + counter;
		}
	}
	
	// S2 protocol 
	// experiment: conduction velocity = 0.05, square stimulus at 220
	if ((t>=220) && (t<(220+0.01))){
		for (x = 0; x < 250; x++)
			for (y = 0; y < 250; y++)
				for (z = 0; z < 5; z++)
					{u[x][y][z] = 10;}
	}
	
	/*// set rotor by deleting area
	// have to make sure the wave front and tail are short enough to form rotor...
	if ((t>=300) && (t<(300+0.05))){
		for (x = 1; x <= 120; x++)
			for (y = 1; y <= 240; y++)
				for (z = 1; z <= 5; z++)
					{u[x][y][z] = -81.2;}
	}*/
	
/*
        str = malloc (50*sizeof(char));
        sprintf (str, "Results/SaveU%d.vtk", cnt);
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
        str = malloc (50*sizeof(char));
        sprintf (str, "Results/SaveV%d.vtk", cnt);
        out = fopen (str, "wt");
        for (x = 0; x <= X; x++) {
          for (y = 0; y <= Y; y++) {
            for (z = 0; z < Z; z++)
              fprintf (out, "%3.1f ", global_v[x][y][z]);
            fprintf (out, "\n");
          }
          fprintf (out, "\n");
            }
        fclose (out);
        free (str);
*/
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

				fu = court (x, y, z); // update CRN model parameters
				new_u[x][y][z] = u[x][y][z] + dt * (du + fu);
			  }
			}
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
      MPI_Send(hh[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(hh[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(hh[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(hh[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

    if (rank < (size-1))
      MPI_Send(jj[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(jj[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(jj[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(jj[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(m[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(m[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(m[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(m[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(d[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(d[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(d[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(d[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(f[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(f[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(f[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(f[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(ina[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(ina[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(ina[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(ina[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(nai[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(nai[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(nai[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(nai[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(cai[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(cai[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(cai[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(cai[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(fca[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(fca[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(fca[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(fca[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(ilca[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(ilca[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(ilca[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(ilca[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(xr[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(xr[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(xr[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(xr[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(xs[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0) 
      MPI_Recv(xs[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(xs[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(xs[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(ikr[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(ikr[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(ikr[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(ikr[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(iks[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(iks[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(iks[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(iks[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(iki[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(iki[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(iki[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(iki[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(yach[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(yach[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(yach[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(yach[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(ikach[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(ikach[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(ikach[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(ikach[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(uakur[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(uakur[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(uakur[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(uakur[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(uikur[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(uikur[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(uikur[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(uikur[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(ikur[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(ikur[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(ikur[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(ikur[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(ato[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(ato[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(ato[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(ato[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(iito[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(iito[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(iito[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(iito[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(ito[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(ito[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(ito[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(ito[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(inaca[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(inaca[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(inaca[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(inaca[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(inak[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(inak[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(inak[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(inak[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(ipca[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(ipca[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(ipca[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(ipca[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(icab[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(icab[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(icab[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(icab[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(inab[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(inab[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(inab[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(inab[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(ki[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(ki[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(ki[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(ki[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

   if (rank < (size-1))
      MPI_Send(nsr[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(nsr[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(nsr[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(nsr[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

  if (rank < (size-1))
      MPI_Send(urel[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(urel[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(urel[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(urel[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

 if (rank < (size-1))
      MPI_Send(vrel[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(vrel[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(vrel[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(vrel[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

 if (rank < (size-1))
      MPI_Send(wrel[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(wrel[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(wrel[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(wrel[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

 if (rank < (size-1))
      MPI_Send(jsr[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(jsr[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(jsr[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(jsr[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

 if (rank < (size-1))
      MPI_Send(itr[BLOCK_HIGH(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    if (rank > 0)
      MPI_Recv(itr[BLOCK_HIGH((rank-1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
    if (rank > 0)
      MPI_Send(itr[BLOCK_LOW(rank, size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
    if (rank < (size-1))
      MPI_Recv(itr[BLOCK_LOW((rank+1), size, X+1)], (Y+1)*(Z+1), MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);

  if (snp % 400 == 0) {  
MPI_Gatherv (&(((float *)u)[recv_disp[rank]]), recv_cnts[rank], MPI_FLOAT, &global_u[0][0][0], recv_cnts, recv_disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Gatherv (&(((float *)ina)[recv_disp[rank]]), recv_cnts[rank], MPI_FLOAT, &global_ina[0][0][0], recv_cnts, recv_disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Gatherv (&(((float *)ilca)[recv_disp[rank]]), recv_cnts[rank], MPI_FLOAT, &global_ilca[0][0][0], recv_cnts, recv_disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    if (!rank) {
		if (snp % 400 == 0) {
			
			str = malloc (40*sizeof(char));
			cnt++;
			sprintf (str, "TestCRN/aa%04d.vtk", cnt);
			out = fopen (str, "wt");
			for (z = 0; z < Z; z++) {
				for (y = 0; y < Y; y++) {
					for (x = 0; x < X; x++)
						fprintf (out, "%2.1f ", global_u[x][y][z]);
						fprintf (out, "\n");
				}
				fprintf (out, "\n");
			}          	
			fclose (out);
			free (str);
		
if (1 == 0) { 
         str = malloc (40*sizeof(char));
         sprintf (str, "Fibrosis/ina%d.vtk", cnt);
         out = fopen (str, "wt");
         for (z = 0; z < Z; z++) {
             for (y = 0; y < Y; y++) {
                 for (x = 0; x < X; x++)
                     fprintf (out, "%2.6f ", global_ina[x][y][z]);
                 fprintf (out, "\n");
                  }
                 fprintf (out, "\n");
          }
          fclose (out);
          free (str);

         str = malloc (40*sizeof(char));
         sprintf (str, "Fibrosis/ilca%d.vtk", cnt);
         out = fopen (str, "wt");
         for (z = 0; z < Z; z++) {
             for (y = 0; y < Y; y++) {
                 for (x = 0; x < X; x++)
                     fprintf (out, "%2.6f ", global_ilca[x][y][z]);
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
 
  if (recv_cnts) free (recv_cnts) ;
  if (recv_disp) free (recv_disp) ;
  MPI_Finalize() ;
 
  return 0 ;
}

/////////////////////////////////////////////////////* Functions that describe the currents begin here */
float comp_ina (int x, int y, int z)
{
        gna = 7.8;
        ena = ((R*temp)/frdy)*log(nao/nai[x][y][z]);

        am = 0.32*(u[x][y][z]+47.13)/(1-exp(-0.1*(u[x][y][z]+47.13)));
        bm = 0.08*exp(-u[x][y][z]/11);
        
        if (u[x][y][z] < -40) 
        {ah = 0.135*exp((80+u[x][y][z])/-6.8);  
        bh = 3.56*exp(0.079*u[x][y][z])+310000*exp(0.35*u[x][y][z]);
        aj = (-127140*exp(0.2444*u[x][y][z])-0.00003474*exp(-0.04391*u[x][y][z]))*((u[x][y][z]+37.78)/(1+exp(0.311*(u[x][y][z]+79.23))));
        bj = (0.1212*exp(-0.01052*u[x][y][z]))/(1+exp(-0.1378*(u[x][y][z]+40.14)));}

        else  
        {ah = 0.0;
        bh = 1/(0.13*(1+exp((u[x][y][z]+10.66)/-11.1)));
        aj = 0.0;
        bj = (0.3*exp(-0.0000002535*u[x][y][z]))/(1+exp(-0.1*(u[x][y][z]+32)));}

        hh[x][y][z] = ah/(ah+bh)-((ah/(ah+bh))-hh[x][y][z])*exp(-dt/(1/(ah+bh)));
        jj[x][y][z] = aj/(aj+bj)-((aj/(aj+bj))-jj[x][y][z])*exp(-dt/(1/(aj+bj)));
        m[x][y][z] = am/(am+bm)-((am/(am+bm))-m[x][y][z])*exp(-dt/(1/(am+bm)));
        
        ina[x][y][z] = gna*m[x][y][z]*m[x][y][z]*m[x][y][z]*hh[x][y][z]*jj[x][y][z]*(u[x][y][z]-ena);
}

float comp_ical (int x, int y, int z)
{
        dss = 1/(1+exp(-(u[x][y][z]+10)/8));
        if (fabs(u[x][y][z]+10) > 0.00001)
         {   taud = (1-exp((u[x][y][z]+10)/-6.24))/(0.035*(u[x][y][z]+10)*(1+exp((u[x][y][z]+10)/-6.24))); }
        else
        {taud = (1/6.24 - (u[x][y][z]+10)/6.24/6.24/2)/(0.035*(1+exp((u[x][y][z]+10)/-6.24)));
        ;} 
        fss = 1/(1+exp((u[x][y][z]+28)/6.9));
        tauf = 9/(0.0197*exp(-pow((0.0337*(u[x][y][z]+10)),2))+0.02);
	fcass = 1/(1+cai[x][y][z]/0.00035);

	taufca = 2;
        d[x][y][z] = dss-(dss-d[x][y][z])*exp(-dt/taud);
        f[x][y][z] = fss-(fss-f[x][y][z])*exp(-dt/tauf);
	fca[x][y][z] = fcass-(fcass-fca[x][y][z])*exp(-dt/tauf);
	ibarca = gcalbar*(u[x][y][z]-65);                
        ilca[x][y][z] = d[x][y][z]*f[x][y][z]*fca[x][y][z]*ibarca;
        //ilcatot = ilca[x][y][z];
}

float comp_ikr (int x, int y, int z)
{
        gkr = 0.0294*sqrt(ko/5.4);
        ekr = ((R*temp)/frdy)*log(ko/ki[x][y][z]);

        xrss = 1/(1+exp(-(u[x][y][z]+14.1)/6.5));
        tauxr = 1/(0.0003*(u[x][y][z]+14.1)/(1-exp(-(u[x][y][z]+14.1)/5))+0.000073898*(u[x][y][z]-3.3328)/(exp((u[x][y][z]-3.3328)/5.1237)-1));
        xr[x][y][z] = xrss-(xrss-xr[x][y][z])*exp(-dt/tauxr);
        r = 1/(1+exp((u[x][y][z]+15)/22.4));
        ikr[x][y][z] = gkr*xr[x][y][z]*r*(u[x][y][z]-ekr);
}

float comp_iks (int x, int y, int z)
{
        gks = 0.129;
        eks = ((R*temp)/frdy)*log(ko/ki[x][y][z]);
	tauxs = 0.5/(0.00004*(u[x][y][z]-19.9)/(1-exp(-(u[x][y][z]-19.9)/17))+0.000035*(u[x][y][z]-19.9)/(exp((u[x][y][z]-19.9)/9)-1));
	xsss = 1/pow((1+exp(-(u[x][y][z]-19.9)/12.7)),0.5);
	xs[x][y][z] = xsss-(xsss-xs[x][y][z])*exp(-dt/tauxs);
	iks[x][y][z] = gks*xs[x][y][z]*xs[x][y][z]*(u[x][y][z]-eks);
}

float comp_iki (int x, int y, int z)
{
        gki = 0.09*pow(ko/5.4,0.4);
        eki = ((R*temp)/frdy)*log(ko/ki[x][y][z]);
        kin = 1/(1+exp(0.07*(u[x][y][z]+80)));
        iki[x][y][z] = gki*kin*(u[x][y][z]-eki);
/*
  // modified from Matsuoka, et al Jap J Physiol 2003:53:105-123
	iku = 0.75*exp(0.035*(v-eki-10))/(1+exp(0.015*(v-eki-140)));
	ikl = 3*exp(-0.048*(v-eki-10))*(1+exp(0.064*(v-eki-38)))/(1+exp(0.03*(v-eki-70)));
	ikay =1/(8000*exp((v-eki-97)/8.5)+7*exp((v-eki-97)/300));
	ikby =1/(0.00014*exp(-(v-eki-97)/9.1)+0.2*exp(-(v-eki-97)/500));
	tauiky = 1/(ikay+ikby);
	ikyss = ikay/(ikay+ikby);
	iky = ikyss - (ikyss-iky)*exp(-dt/tauiky);
	foiki = ikl/(iku+ikl);
	fbiki = iku/(iku+ikl);
	iki = gki*(pow(foiki,4)+8*pow(foiki,3)*fbiki/3+2*foiki*foiki*fbiki*fbiki)*iky*(v-eki); 

*/
  }

float comp_ikach (int x, int y, int z)
{

		gkach = 0.135;
		ekach = ((R*temp)/frdy)*log(ko/ki[x][y][z]);
		alphayach= 1.232e-2/(1+0.0042/ach)+0.0002475;
		betayach = 0.01*exp(0.0133*(u[x][y][z]+40));
		tauyach = 1/(alphayach+betayach);
		yachss = alphayach/(alphayach+betayach);
		yach[x][y][z] = yachss-(yachss-yach[x][y][z])*exp(-dt/tauyach);
		ikach[x][y][z] = gkach*yach[x][y][z]*(u[x][y][z]-ekach)/(1+exp((u[x][y][z]+20)/20));
}

float comp_ikur (int x, int y, int z)
{

		gkur = 0.005+0.05/(1+exp(-(u[x][y][z]-15)/13));
                ekur = ((R*temp)/frdy)*log(ko/ki[x][y][z]);
		alphauakur = 0.65/(exp(-(u[x][y][z]+10)/8.5)+exp(-(u[x][y][z]-30)/59.0));
		betauakur = 0.65/(2.5+exp((u[x][y][z]+82)/17.0));
		tauuakur = 1/(3*(alphauakur+betauakur));
		uakurss = 1/(1+exp(-(u[x][y][z]+30.3)/9.6));
		alphauikur = 1/(21+exp(-(u[x][y][z]-185)/28));
		betauikur = exp((u[x][y][z]-158)/16);
		tauuikur = 1/(3*(alphauikur+betauikur));
		uikurss = 1/(1+exp((u[x][y][z]-99.45)/27.48));
		uakur[x][y][z] = uakurss-(uakurss-uakur[x][y][z])*exp(-dt/tauuakur);
		uikur[x][y][z] = uikurss-(uikurss-uikur[x][y][z])*exp(-dt/tauuikur);
		ikur[x][y][z] = gkur*uakur[x][y][z]*uakur[x][y][z]*uakur[x][y][z]*uikur[x][y][z]*(u[x][y][z]-ekur);
}


float comp_ito (int x, int y, int z)
{
        gito = 0.1652;
        erevto = ((R*temp)/frdy)*log(ko/ki[x][y][z]);
        alphaato = 0.65/(exp(-(u[x][y][z]+10)/8.5)+exp(-(u[x][y][z]-30)/59));
        betaato = 0.65/(2.5+exp((u[x][y][z]+82)/17));
        tauato = 1/(3*(alphaato+betaato));
        atoss = 1/(1+exp(-(u[x][y][z]+20.47)/17.54));
        ato[x][y][z] = atoss-(atoss-ato[x][y][z])*exp(-dt/tauato);
        alphaiito = 1/(18.53+exp((u[x][y][z]+113.7)/10.95));
        betaiito = 1/(35.56+exp(-(u[x][y][z]+1.26)/7.44));
        tauiito = 1/(3*(alphaiito+betaiito));
        iitoss = 1/(1+exp((u[x][y][z]+43.1)/5.3));
        iito[x][y][z] = iitoss-(iitoss-iito[x][y][z])*exp(-dt/tauiito);
        ito[x][y][z] = gito*ato[x][y][z]*ato[x][y][z]*ato[x][y][z]*iito[x][y][z]*(u[x][y][z]-erevto);
}

float comp_inaca (int x, int y, int z)
{
	inaca[x][y][z] = 1750*(exp(gammas*frdy*u[x][y][z]/(R*temp))*nai[x][y][z]*nai[x][y][z]*nai[x][y][z]*cao-exp((gammas-1)*frdy*u[x][y][z]/(R*temp))*nao*nao*nao*cai[x][y][z])/((pow(kmnancx,3)+pow(nao,3))*(kmcancx+cao)*(1+ksatncx*exp((gammas-1)*frdy*u[x][y][z]/(R*temp))));

}

float comp_inak (int x, int y, int z)
{
        sigma = (exp(nao/67.3)-1)/7;

        //fnak = 1/(1+0.1245*exp((-0.1*v*frdy)/(R*temp))+0.0365*sigma*exp((-v*frdy)/(R*temp)));
	fnak=(u[x][y][z]+150)/(u[x][y][z]+200);
        inak[x][y][z] = ibarnak*fnak*(1/(1+pow((kmnai/nai[x][y][z]),1.5)))*(ko/(ko+kmko));
}

float comp_ipca (int x, int y, int z)
{
        ipca[x][y][z] = (ibarpca*cai[x][y][z])/(kmpca+cai[x][y][z]);       
}

float comp_icab (int x, int y, int z)
{
        gcab = 0.00113;
        ecan = ((R*temp)/frdy)*log(cao/cai[x][y][z]);
        icab[x][y][z] = gcab*(u[x][y][z]-ecan);
}

float comp_inab (int x, int y, int z)
{
        gnab = 0.000674;
        enan = ((R*temp)/frdy)*log(nao/nai[x][y][z]);
        inab[x][y][z] = gnab*(u[x][y][z]-enan);
}

/* Total sum of currents is calculated here, if the time is between stimtime = 0 and stimtime = 0.5, a stimulus is applied */
float comp_it (int x, int y, int z)
{
        naiont = ina[x][y][z]+inab[x][y][z]+3*inak[x][y][z]+3*inaca[x][y][z]+1.5e-2;
        kiont = ikr[x][y][z]+iks[x][y][z]+iki[x][y][z]-2*inak[x][y][z]+ito[x][y][z]+ikur[x][y][z]+ikach[x][y][z]+1.5e-2;
        caiont = ilca[x][y][z]+icab[x][y][z]+ipca[x][y][z]-2*inaca[x][y][z];
        it = naiont+kiont+caiont;

}

/* Functions that calculate intracellular ion concentrations begins here */

float conc_nai (int x, int y, int z)
{
        dnai = -dt*naiont*acap/(vmyo*zna*frdy);
        nai[x][y][z] = dnai + nai[x][y][z];
}

float conc_ki (int x, int y, int z)
{
        dki = -dt*kiont*acap/(vmyo*zk*frdy);
        ki[x][y][z] = dki + ki[x][y][z];
}

float conc_nsr (int x, int y, int z)
{
        kleak = iupbar/nsrbar;
        ileak = kleak*nsr[x][y][z];
        iup = iupbar*cai[x][y][z]/(cai[x][y][z]+kmup);
	csqn = csqnbar*(jsr[x][y][z]/(jsr[x][y][z]+kmcsqn));
	dnsr = dt*(iup-ileak-itr[x][y][z]*vjsr/vnsr);
        nsr[x][y][z] = dnsr+nsr[x][y][z];
}

float conc_jsr (int x, int y, int z)
{
//		fn = vjsr*(1e-12)*ireljsrol-(5e-13)*(iiontlca/2+inaca/5)*acap/frdy; 
	fn = vjsr*(1e-12)*ireljsrol-(1e-12)*caiont*acap/(2*frdy); 
	tauurel = 8.0;
	urelss = 1/(1+exp(-(fn-3.4175e-13)/13.67e-16));
	tauvrel = 1.91+2.09/(1+exp(-(fn-3.4175e-13)/13.67e-16));
	vrelss = 1-1/(1+exp(-(fn-6.835e-14)/13.67e-16));
	tauwrel = 6.0*(1-exp(-(u[x][y][z]-7.9)/5))/((1+0.3*exp(-(u[x][y][z]-7.9)/5))*(u[x][y][z]-7.9));
	wrelss = 1-1/(1+exp(-(u[x][y][z]-40)/17));
	urel[x][y][z] = urelss-(urelss-urel[x][y][z])*exp(-dt/tauurel);
	vrel[x][y][z] = vrelss-(vrelss-vrel[x][y][z])*exp(-dt/tauvrel);
	wrel[x][y][z] = wrelss-(wrelss-wrel[x][y][z])*exp(-dt/tauwrel);
        greljsrol = grelbarjsrol*urel[x][y][z]*urel[x][y][z]*vrel[x][y][z]*wrel[x][y][z];
        ireljsrol = greljsrol*(jsr[x][y][z]-cai[x][y][z]); 
	djsr = dt*(itr[x][y][z]-0.5*ireljsrol)/(1+csqnbar*kmcsqn/pow((jsr[x][y][z]+kmcsqn),2)); //LAI
        jsr[x][y][z] = djsr+jsr[x][y][z];
}

float calc_itr (int x, int y, int z)
{
        itr[x][y][z] = (nsr[x][y][z]-jsr[x][y][z])/tautr;
}

float conc_cai (int x, int y, int z)
{
        trpn = trpnbar*(cai[x][y][z]/(cai[x][y][z]+kmtrpn));
        cmdn = cmdnbar*(cai[x][y][z]/(cai[x][y][z]+kmcmdn));
	b1cai = -caiont*acap/(2*frdy*vmyo)+(vnsr*(ileak-iup)+0.5*ireljsrol*vjsr)/vmyo; //LAI
	b2cai = 1+trpnbar*kmtrpn/pow((cai[x][y][z]+kmtrpn),2)+cmdn*kmcmdn/pow((cai[x][y][z]+kmcmdn),2);
	dcai = dt*b1cai/b2cai;
        cai[x][y][z] = dcai+cai[x][y][z];
}

/* Values are printed to a file called ap. The voltage and currents can be plotted versus time using graphing software. */
/*
void prttofile ()
{
if(increment%100 == 0)
   fprintf (ap, "%.3f\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n", t, v, ikr, ilca, iks, ito, cai, iki, ina);
}
*/


