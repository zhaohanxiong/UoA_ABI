%%========= Time discretization
 
ht=0.005;            % Delta_t
t = 0:ht:1000;        % Time vector

%%============ Parameters

% Model constants

C_m = 1;
V_0 = -83.91; % Adjusted resting potential
V_fi = 3.6;

tau_d = 0.125;
tau_r = 70;
tau_si = 114;
tau_0 = 32.5;
tau_v_plus = 5.75;
tau_v1_minus = 82.5;
tau_v2_minus = 60;
tau_w_plus = 300;
tau_w_minus = 400;

u_c = 0.16;
u_v = 0.04;
u_csi = 0.85;

k = 10;


%%============ Initialize variables

% Gating variables

u = zeros(length(t),1);
v = zeros(length(t),1);
w = zeros(length(t),1);

% Currents

J_fi = zeros(length(t)-1,1);
J_so = zeros(length(t)-1,1);
J_si = zeros(length(t)-1,1);

% Membrane potential

Vm = zeros(length(t)-1,1);
%% ============== Stimulus

J_stim = zeros(length(t)-1,1);

% J_stim(20001:20401) = -0.2; % Stimulus set-up used in paper

J_stim(10001:10401) = -0.2; % For report

%%========== Initial conditions

% Initial conditions used in paper

u(1,1) = 1.18*10^-8;
v(1,1) = 1;
w(1,1) = 1;


%%========== Computation

for ite = 1:length(t)-1

	% Component: fast inward current
	J_fi(ite,1) = ((-v(ite,1))*heaviside(u(ite,1)-u_c)*(1-u(ite,1))*(u(ite,1)-u_c))/(tau_d);

	% Component: slow outward current
	J_so(ite,1) = ((u(ite,1)*(heaviside(u_c-u(ite,1))))/(tau_0)) + ((heaviside(u(ite,1)-u_c))/(tau_r));

	% Component: slow inward current
	J_si(ite,1) = ((-w(ite,1))*(1+tanh(k*(u(ite,1)-u_csi))))/(2*tau_si);

	% Component: fast inward current v gate
	tau_v_minus = heaviside(u(ite,1)-u_v)*tau_v1_minus + (heaviside(u_v-u(ite,1)))*tau_v2_minus;

	% Numerical integration using forward euler
	u(ite+1,1) = u(ite,1) + ht*(-(J_fi(ite,1)+J_so(ite,1)+J_si(ite,1)+J_stim(ite,1)));
	v(ite+1,1) = v(ite,1) +  ht*((((heaviside(u_c-u(ite,1)))*(1-v(ite,1)))/(tau_v_minus)) -  ((heaviside(u(ite,1)-u_c)*v(ite,1))/(tau_v_plus)));
	w(ite+1,1) = w(ite,1) +  ht*((((heaviside(u_c-u(ite,1)))*(1-w(ite,1)))/(tau_w_minus)) -  ((heaviside(u(ite,1)-u_c)*w(ite,1))/(tau_w_plus)));

	% Membrane potential
	Vm(ite,1) = V_0 + u(ite,1)*(V_fi-V_0);

end

Andy_Vm = Vm;

figure(3)
plot(t(1,1:length(t)-1),Vm,'b','LineWidth',5)
ylabel('\fontsize{20} V(mV)')
xlabel('\fontsize{20} Time(ms)')
title('\fontsize{20} Time course of AP in Fenton-Karma model')
set(gca,'fontsize',20)

figure(4)
plot(t,u,'g','LineWidth',5)
hold on
plot(t,v,'y','LineWidth',5)
plot(t,w,'r','LineWidth',5)
ylabel('\fontsize{20} Open probability')
xlabel('\fontsize{20} Time(ms)')
legend('\fontsize{14} u','\fontsize{14} v','\fontsize{14} w')
title('\fontsize{20} Time course of u,v, and w in Fenton-Karma model')
ylim([0 1])
set(gca,'fontsize',20)

%% check with C code FK
close all;

load('C:\Users\zxio506\Desktop\Vm_single_cell_FK3current.mat')
Vm = squeeze(Vm(1,1,1,:));

figure;hold on;
plot(t(1,1:length(t)-1),Andy_Vm,'g','LineWidth',5)
plot(linspace(1,max(t),length(Vm))-1,Vm,'r','LineWidth',1)
title('Comparing MATLAB and C code: 4-Current FK')
legend('Andy MATLAB','Zhaohan C')

figure, plot(Andy_Vm(1:200:length(Andy_Vm))-Vm);
title('numerical error between the two models')