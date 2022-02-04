%%Time discretization
ht = 0.005;            % Delta_t
t  = 0:ht:1000;        % Time vector

%% Parameters
V_0 = -83.91; % Adjusted resting potential, V_0 = -85;

tau_v1_minus = 16.3;
tau_v2_minus = 1150;
tau_v_plus = 1.703;
tau_w1_minus = 79.963;
tau_w2_minus = 28.136;
tau_w_plus = 213.55;
tau_fi = 0.084;
tau_o1 = 250.03;
tau_o2 = 16.632;
tau_so1 = 73.675;
tau_so2 = 6.554;
tau_s1 = 9.876;
tau_s2 = 4.203;
tau_si = 10.699;
tau_w_inf = 0.223;

theta_v = 0.3;
theta_w = 0.1817;
theta_v_minus = 0.1007;
theta_o = 0.0155;

u_w_minus = 0.01;
u_o = 0;
u_u = 1.0089;
u_so = 0.5921;
u_s = 0.8157;

k_w_minus = 60.219;
k_so = 2.975;
k_s = 2.227;

w_inf_star = 0.902;
V_mu = 85.7;

%%============ Initialize variables
% Gating variables
u = zeros(length(t),1);
v = zeros(length(t),1);
w = zeros(length(t),1);
s = zeros(length(t),1);

% Gating steady state
v_inf = zeros(length(t)-1,1);
w_inf = zeros(length(t)-1,1);

% Relaxation times
tau_v_minus = zeros(length(t)-1,1);
tau_w_minus = zeros(length(t)-1,1);
tau_so = zeros(length(t)-1,1);
tau_s = zeros(length(t)-1,1);
tau_o = zeros(length(t)-1,1);

% Currents
J_fi = zeros(length(t)-1,1);
J_so = zeros(length(t)-1,1);
J_si = zeros(length(t)-1,1);

% Membrane potential
Vm = zeros(length(t)-1,1);

%%Stimulus
J_stim = zeros(length(t)-1,1);

% J_stim(20001:20401) = -0.2; % Stimulus set-up used in paper
J_stim(10000:10400) = -0.2; % For report

%% Initial conditions
u(1,1) = 1.18*10^-8;
v(1,1) = 1;
w(1,1) = 1;
%***
s(1,1) = 0.02587;

%% Computer Simulation
for ite = 1:length(t)-1 

	% Relaxation times
	tau_v_minus(ite,1) = (tau_v1_minus)*(1-heaviside(u(ite,1)-theta_v_minus)) + (tau_v2_minus)*(heaviside(u(ite,1)-theta_v_minus));
	tau_w_minus(ite,1) = (tau_w1_minus)+0.5*(tau_w2_minus-tau_w1_minus)*(1+tanh(k_w_minus*(u(ite,1)-u_w_minus)));
	tau_so(ite,1) = tau_so1 +0.5*(tau_so2-tau_so1)*(1+tanh(k_so*(u(ite,1)-u_so)));
	tau_s(ite,1) = tau_s1*(1-heaviside(u(ite,1)-theta_w)) + tau_s2*(heaviside(u(ite,1)-theta_w));
	tau_o(ite,1) = tau_o1*(1-heaviside(u(ite,1)-theta_o)) + tau_o2*(heaviside(u(ite,1)-theta_o));

	% J's
	J_fi(ite,1) = ((-v(ite,1)*(u(ite,1)-theta_v)*(u_u-u(ite,1)))/(tau_fi))*(heaviside(u(ite,1)-theta_v));
	J_so(ite,1) = ((u(ite,1)-u_o)/(tau_o(ite,1)))*(1-heaviside(u(ite,1)-theta_w)) + ((heaviside(u(ite,1)-theta_w))/(tau_so(ite,1)));
	J_si(ite,1) = (-s(ite,1)*w(ite,1)*(heaviside(u(ite,1)-theta_w)))/(tau_si);

	% Gating steady state
	v_inf(ite,1) = heaviside(theta_v_minus - u(ite,1));
	w_inf(ite,1) = (1-((u(ite,1))/(tau_w_inf)))*(1-heaviside(u(ite,1)-theta_o)) + w_inf_star*(heaviside(u(ite,1)-theta_o));

	% Numerical integration using forward euler
	u(ite+1,1) = u(ite,1) + ht*(-(J_fi(ite,1)+J_so(ite,1)+J_si(ite,1)+J_stim(ite,1)));
	v(ite+1,1) = v(ite,1) + ht*(((v_inf(ite,1)-v(ite,1))/(tau_v_minus(ite,1)))*(1-heaviside(u(ite,1)-theta_v)) - ((v(ite,1)*heaviside(u(ite,1)-theta_v))/(tau_v_plus)));
	w(ite+1,1) = w(ite,1) + ht*(((w_inf(ite,1)-w(ite,1))/(tau_w_minus(ite,1)))*(1-heaviside(u(ite,1)-theta_w)) - ((w(ite,1)*heaviside(u(ite,1)-theta_w))/(tau_w_plus)));
	s(ite+1,1) = s(ite,1) + ht*((((1+tanh(k_s*(u(ite,1)-u_s)))/(2))-s(ite,1))/(tau_s(ite,1)));

	% Membrane potential
	Vm(ite,1) = V_mu*u(ite,1) + V_0;

end

Andy_Vm = Vm;

%% Plotting
figure(3)
plot(t(1,1:length(t)-1),Andy_Vm,'b','LineWidth',5)
ylabel('\fontsize{20} V(mV)')
xlabel('\fontsize{20} Time(ms)')
title('\fontsize{20} Time course of AP in Four-Current Fenton-Karma model (control)')
set(gca,'fontsize',20)

figure(4)
plot(t,u,'g','LineWidth',5)
hold on
plot(t,v,'y','LineWidth',5)
plot(t,w,'r','LineWidth',5)
plot(t,s,'k','LineWidth',5)
ylabel('\fontsize{20} Open probability')
xlabel('\fontsize{20} Time(ms)')
legend('\fontsize{14} u','\fontsize{14} v','\fontsize{14} w','\fontsize{14} s')
title('\fontsize{20} Time course of u,v, and w in Four-Current Fenton-Karma model (control)')
% ylim([0 1])
set(gca,'fontsize',20)

%% check with C code FK
close all;

load('Vm_single_cell_FK4current.mat')
Vm = squeeze(Vm(1,1,1,:));

figure;hold on;
plot(t(1,1:length(t)-1),Andy_Vm,'g','LineWidth',5)
plot(1:length(Vm),Vm,'r','LineWidth',1)
title('Comparing MATLAB and C code: 4-Current FK')
legend('Andy MATLAB','Zhaohan C')

figure, plot(Andy_Vm(1:200:length(Andy_Vm))-Vm);
title('numerical error between the two models')