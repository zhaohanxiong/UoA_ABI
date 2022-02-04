function [sig] = ecgsynV2(sfecg,N,Anoise,hrmean,hrstd,lfhfratio,sfint,ti,ai,bi)

% input:
% 	sfecg 		= ECG sampling frequency [256 Hertz]
% 	N			= approximate number of heart beats [256]
% 	Anoise 		= Additive uniformly distributed measurement noise [0 mV]
% 	hrmean 		= Mean heart rate [60 beats per minute]
% 	hrstd 		= Standard deviation of heart rate [1 beat per minute]
% 	lfhfratio 	= LF/HF ratio [0.5]
% 	sfint 		= Internal sampling frequency [256 Hertz]
% 	ti 			= angles of extrema
% 	ai 			= z-position of extrema
% 	bi 			= Gaussian width of peaks
%
% output:
%	sig			= output ECG

% adjust extrema parameters for mean heart rate 
hrfact  = sqrt(hrmean/60);
hrfact2 = sqrt(hrfact);
bi      = hrfact*bi;
ti      = [hrfact2 hrfact2 hrfact 1 hrfact hrfact2 hrfact2].*ti;

% check that sfint is an integer multiple of sfecg 
q   = round(sfint/sfecg);
qd  = sfint/sfecg;

if q ~= qd
    error(['Internal sampling frequency (sfint) must be an integer ' ... 
    'multiple of the ECG sampling frequency (sfecg).']);
end

% define frequency parameters for rr process 
% flo and fhi correspond to the Mayer waves and respiratory rate respectively
flo     = 0.10;
flostd  = 0.01;

fhi     = 0.25;
fhistd  = 0.01;

% calculate time scales for rr and total output
sampfreqrr  = 1;
trr         = 1/sampfreqrr; 
rrmean      = (60/hrmean);	 
Nrr         = 2^(ceil(log2(N*rrmean/trr)));

% compute rr process
rr0 = rrprocess(flo,fhi,flostd,fhistd,lfhfratio,hrmean,hrstd,sampfreqrr,Nrr);

% upsample rr time series from 1 Hz to sfint Hz
rr = interp(rr0,sfint);
if sum(rr < 0) > 0
    fprintf('negative rr');
end
rr(rr < 0) = min(rr(rr>0)); % fk me in the ass

% make the rrn time series
dt      = 1/sfint;
rrn     = zeros(length(rr),1);
tecg    = 0;
i       = 1;

while i <= length(rr)
   tecg         = tecg+rr(i);
   ip           = round(tecg/dt);
   rrn(i:ip)    = rr(i);
   i            = ip+1;
end 

% integrate system using fourth order Runge-Kutta
%          x   y    P   C   T
x0      = [1.0 0.0 0 15 0];
Tspan   = [0:dt:(ip-1)*dt];

[~,X0] = ode45('derivsecgsynV2',Tspan,x0,[],rrn,sfint,ti,ai,bi,fhi);

% downsample to required sfecg
X 	= X0(1:q:end,:);
ECG = X(:,3) + X(:,4) + X(:,5); % plot(X(1:3000,3))

% Scale signal to lie between -0.4 and 1.2 mV
emin    = min(ECG);
emax    = max(ECG);
erange  = emax - emin;
ECG 	= (ECG - emin)*(1.6)/erange - 0.4;

% include additive uniformly distributed measurement noise 
eta = 2*rand(length(ECG),1) - 1;
sig = ECG + Anoise*eta;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rr = rrprocess(flo, fhi, flostd, fhistd, lfhfratio, hrmean, hrstd, sfrr, n)

w1 		= 2*pi*flo;
w2 		= 2*pi*fhi;
c1 		= 2*pi*flostd;
c2 		= 2*pi*fhistd;
sig2 	= 1;
sig1 	= lfhfratio;
rrmean 	= 60/hrmean;
rrstd 	= 60*hrstd/(hrmean*hrmean);

df 	= sfrr/n;
w 	= [0:n-1]'*2*pi*df;
dw1 = w-w1;
dw2 = w-w2;

Hw1 = sig1*exp(-0.5*(dw1/c1).^2)/sqrt(2*pi*c1^2);
Hw2 = sig2*exp(-0.5*(dw2/c2).^2)/sqrt(2*pi*c2^2);
Hw 	= Hw1 + Hw2;
Hw0 = [Hw(1:n/2); Hw(n/2:-1:1)];
Sw 	= (sfrr/2)*sqrt(Hw0);

ph0 = 2*pi*rand(n/2-1,1);
ph 	= [ 0; ph0; 0; -flipud(ph0) ]; 
SwC = Sw .* exp(j*ph);
x 	= (1/n)*real(ifft(SwC));

xstd 	= std(x);
ratio 	= rrstd/xstd;
rr 		= rrmean + x*ratio;