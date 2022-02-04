%% 2003 paper McSharry et al.
sfecg       = 300;              % sampling freqency
N           = 25;               % number of heart beats
Anoise      = 0;                % uniformly distributed noise in mV
hrmean      = 60;               % mean heart rate bpm
hrstd       = 20;                % standard deviation of heart rate per min
lfhfratio   = 0.5;              % LF/HF
sfint       = 300;              % internal sampling frequency

%       P  Q  R  S  T morphology
ti = [-70 -15 0 15 100]*pi/180; % angles of PQRST
ai = [1.2 -5 30 -7.5 0.75];     % heights of PQRST
bi = [0.25 0.1 0.1 0.1 0.4];    % widths of PQRST

[ecg,~] = ecgsyn(sfecg,N,Anoise,hrmean,hrstd,lfhfratio,sfint,ti,ai,bi);

%% 2010 paper Sayadi et al.
sfecg  = 300;	% sampling freqency
N      = 25;	% number of heart beats
Anoise = 0;		% uniformly distributed noise in mV

% Choose arrhythmia: sinus rhythm (N), sinus bradycardia (SB), sinus tachycardia (ST), 
% ventricular flutter (VF), atrial fibrillation (AF), ventricular tachycardia (VT)
type = 'N';

if strcmp(type,'N')
    
	hrmean      = 60;		% mean heart rate bpm
	hrstd       = 0;		% standard deviation of heart rate per min
	lfhfratio   = 0.5;		% LF/HF
	%          P-   P+  Q  R   S   T-   T+   morphology
    ai 		= [3.0 3.0 -5.0 30 -2.0 2.5 2.5];       % heights
    bi 		= [0.25 0.25 0.1 0.1 0.1 0.5 0.5];		% widths
	thetai 	= [-7/15 -7/15 1/12 0 1/12 5/9 5/9]*pi;	% angles
    
elseif strcmp(type,'SB')
    
	hrmean      = 40;		% mean heart rate bpm
	hrstd       = 2;		% standard deviation of heart rate per min
	lfhfratio   = 0.4;		% LF/HF
	%         P-   P+   Q   R   S   T-   T+   morphology
	ai 		= [0.7 0.8 -1 20 -9.5 0.27 0.15];		% heights
	bi 		= [0.2 0.1 0.1 0.1 0.1 0.4 0.55];		% widths
	thetai 	= [-3/8 -1/3 -1/13 0 1/15 2/5 4/7]*pi;	% angles
    
elseif strcmp(type,'ST')
    
	hrmean      = 140;		% mean heart rate bpm
	hrstd       = 10;		% standard deviation of heart rate per min
	lfhfratio   = 0.6;		% LF/HF
	%         P-   P+   Q   R   S   T-   T+   morphology
	ai 		= [0.7 0.8 -7 20 -9.5 0.27 0.15];		% heights
	bi 		= [0.2 0.1 0.1 0.1 0.1 0.4 0.55];		% widths
	thetai 	= [-3/7 -3/8 -1/13 0 1/17 1/2 4/7]*pi;	% angles
    
elseif strcmp(type,'VF')
    
	hrmean      = 300;		% mean heart rate bpm
	hrstd       = 20;		% standard deviation of heart rate per min
	lfhfratio   = 0.5;		% LF/HF
	%         P-   P+   Q   R  S   T-  T+   morphology
	ai 		= [0.0 0.0 0.0 20 -20 0.0 0.0];				% heights
	bi 		= [0.1 0.1 0.1 0.6 0.6 0.1 0.1];			% widths
	thetai 	= [-1/9 -2/3 -1/12 -1/2 -1/2 3/8 5/8]*pi;	% angles
    
elseif strcmp(type,'AF')
    
	hrmean      = 60;		% mean heart rate bpm
    hrstd       = 5;		% standard deviation of heart rate per min
	lfhfratio   = 0.5;		% LF/HF
	%         P-   P+   Q   R   S   T-   T+   morphology
	ai 		= [0.7 0.9 0.6 18 -0.1 0.62 0.55];		% heights
    bi 		= [0.12 0.13 0.12 0.1 0.05 0.15 0.17];	% widths
	thetai 	= [-5/7 -1/2 -1/4 0 1/30 1/4 7/11]*pi;	% angles
    
elseif strcmp(type,'VT')
    
	hrmean      = 100;		% mean heart rate bpm
	hrstd       = 5;		% standard deviation of heart rate per min
	lfhfratio   = 0.5;		% LF/HF
	%         P-   P+   Q   R   S   T-   T+   morphology
	ai 		= [1.0 1.0 -12 1.0 3.0 5.0 3.0];			% heights
	bi 		= [0.2 0.1 0.2 0.3 0.4 0.5 0.45];			% widths
	thetai 	= [-10/13 -2/3 -1/3 0 2/11 1/2 20/23]*pi;	% angles
    
else
	error('Incorrect pathology entered')
end

sfint = sfecg*1; % internal sampling frequency

[ecg] = ecgsynV2(sfecg,N,Anoise,hrmean,hrstd,lfhfratio,sfint,thetai,ai,bi);

plot(ecg);

% write to output
val=ecg(1:2501);
val=val-min(val);
val=val/max(val)*255;
%save('C:\Users\zxio506\Desktop\A_fake60bpm.mat','val');