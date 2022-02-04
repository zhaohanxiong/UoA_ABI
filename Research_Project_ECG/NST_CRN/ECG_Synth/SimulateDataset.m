%% Default Variables
sfecg       = 300;	% sampling freqency
N           = 25;	% number of heart beats
Anoise      = 0;	% uniformly distributed noise in mV
lfhfratio   = 0.5;	% LF/HF
hrmean      = 60;	% mean heart rate bpm
hrstd       = 0;	% standard deviation of heart rate per min
sfint       = 300;  % internal sampling frequency

%          P-    P+    Q    R   S    T-  T+         morphology
ai 		= [3.0   3.0   -5.0 30  -2.0 2.5 2.5];      % heights
bi 		= [0.25  0.25  0.1  0.1 0.1  0.5 0.5];		% widths
thetai 	= [-7/15 -7/15 1/12 0   1/12 5/9 5/9]*pi;	% angles

%% Changing Variables
dat = zeros(21,2500);
ref = zeros(21,2);
i = 0;

for hrmean = 50:5:150
    for repeat = 1:1
        %for hrstd = 5:5:20
            % counter
            i = i + 1;
            fprintf('%i: hrmean = %i hrstd = %i\n',i,hrmean,hrstd);

            % simulate data
            [ecg] = ecgsynV2(sfecg,N,Anoise,hrmean,hrstd,lfhfratio,sfint,thetai,ai,bi);

            % store
            dat(i,:) = ecg(2001:4500);
            ref(i,:) = [hrmean, hrstd];
        %end
    end
end

fprintf('\nTotal of %i ECGs simulated\n\n',i);

save('C:\Users\zxio506\Desktop\ECGtemplateN.mat','dat');
save('C:\Users\zxio506\Desktop\ParameterRefN.mat','ref');