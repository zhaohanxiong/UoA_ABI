function [Recomposed_Phase] = ExtraCellularUnipolarPotential2PhaseMap(Phi,signal_freq)

% input is the unipolar extra cellular potential, input has to be 1000 Hz !!
% output is the phase map of the recomposed electrogram

%% Nested Function to calculate the phase for a single 1D electrogram
    function [Electrogram_Phase, Phase, Recomposed_Signal] = Unipolar2Phase_Electrogram(Electrogram, Period)
        % MATLAB function calculating phase of unipolar electroram
        %   Parameters:
        %       Electrogram - original electrogram, should be one-dimensional array
        %                     of doubles
        %       Period - base cycle length of the activity (can be obtained as dominant frequency of 
        %               electrogram or just put manually after electrogram inspection;
        %               units: same as sampling of Electrogram);
        %                
        % STEP ONE: SINUSOIDAL RECOMPOSITION
        Recomposed_Signal = zeros(numel(Electrogram),1);
        Period = round(Period); 

        % create sinusoid (to speed up calculations)
        Sinusoid_Wavelet = zeros(Period+1,1);
        for t=1:Period
            Sinusoid_Wavelet(t) = sin( 2*pi*(t-Period/2)/Period);
        end

        % calculation of the recomposed signal
        for t=2:numel(Recomposed_Signal)-1
            diff = Electrogram(t+1) - Electrogram(t-1);
            if diff < 0 
                for tt=-floor(Period/2):floor(Period/2)
                    if t+tt>0 && t+tt<numel(Electrogram)
                       Recomposed_Signal(t+tt)= Recomposed_Signal(t+tt) + diff*Sinusoid_Wavelet(floor(tt+Period/2+1)); 
                    end
                end
            end
        end

        % STEP TWO: COMPUTATION OF THE PHASE
        h = hilbert(Recomposed_Signal);
        Electrogram_Phase=atan2(real(h),-imag(h));
        Phase = h; 
    end

%% Calculate the phase for each individual electrogram, to get the whole electrogram
% calculate frequency
%signal_freq = 1000/3;

% compute period
freqgrid = zeros(size(Phi,1),size(Phi,2));
for i = 1:size(freqgrid,1)
    for j = 1:size(freqgrid,2)
        if sum(1-isnan(Phi(i,j,:))) > 0
            Virt = squeeze(Phi(i,j,:));
            f = signal_freq+1;                               % sampling frequency
            f_cutoff = 12;                          % cutoff frequency
            fnorm1 = f_cutoff/(f/2);                % normalized cut off freq
            f_cutoff = 4;                           % cutoff frequency
            fnorm2 = f_cutoff/(f/2);
            [b1,a1] = butter(2,[fnorm2, fnorm1]);   % Low pass Butterworth filter of order 10
            ry = filtfilt(b1,a1,(abs(Virt)));       % filtering
            [amp, freq] = findFFT(ry);
            index0 = find(amp==max(amp));
            freqgrid(i,j) = 1000*freq(index0(1));
        end
    end
end

Recomposed_Phase = nan(size(Phi));        % phase of recomposed electrogram
Phase_Map = zeros(size(Phi));               % phase map
Recomposed_Electrogram = zeros(size(Phi));  % recomposed electrogram

for i = 1:size(Phi,1)
    for j = 1:size(Phi,2)
        if sum(1-isnan(Phi(i,j,:))) > 0
            period = signal_freq/freqgrid(i,j);
            [Recomposed_Phase(i,j,:),Phase_Map(i,j,:),Recomposed_Electrogram(i,j,:)] = Unipolar2Phase_Electrogram(Phi(i,j,:),period);
        end
    end
end

% figure, imagesc(squeeze(Phase_Map(:,:,end/2)));colormap('Jet'); % phase map
% figure, plot(real(squeeze(Phase_Map(7,7,:))),-imag(squeeze(Phase_Map(7,7,:))));
% figure, plot(1:size(Phi,3), squeeze(Recomposed_Phase(7,7,:)),'--b',1:size(Phi,3), squeeze(Recomposed_Electrogram(7,7,:))./100,'-r');

end