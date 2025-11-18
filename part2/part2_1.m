%% Part 2.1 , analyzing the channel , and trying to transmit songs without modulation
% use label to choose songs for trying to transmit and get the SIR
clc;
clear;
close all;

label = 1; %% CHOOSE THE LABEL 1 FOR SONG ONE, AND LABEL TWO FOR SONG TWO


fs = 44100;
load('channel.mat'); 




if label == 1
    load('313497_SONG1.mat');  
    [x_song, fs_song] = audioread('John Lennon - Imagine.mp3');

elseif label == 2
    load('313497_SONG2.mat');
    [x_song, fs_song] = audioread('ABBA - Mamma Mia.mp3');
    
else
    error('Invalid song label. Use "song1" or "song2".');
end

% if stereo
if size(x_song,2) > 1
    x_song = x_song(:,1);
end




x_channel = channel; 


% Match lengths
min_len = min(length(x_song), length(x_channel));
x_song = x_song(1:min_len);
x_channel = x_channel(1:min_len);



x_tx = x_song + x_channel;  % Simulated transmission

delay = finddelay(x_song, x_tx);
x_tx_aligned = circshift(x_tx, -delay);

e = x_tx_aligned - x_song;
SIR = mean(x_tx_aligned.^2) / mean(e.^2);
SIR_dB = 10*log10(SIR);

fprintf('Baseband Transmission SIR = %.2f dB\n', SIR_dB);






% Frequency-domain analysis (two-sided)
nfft = 2^nextpow2(length(x_channel));
Xf = fftshift(fft(x_channel, nfft));  % Shift zero freq to center
Xf_power = abs(Xf).^2;                % Full power spectrum (magnitude squared)
f = fs * (-nfft/2 : nfft/2 - 1) / nfft;  % Frequency axis with negative freqs


% Normalize the power
total_power = sum(Xf_power);
cumulative_power = cumsum(Xf_power);

% Find OBW bounds
lower_idx = find(cumulative_power >= 0.005 * total_power, 1, 'first');
upper_idx = find(cumulative_power >= 0.995 * total_power, 1, 'first');

% Compute OBW
OBW = f(upper_idx) - f(lower_idx);
fprintf('Occupied Bandwidth (99%%): %.2f Hz\n', OBW);


% figure;
% plot(f, 10*log10(Xf_power)); % Power in dB
% hold on;
% xline(f(upper_idx), 'r--', 'Upper Bound');
% xlabel('Frequency (Hz)');
% ylabel('Power (dB)');
% title('Power Spectrum with 99% Occupied Bandwidth (Two-Sided)');
% legend('Power Spectrum', 'Upper Bound');


figure;
plot(f, abs(Xf));
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Two-Sided Spectrum of the Channel Signal');
grid on;
xlim([-fs/2 fs/2]);
