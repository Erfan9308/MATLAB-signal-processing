%% Part 2.2 , loading songs and modulation

clc;
clear;
close all;

fs = 44100;  % Sampling frequency

[x1, ~] = audioread('John Lennon - Imagine.mp3');
[x2, ~] = audioread('ABBA - Mamma Mia.mp3');


if size(x1,2) > 1, x1 = x1(:,1); end
if size(x2,2) > 1, x2 = x2(:,1); end


load('channel.mat');  
x_channel = channel;

min_len = min([length(x1), length(x2), length(x_channel)]);
x1 = x1(1:min_len);
x2 = x2(1:min_len);
x_channel = x_channel(1:min_len);



fc1 = 3000;     % Song 1 (Imagine)
fc2 = 13000;    % Song 2 (Mamma Mia)
t = (0:min_len-1)' / fs;

% AM Modulation (Double Sideband)
x1_mod = x1 .* cos(2*pi*fc1*t);
x2_mod = x2 .* cos(2*pi*fc2*t);

% Multiplexed signal with channel interference
x_tx = x1_mod + x2_mod + x_channel;

% Plot frequency spectrum of transmitted signal
nfft = 2^nextpow2(length(x_tx));
f = fs * (-nfft/2:(nfft/2)-1) / nfft;
Xf = fftshift(fft(x_tx, nfft));

figure;
plot(f, abs(Xf));
xlabel('Frequency (Hz)');
ylabel('Magnitude')
title('Spectrum of Multiplexed Signal');
grid on;
