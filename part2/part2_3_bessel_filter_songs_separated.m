%% part2.3 each song separately , use the label to choose the song one or song two

clc;
clear;
close all;

label = 1; %% CHOOSE THE LABEL 1 FOR SONG ONE, AND LABEL TWO FOR SONG TWO

fs = 44100;


% Load reference song
if label == 1
    [x_ref, ~] = audioread('John Lennon - Imagine.mp3');
    [x1, ~] = audioread('John Lennon - Imagine.mp3');
    fc = 18000;
elseif label == 2
    [x_ref, ~] = audioread('ABBA - Mamma Mia.mp3');
    [x1, ~] = audioread('ABBA - Mamma Mia.mp3');
    fc = 16000;
else
    error('Invalid song label. Use 1 or 2.');
end

if size(x_ref,2) > 1
    x_ref = x_ref(:,1);
end

if size(x1,2) > 1, x1 = x1(:,1); end
load('channel.mat');  

min_len = min([length(x1), length(channel), length(x_ref)]);
x1 = x1(1:min_len);
x_channel = channel(1:min_len);
x_ref = x_ref(1:min_len);



t = (0:min_len-1)' / fs;

% Modulation
x1_mod = x1 .* cos(2*pi*fc*t);
x_tx = x1_mod + x_channel;

% Demodulate the selected song
if label == 1
    x_demod = x_tx .* (2 * cos(2*pi*fc*t));
    song_name = 'Song One';
    fc_values = 500:100:6000;
elseif label == 2
    x_demod = x_tx .* (2 * cos(2*pi*fc*t));
    song_name = 'Song Two';
    fc_values = 5000:100:9000;
end

% Optimization loop over cutoff frequencies
SIR_values = zeros(size(fc_values));
best_SIR = -Inf;

for i = 1:length(fc_values)
    fc_lp = fc_values(i);

    % Design Bessel filter
    [b_a, a_a] = besself(4, 1);
    [b_a, a_a] = lp2lp(b_a, a_a, 2*pi*fc_lp);
    [bz, az] = bilinear(b_a, a_a, fs);

    % Filter signal
    x_rec = filter(bz, az, x_demod);

    % Align with reference
    delay = finddelay(x_ref, x_rec);
    x_rec_aligned = circshift(x_rec, -delay);

    % Compute SIR
    e = x_rec_aligned - x_ref;
    SIR = mean(x_rec_aligned.^2) / mean(e.^2);
    SIR_dB = 10*log10(SIR);
    SIR_values(i) = SIR_dB;

    if SIR_dB > best_SIR
        best_SIR = SIR_dB;
        best_fc = fc_lp;
        best_rec = x_rec_aligned;
    end
end

fprintf('Best cutoff frequency: %.0f Hz\n', best_fc);
fprintf('Max SIR: %.2f dB\n', best_SIR);

% Plot SIR vs. cutoff frequency
figure;
plot(fc_values, SIR_values, 'o-', 'LineWidth', 1.5);
xlabel('Cutoff Frequency (Hz)');
ylabel('SIR (dB)');
title(['SIR vs Cutoff Frequency for ', song_name]);
grid on;

% Plot spectrum of filtered vs transmitted signal
nfft = 2^nextpow2(length(x_demod));
f = fs * (-nfft/2:(nfft/2)-1) / nfft;
X_demod = fftshift(fft(x_demod, nfft));
X_rec = fftshift(fft(best_rec, nfft));

figure;
plot(f, abs(X_demod), 'b', 'DisplayName', 'Demodulated Signal');
hold on;
plot(f, abs(X_rec), 'r', 'DisplayName', 'Recovered Signal (Filtered)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title(['Spectrum of Demodulated vs Recovered Signal - ', song_name]);
legend;
grid on;