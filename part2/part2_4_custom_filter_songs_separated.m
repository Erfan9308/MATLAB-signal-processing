%% Part 2.4, for each song separately, use the label for song one or song two 

clc;
clear;
close all;
label = 2;  %% CHOOSE THE LABEL 1 FOR SONG ONE, AND LABEL TWO FOR SONG TWO


fs = 44100;


% === Load audio ===
[x1, ~] = audioread('John Lennon - Imagine.mp3');
[x2, ~] = audioread('ABBA - Mamma Mia.mp3');

if size(x1,2) > 1, x1 = x1(:,1); end
if size(x2,2) > 1, x2 = x2(:,1); end
load('channel.mat');

% Match length
min_len = min([length(x1), length(x2), length(channel)]);
x1 = x1(1:min_len);
x2 = x2(1:min_len);
x_channel = channel(1:min_len);
t = (0:min_len-1)' / fs;

% === Select signal and carrier
if label == 1
    x_tx_base = x1;
    x_ref = x1;
    fc = 18000;
    song_name = 'Song One';
elseif label == 2
    x_tx_base = x2;
    x_ref = x2;
    fc = 16000;
    song_name = 'Song Two';
else
    error('Invalid label');
end

% === Modulate and transmit only selected song
x_mod = x_tx_base .* cos(2*pi*fc*t);
x_tx = x_mod + x_channel;

% === Demodulate
x_demod = x_tx .* (2 * cos(2*pi*fc*t));

% === Sweep over pole radius and zero angle
rp_values = 0.50:0.1:0.99;
omega_z_values = pi/3:0.1:pi;
rz = 1; % fixed zero radius

SIR_matrix = zeros(length(rp_values), length(omega_z_values));
best_SIR = -Inf;
omega_p = 0;

for i = 1:length(rp_values)
    rp = rp_values(i);
    for j = 1:length(omega_z_values)
        omega_z = omega_z_values(j);

        % Filter coefficients
        b = [1, -2*rz*cos(omega_z), rz^2];
        a = [1, -2*rp*cos(omega_p), rp^2];

        % Normalize gain at DC
        gain_at_dc = sum(b) / sum(a);
        b = b / gain_at_dc;

        % Filter and align
        x_rec = filter(b, a, x_demod);
        delay = finddelay(x_ref, x_rec);
        x_rec_aligned = circshift(x_rec, -delay);

        % Compute SIR
        e = x_rec_aligned - x_ref;
        SIR = mean(x_rec_aligned.^2) / mean(e.^2);
        SIR_dB = 10*log10(SIR);
        SIR_matrix(i,j) = SIR_dB;

        if SIR_dB > best_SIR
            best_SIR = SIR_dB;
            best_rp = rp;
            best_omega_z = omega_z;
            best_rec = x_rec_aligned;
        end
    end
end

fprintf('Best zero angle: %.3f rad\n', best_omega_z);
fprintf('Best pole radius: %.3f\n', best_rp);
fprintf('Max SIR: %.2f dB\n', best_SIR);

% Recalculate best filter
b = [1, -2*rz*cos(best_omega_z), rz^2];
a = [1, -2*best_rp*cos(omega_p), best_rp^2];
b = b / (sum(b) / sum(a));

% === Z-plane plot
figure;
zplane(b, a);
title(['Z-Plane of Optimized Filter - ', song_name]);
hold on;
plot(nan, nan, 'ko', 'DisplayName', 'Zero (o)');
plot(nan, nan, 'kx', 'DisplayName', 'Pole (x)');
legend;

% === Frequency response
N = 2048;
[H, w] = freqz(b, a, N, 'whole');
H_shifted = fftshift(H);
f = linspace(-fs/2, fs/2, N);
figure;
plot(f, 20*log10(abs(H_shifted)));
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title(['Frequency Response of custom filter -', song_name]);
grid on;
xlim([-fs/2, fs/2]);

% === 3D SIR surface plot
[WZ, RP] = meshgrid(omega_z_values, rp_values);
figure;
surf(WZ * fs / (2*pi), RP, SIR_matrix);
xlabel('Zero Frequency (Hz)');
ylabel('Pole Radius (r_p)');
zlabel('SIR (dB)');
title(['SIR vs. Pole Radius and Zero Frequency - ', song_name]);
shading interp;
grid on;

% === Spectrum plot
nfft = 2^nextpow2(length(x_demod));
f = fs * (-nfft/2:(nfft/2)-1) / nfft;
X_demod = fftshift(fft(x_demod, nfft));
X_rec = fftshift(fft(best_rec, nfft));

figure;
plot(f, abs(X_demod), 'b', 'DisplayName', 'Demodulated');
hold on;
plot(f, abs(X_rec), 'r', 'DisplayName', 'Recovered');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title(['Demodulated vs. Filtered Spectrum - ', song_name]);
legend;
grid on;
