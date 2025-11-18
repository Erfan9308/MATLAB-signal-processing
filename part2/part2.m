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


%% Part 2.2 , loading songs and modulation

clc;
clear;
close all;

fs = 44100;  % Sampling frequency

[x1, ~] = audioread('John Lennon - Imagine.mp3');
[x2, ~] = audioread('ABBA - Mamma Mia.mp3');

% Use only left channel if stereo
if size(x1,2) > 1, x1 = x1(:,1); end
if size(x2,2) > 1, x2 = x2(:,1); end

% Load channel signal
load('channel.mat');  % Assumes variable is 'channel'
x_channel = channel;

% Match lengths
min_len = min([length(x1), length(x2), length(x_channel)]);
x1 = x1(1:min_len);
x2 = x2(1:min_len);
x_channel = x_channel(1:min_len);


% Carrier frequencies from OBW analysis
fc1 = 2500;     % Song 1 (Imagine)
fc2 = 12000;    % Song 2 (Mamma Mia)
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




%% part2.3 transmitting songs together , use the label to choose the song one or song two

clc;
clear;
close all;

label = 1; %% CHOOSE THE LABEL 1 FOR SONG ONE, AND LABEL TWO FOR SONG TWO

fs = 44100;


% Load reference song
if label == 1
    [x_ref, ~] = audioread('John Lennon - Imagine.mp3');
elseif label == 2
    [x_ref, ~] = audioread('ABBA - Mamma Mia.mp3');
else
    error('Invalid song label. Use 1 or 2.');
end

if size(x_ref,2) > 1
    x_ref = x_ref(:,1);
end

% Load channel and both songs
load('channel.mat');  % Contains 'channel'
[x1, ~] = audioread('John Lennon - Imagine.mp3');
[x2, ~] = audioread('ABBA - Mamma Mia.mp3');
if size(x1,2) > 1, x1 = x1(:,1); end
if size(x2,2) > 1, x2 = x2(:,1); end

min_len = min([length(x1), length(x2), length(channel), length(x_ref)]);
x1 = x1(1:min_len);
x2 = x2(1:min_len);
x_channel = channel(1:min_len);
x_ref = x_ref(1:min_len);






%%%%% case one, sone one carrier frequency 3000 and song two 13000 Hz

% Carrier frequencies
fc1 = 3000;
fc2 = 13000;

t = (0:min_len-1)' / fs;

% Modulation
x1_mod = x1 .* cos(2*pi*fc1*t);
x2_mod = x2 .* cos(2*pi*fc2*t);
x_tx = x1_mod + x2_mod + x_channel;

% Demodulate the selected song
if label == 1
    x_demod = x_tx .* (2 * cos(2*pi*fc1*t));
    song_name = 'Song One';
    fc_values = 500:100:4000;  % Hz
elseif label == 2
    x_demod = x_tx .* (2 * cos(2*pi*fc2*t));
    song_name = 'Song Two';
    fc_values = 2000:100:9000;  % Hz
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













%%%%% case two, sone one carrier frequency 5000 and song two 15000 Hz

% Carrier frequencies
fc1 = 5000;
fc2 = 15000;

t = (0:min_len-1)' / fs;

% Modulation
x1_mod = x1 .* cos(2*pi*fc1*t);
x2_mod = x2 .* cos(2*pi*fc2*t);
x_tx = x1_mod + x2_mod + x_channel;

% Demodulate the selected song
if label == 1
    x_demod = x_tx .* (2 * cos(2*pi*fc1*t));
    song_name = 'Song One';
    fc_values = 500:100:4000;  % Hz
elseif label == 2
    x_demod = x_tx .* (2 * cos(2*pi*fc2*t));
    song_name = 'Song Two';
    fc_values = 2000:100:9000;  % Hz
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

%% Part 2.4, both songs together, use label for song one & song two

clc;
clear;
close all;


label = 1; %% CHOOSE THE LABEL 1 FOR SONG ONE, AND LABEL TWO FOR SONG TWO


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

% Reference signal based on label
if label == 1
    x_ref = x1;
    fc = 5000;
    song_name = 'Song One';
elseif label == 2
    x_ref = x2;
    fc = 15000;
    song_name = 'Song Two';
else
    error('Invalid label');
end

% === Modulate and transmit ===
x_mod1 = x1 .* cos(2*pi*5000*t); 
x_mod2 = x2 .* cos(2*pi*15000*t);
x_tx = x_mod1 + x_mod2 + x_channel;

% === Demodulate selected song
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
surf(WZ * fs / (2*pi), RP, SIR_matrix); % convert omega_z to Hz
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
