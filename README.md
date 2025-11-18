# MATLAB-signal-processing
Simulation of a communication system in MATLAB: noise removal with Bessel &amp; notch filters, AM modulation, SIR optimization, and signal recovery analysis.


Communication System Simulation (MATLAB)

This repository contains the full implementation of a digital communication system simulation developed as the final project for the Signals and Systems course at Politecnico di Torino.

The project includes:

Noise removal from corrupted audio signals

Bessel and notch filter design

AM modulation and spectral multiplexing

Coherent demodulation and reconstruction

SIR-based optimization of filter and carrier choices

MATLAB scripts, datasets, plots, and the full report

Two audio signals are processed throughout the project:
Song 1: Imagine – John Lennon
Song 2: Mamma Mia – ABBA

📘 Project Overview
Part 1 – Signal Recovery

The goal is to remove a strong interference tone at 5567.5 Hz.

Techniques implemented:

1. Fourth-order Bessel Low-Pass Filter

Implemented via bilinear transform

Sweep of cutoff frequencies

Delay compensation using finddelay + circshift

SIR evaluation for both songs

2. Custom Digital Notch Filter

Pole–zero placement on the unit circle

Sweeping of pole radius

Achieved up to 37.6 dB SIR

Nearly perfect reconstruction of both audio signals

Part 2 – Transmission Over a Shared Channel

A simulated channel has an already-occupied baseband.
Both songs must be transmitted without overlapping the channel disturbance.

Key steps:

AM modulation using different carrier frequencies

Multiplexing the two songs in distinct spectral regions

Coherent demodulation

Recovery via:

Optimized Bessel LPF

Custom IIR LPF (poles + zeros)

Main findings:

Higher carrier frequencies → cleaner demodulation

Bessel filters outperform custom LPF in most cases

Single-song transmission yields the highest SIR values
(up to 24.61 dB with Bessel LPF)
