# Communication System Simulation (MATLAB)

This repository contains the full implementation of a digital communication system simulation developed as the final project for the *Signals and Systems* course at Politecnico di Torino.

The project includes:
- Noise removal from corrupted audio signals  
- Bessel and notch filter design  
- AM modulation and spectral multiplexing  
- Coherent demodulation and reconstruction  
- SIR-based optimization of filter and carrier choices  
- MATLAB scripts, datasets, plots, and the full academic report  

Two audio signals are processed throughout the project:  
**Song 1:** Imagine – John Lennon  
**Song 2:** Mamma Mia – ABBA  

---

## 📘 Project Overview

### **Part 1 – Signal Recovery**
Removal of a strong narrowband interference at **5567.5 Hz**.

#### **1. Fourth-Order Bessel Low-Pass Filter**
- Designed using bilinear transform  
- Sweep of cutoff frequencies  
- Delay compensation (`finddelay` + `circshift`)  
- SIR evaluation for both signals  

#### **2. Custom Digital Notch Filter**
- Pole–zero placement on the unit circle  
- Sweep of pole radius  
- Achieved up to **37.6 dB SIR**  
- Near-perfect reconstruction of both songs  

---

### **Part 2 – Shared Channel Transmission**

Simulation of transmitting both songs over a channel whose baseband is already occupied.

#### **Techniques Used**
- Amplitude modulation (AM)  
- Spectral separation of the two songs  
- Coherent demodulation  
- Recovery using:  
  - Optimized Bessel low-pass filter  
  - Custom IIR low-pass filter  

#### **Key Findings**
- Higher carrier frequencies → cleaner demodulation  
- Bessel LPF generally outperforms custom IIR LPF  
- Single-song transmission yields best SIR values  
  (up to **24.61 dB** with Bessel LPF)  

---

## 📁 Repository Structure

