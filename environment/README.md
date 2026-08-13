### NVIDIA CUDA C++ Development Environment Setup ###
Install Ubuntu 24.04 on windows 11, dual boot. Total 1.7TB free space on windows 11.   

This directory contains comprehensive documentation for provisioning a bare-metal, dual-boot Linux development environment tailored specifically for High-Performance Computing (HPC), low-level GPU programming, and AI/ML workloads. 

The guide details the precise steps required to safely partition a Windows 11 host, install Ubuntu 24.04 LTS, alongside Windows 11, targeting NVIDIA RTX 5080 (Blackwell SM120) with CUDA 12.9 and configure a complete NVIDIA software stack for native CUDA C++ kernel development.  

## Hardware Architecture

This environment was architected and profiled on the following dedicated local workstation:

| Component | Specification |
| :--- | :--- |
| **CPU** | AMD Ryzen 7 9800X3D |
| **GPU** | NVIDIA GeForce RTX 5080 (Blackwell Architecture) |
| **VRAM** | 16GB |
| **Storage Strategy** | ~900GB dedicated to Ubuntu root / 32GB Swapfile for ML/AI workloads |

## Software & Toolchain Stack

The environment relies on specific package versions to ensure compatibility with modern NVIDIA hardware and advanced C++ libraries (such as CuTe and CUTLASS). 

* **Operating System:** Ubuntu 24.04 LTS
* **NVIDIA Driver:** 595.71.05
* **CUDA Driver Capability:** 13.2 (via `nvidia-smi`)
* **CUDA Compiler Toolkit:** 12.9 (via `nvcc`)
* **C++ Build System:** `gcc`, `make`, `cmake`
* **Profiling Tools:** NVIDIA Nsight Systems (`nsys`) & Nsight Compute (`ncu`)
* **Containerization:** Docker Engine with NVIDIA Container Toolkit

## Guide Contents

The master setup guide ([`setup.md`](setup.md)) provides reproducible, step-by-step terminal commands for the entire configuration lifecycle:

1. **Windows Partitioning & Pre-requisites:** Safe disk shrinking using `diskpart`, managing unmovable files, and disabling Fast/Secure Boot in MSI BIOS.
2. **Ubuntu 24.04 Installation:** Optimal partition sizing for dual-boot and GRUB bootloader configuration.
3. **Swapfile Configuration:** Expanding virtual memory to 32GB to prevent Out-Of-Memory (OOM) errors during heavy dataset loading or large-model training.
4. **CUDA Toolkit & Driver Configuration:** Utilizing the official NVIDIA APT network repository to manage proprietary drivers and the `12.9` toolkit.
5. **C++ Toolchain & IDE Setup:** Configuring VS Code extensions, native compilers, and NVIDIA Nsight profilers.
6. **Containerization & Python:** Configuring the NVIDIA Container Toolkit for GPU passthrough in Docker, and setting up isolated Python virtual environments for nightly PyTorch builds.

## How to Use This Documentation

This documentation serves as a reference for replicating this exact build. If you are cloning this repository to compile the native CUDA kernels, please ensure your local `nvcc` compiler matches or exceeds version `12.9`, as older versions may struggle with the template metaprogramming required by modern CuTe layouts. 

For full installation commands and troubleshooting steps, read the [Detailed Setup Guide](setup.md).

Please see the following upgrade output:  
<pre>
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.05              Driver Version: 595.71.05      CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5080        Off |   00000000:01:00.0  On |                  N/A |
|  0%   40C    P8             19W /  360W |     497MiB /  16303MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3083      G   /usr/lib/xorg/Xorg                      246MiB |
|    0   N/A  N/A            3311      G   /usr/bin/gnome-shell                     54MiB |
|    0   N/A  N/A            4378      G   ...rack-uuid=3190708988185955192         92MiB |
|    0   N/A  N/A            8517      G   ...rack-uuid=3190708988185955192         55MiB |
+-----------------------------------------------------------------------------------------+
</pre>
