
### Before start - record current state
bash\
Record everything about your current installation\
`nvcc --version`\
`nvidia-smi`\
`dpkg -l | grep cuda > ~/cuda_backup_list.txt`\
`dpkg -l | grep nvidia > ~/nvidia_backup_list.txt`
`cat ~/cuda_backup_list.txt`\
`cat ~/nvidia_backup_list.txt`\
Save these outputs somewhere safe. This is your rollback reference.

### Important clarification on your CUDA version
Verify first:\
bash\
`nvcc --version`\
`ls /usr/local/ | grep cuda`\
This matters for rollback — you need to know exactly what you have.
`____________________________________________________________________`
### Method choice — APT package manager (recommended)
The cleanest approach is to use NVIDIA's official APT repository and NVIDIA-provided Debian packages for **Ubuntu 24.04** instead of the standalone .run installer. NVIDIA officially supports **Ubuntu 24.04** for CUDA, and its Linux installation guide recommends distribution-specific packages where possible because they integrate with the system package manager more cleanly.

_______________________________________________________________________________
### Step 1 — Add NVIDIA APT repository (if not already added) ###
**Download and install the CUDA keyring**\
bash\
`wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb`\
`sudo dpkg -i cuda-keyring_1.1-1_all.deb`\
`sudo apt-get update`
________________________________________________________________________________
### Step 2 — Install CUDA 13.2 toolkit (versioned, not generic) ###
A key reason to use versioned package names is version control. The generic cuda and cuda-toolkit packages track newer versions automatically, which is not what you want when pinning to a specific version.\
bash\
**Install specifically versioned 13-2** — does NOT remove your existing CUDA\
`sudo apt-get install cuda-toolkit-13-2`\
This installs CUDA 13.2 alongside your existing CUDA 12.x — they coexist in separate directories:\
`/usr/local/cuda-12.8/`   ← your current installation, untouched\
`/usr/local/cuda-13.2/`   ← new installation\
`/usr/local/cuda`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;← symlink, currently points to 12
_______________________________________________________________________________________
### Step 3 — Switch between versions via symlink ###
bash\
**Switch to CUDA 13.2**\
`sudo rm /usr/local/cuda`\
`sudo ln -s /usr/local/cuda-13.2 /usr/local/cuda`

**Verify**\
`nvcc --version`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  \#should show 13.2
_____________________________________________________________________________________
### Step 4 — Update PATH in your .bashrc ###
bash\
**Check what you currently have:**\
`grep cuda ~/.bashrc`\
</br>
**It likely already has something like:**\
`export PATH=/usr/local/cuda/bin:$PATH`\
`export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`\
</br>
**These use the symlink /usr/local/cuda so they work automatically**\
**after you update the symlink — no .bashrc change needed**\
`source ~/.bashrc`\
`nvcc --version`
___________________________________________________________________________________________
### Step 5 — Verify your driver compatibility ###
bash\
`nvidia-smi`\
**Check that Driver Version is compatible with CUDA 13.2**\
**CUDA 13.2 requires driver >= 595.xx**\
CUDA 13.2.1 requires driver version 595.58.03. You already have nvidia-driver-595-open installed from your earlier system fix — so you should be compatible.
_______________________________________________________________________________________
### Step 6 — Test your existing code compiles ###
bash\
`cd ~/<XXX>/sgemm/sgemm_cublas`\
`mkdir build_13 && cd build_13`\
`cmake .. -DCMAKE_BUILD_TYPE=Release`\
`make -j$(nproc)`\
`./cublas_gemmex`
____________________________________________________________________________________
Rollback procedure — if anything goes wrong\
Since CUDA versions coexist, rollback is just switching the symlink back:\
bash\
**Instant rollback — one command**\
`sudo rm /usr/local/cuda`\
`sudo ln -s /usr/local/cuda-12.8 /usr/local/cuda\`

**Verify rollback**\
`nvcc --version  # back to 12.8`\
If you want to fully remove 13.2:\
bash\
`sudo apt-get remove cuda-toolkit-13-2`\
`sudo apt-get autoremove`
______________________________________________________________________________________
**Summary of the safety guarantees**\
Risk	Mitigation\
| Risk                              | Mitigation   |
| :---------                        | :----------: |
| Breaks existing cod               | Both versions coexist, symlink switch is instant         |
| :---------                        | :----------: |
|Driver incompatible | You already have 595 driver|
| :---------                        | :----------: |
|Can't roll back                 |Rollback is one ln -s command
| :---------                        | :----------: |
|Loses current install  | 	/usr/local/cuda-12.8 is untouched throughout|

Breaks existing code	Both versions coexist, symlink switch is instant
Driver incompatible	You already have 595 driver
Can't roll back	Rollback is one ln -s command
Loses current install	/usr/local/cuda-12.8 is untouched throughout
The APT method with versioned package names is safe precisely because it never touches your existing installation.





