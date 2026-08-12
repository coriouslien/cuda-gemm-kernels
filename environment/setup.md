
### Before start - record current state
bash\
Record everything about your current installation\
`nvcc --version`\
`nvidia-smi`\
`dpkg -l | grep cuda > ~/cuda_backup_list.txt`\
`dpkg -l | grep nvidia > ~/nvidia_backup_list.txt`
`cat ~/cuda_backup_list.txt`\
`cat ~/nvidia_backup_list.txt`\
Save these outputs somewhere safe. This is your rollback reference.\

### Important clarification on your CUDA version
Verify first:\
bash\
`nvcc --version`\
`ls /usr/local/ | grep cuda`\
This matters for rollback — you need to know exactly what you have.\
`____________________________________________________________________`
### Method choice — APT package manager (recommended)
The cleanest approach is to use NVIDIA's official APT repository and NVIDIA-provided Debian packages for **Ubuntu 24.04** instead of the standalone .run installer. NVIDIA officially supports **Ubuntu 24.04** for CUDA, and its Linux installation guide recommends distribution-specific packages where possible because they integrate with the system package manager more cleanly.\

_______________________________________________________________________________
### Step 1 — Add NVIDIA APT repository (if not already added) ###
**Download and install the CUDA keyring**\
bash\
`wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb`\
`sudo dpkg -i cuda-keyring_1.1-1_all.deb`\
`sudo apt-get update`\
________________________________________________________________________________
### Step 2 — Install CUDA 13.2 toolkit (versioned, not generic) ###
A key reason to use versioned package names is version control. The generic cuda and cuda-toolkit packages track newer versions automatically, which is not what you want when pinning to a specific version.\
bash\
** Install specifically versioned 13-2 ** — does NOT remove your existing CUDA\
`sudo apt-get install cuda-toolkit-13-2`\
This installs CUDA 13.2 alongside your existing CUDA 12.x — they coexist in separate directories:\
`/usr/local/cuda-12.8/`   ← your current installation, untouched\
`/usr/local/cuda-13.2/`   ← new installation\
`/usr/local/cuda`____     ← symlink, currently points to 12\
_______________________________________________________________________________________


