
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
The cleanest approach is to use NVIDIA's official APT repository and NVIDIA-provided Debian packages\ for **Ubuntu 24.04** instead of the standalone .run installer. NVIDIA officially supports Ubuntu 24.04\ for CUDA, and its Linux installation guide recommends distribution-specific packages where possible\ because they integrate with the system package manager more cleanly.


