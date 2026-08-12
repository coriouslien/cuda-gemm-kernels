
## Before start - record current state\
bash\
Record everything about your current installation\
`nvcc --version`\
nvidia-smi
dpkg -l | grep cuda > ~/cuda_backup_list.txt
dpkg -l | grep nvidia > ~/nvidia_backup_list.txt
cat ~/cuda_backup_list.txt
cat ~/nvidia_backup_list.txt
Save these outputs somewhere safe. This is your rollback reference.

