## Step 1: Disable the blockers ##
Open cmd as Administrator and run these commands one by one:\n
**Disable hibernation** (frees up several GB):\
On cmd type:\
`powercfg /h off`\
**Disable page file** (temporarily):
1.	Search "Advanced System Settings" → click **Settings** under Performance
2.	Go to **Advanced** tab → **Change** under Virtual Memory
3.	Uncheck "Automatically manage" → select **No paging file** → click Set → OK
4.	**Restart your PC**
  
**Disable System Restore** (optional but helps):\
command on cmd:\
`Disable-ComputerRestore -Drive "C:\"`
_________________________________________________________________________________________________________
### Step 2: Defrag and clear unmovable files ###
Run this in cmd as Administrator:\
**Run the disk defragmenter to consolidate unmovable files to the front**\
`defrag C: /U /V`\
Then open **Event Viewer** → **Windows Logs** → **Application** and search for "defrag" to see if any files are still unmovable.
_________________________________________________________________________________________________________
### Step 3: Try shrinking again — but use diskpart for more control ###
At the CMD command line as Administrator:  

`diskpart`  
  
Then inside diskpart:  
  
`list disk`  
select disk 0  
list volume  
select volume 2   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     \# select your C: drive volume number  
shrink desired=900000 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \# ~900GB for Ubuntu, leaves ~800GB for Windows  
______________________________________________________________________________________________________________    
### 1.7TB free, a reasonable split is:  
•	**Windows C: drive:** keep ~200–300 GB free (so shrink by ~1.4–1.5TB)  
•	**Ubuntu**: 300–700 GB is very generous — even 100 GB is plenty for most Linux use cases  
______________________________________________________________________________________________________________  
If it still won't shrink far enough  
The nuclear option that always works: use **GParted** from the Ubuntu live USB itself to resize the partition before installing. Boot into the Ubuntu installer, open GParted (it's included), shrink C: from there, then proceed with installation. GParted bypasses Windows' restrictions entirely.  
  
To undo "Create a restore point"      
To undo/delete a restore point you created:    
**Option 1: Via System Properties (GUI)**  
1.	Press **Win + R** → type sysdm.cpl → Enter  
2.	Go to the **System Protection** tab  
3.	Click **Configure** next to your C: drive  
4.	Click **Delete** — this removes **all** restore points for that drive  
5.	Click Apply → OK  
_______________________________________________________________________________________________________________  
**Option 2: Via Disk Cleanup**  
1.	Search **Disk Cleanup** → Run it on C:  
2.	Click **Clean up system** files  
3.	Go to the **More Options** tab  
4.	Under System Restore and Shadow Copies, click **Clean up**  
5.	Confirm deletion  
_______________________________________________________________________________________________________________  
**Option 3: Via PowerShell (precise control)** 
List all restore points first:
At the CMD command line:       
Get-ComputerRestorePoint    
Then delete all of them:    
At the CMD command line:    
`vssadmin delete shadows /for=C: /all /quiet`  
________________________________________
**Note:** You can't delete a single specific restore point through built-in Windows tools — it's all or nothing. If you only want to free up space, Option 1 or the vssadmin command is the quickest.  
  
After deleting, go back to **Disk Management** and try shrinking C: again — you should be able to reclaim significantly more space.  
Please see the list volume below. I select volume 0, but I still cannot shrink the disk of 1400000MB.  
DISKPART> select disk 0  
Disk 0 is now the selected disk. 
`DISKPART> list volume`   

| Volumn \###      | Ltr Label | Fs | Type | Size | Status | Info
| :---------       | :----------: |:----------: |:----------: |:----------: |:----------: |:----------: |    
| Volume 0         | C Windows    |NTFS         | Partition |1835 GB |  Healthy|    Boot |     
| Volume 1         | SYSTEM       | FAT32       | Partition |   100 MB | Healthy  |  System |    
| Volume 2         | WinRE tools  | NTFS        |Partition  |  800 MB | Healthy   | Hidden   |  
| Volume 3         | BIOS_RVY     | NTFS        | Partition |    26 GB |  Healthy  |  Hidden|

**The correct diskpart commands**  
`select volume 0`  
shrink desired=900000  
This shrinks C: by ~900 GB, leaving ~935 GB for Windows which is plenty.  
________________________________________________________________________________________________________________  

**If it still fails — check what's blocking it**  
Run this in a separate PowerShell window (not diskpart):    
  
defrag C: /U /V  

Look for the line **"You cannot shrink a volume beyond..."** — it will tell you exactly how much you *can* shrink.  
To find the exact maximum shrinkable amount in diskpart:  
select volume 0  
shrink querymax  
This tells you the maximum MB you can actually shrink right now. Use that number (or slightly less) in your shrink command.  
________________________________________________________________________________________________________________________  
Most likely culprits if querymax is low  
| Issue                             | Fix  |
| :---------                        | :----------: |
| Hibernation file	                |powercfg /h off in PowerShell|
| Page file	                        |Disable it in Advanced System Settings → restart|
| System restore points	            |Delete them (as we did earlier)|
| MFT / unmovable NTFS files	      | Run defrag C: /U /V|

  

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

| Risk                              | Mitigation   |
| :---------                        | :----------: |
| Breaks existing cod               | Both versions coexist, symlink switch is instant         |
|Driver incompatible                | You already have 595 driver|
|Can't roll back                    |Rollback is one ln -s command
|Loses current install              | /usr/local/cuda-12.8 is untouched throughout|

The APT method with versioned package names is safe precisely because it never touches your existing installation.





