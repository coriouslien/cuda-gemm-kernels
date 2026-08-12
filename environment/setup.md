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

### Install Ubuntu 24.04 - what to do ###  
1.	Create Ubuntu USB installer  
•	Download Ubuntu 24.04 LTS from ubuntu.com  
•	Flash it with Rufus (Windows) → choose GPT partition scheme  
•	Use a USB stick of 8GB+  
2.	Boot from USB  
•	Restart → spam F2, F12, or Del (depends on your motherboard) to enter boot menu  
•	Select your USB drive  
3.	In the Ubuntu installer  
•	When it asks about installation type, choose "Erase disk and install Ubuntu"  
•	This wipes Windows and uses the full disk — done

**One thing to do BEFORE wiping Windows**   
Make sure **Secure Boot** and **Fast Boot** are disabled in your BIOS — these can cause issues with NVIDIA drivers on Ubuntu. You'll be in the BIOS anyway when selecting the USB boot device.  

The desired shrink space.  
Based on the **1,835 GB** Volume 0 (C: drive), here's a simple breakdown:  
**If keeping dual boot**  
| Ubuntu Use       | Shrink by | Ubuntu Gets  | Windows Keeps   |
| :--------- | :----------: |:----------: |:----------: |
| Minimal	   | 500,000 MB	  | ~488 GB	    |~1,347 GB
| Recommended|	900,000 MB	|~879 GB	|~956 GB|
|Maximum	|1,400,000 MB	|~1,367 GB	|~468 GB|

For CUDA/AI development, **900,000 MB is the sweet spot** — enough for datasets, models, and Docker containers.  

If you don't need Windows...  
Just run this in diskpart to check the maximum possible:  
select volume 0  
shrink querymax  
Then use whatever number it returns. But honestly — don't bother shrinking at all.  
Since you're wiping Windows:  
•	Boot from Ubuntu USB  
•	Choose "Erase disk and install Ubuntu"  
•	You get the full 1,835 GB with zero hassle  
•	No diskpart gymnastics needed  
________________________________________________________________________________________________________________  
Recommendation: Skip the shrink entirely, boot Ubuntu USB, and erase the whole disk. It's 3 clicks in the installer and you're done.    
run shrink querymax, will it erase Windows 11?  
shrink querymax is completely safe — it is a read-only query that just tells you the maximum amount you can shrink.  
It only **reports a number**, it does nothing to your disk.  

_________________________________________________________________________________________________________________  
**What each command actually does**  
|Command	       |Does it change anything?|
|:---------      | :----------: |
|shrink querymax |No — just displays a number|
|shrink desired=900000	| Yes — actually shrinks the partition|
|format	       | Yes — erases data|
| delete volume | Yes — deletes partition|
  
**Safe to run right now**  
select volume 0  
shrink querymax  
It will just print something like:  
The maximum number of reclaimable bytes is: XXXXXX MB  
That number tells us exactly how much space we can work with. Go ahead and run it — nothing will change on your disk.
Please see below: DISKPART> select volume 0  
Volume 0 is the selected volume.  
DISKPART> shrink querymax  
The maximum number of reclaimable bytes is:  917 GB (939659 MB)    

Windows will only let you reclaim 917 GB due to unmovable system files blocking the rest. Here are your options:  
**Option 1: Shrink with diskpart right now (quick)**   
shrink desired=939659  
This gives Ubuntu **~917 GB** — perfectly fine for CUDA/AI work.   
_____________________________________________________________________________________________________________________  
**Option 2: Get the full 1.8TB (recommended if you don't need Windows)**
If you don't need Windows, skip the shrink entirely:  
1.	Make Ubuntu USB installer  
2.	Boot from it  
3.	Choose **"Erase disk and install Ubuntu"**  
4.	Get the full **1,835** GB  
917 GB vs 1,835 GB — that's **double the space** for your AI datasets and models, for zero extra effort.  
________________________________________
Recommendation:  
Go with **Option 2**. For CUDA/AI development, storage fills up fast:  
•	Large language models = 10–70 GB each  
•	Training datasets = can be hundreds of GB  
•	Docker images = 5–20 GB each  
•	Virtual environments = stack up quickly  
The full 1.8TB will serve you much better in the long run, and erasing via the Ubuntu installer is actually easier.  
  
Option 2 WILL erase Windows 11.  
What each option does  
|Option	                        |Erases Windows?	|Ubuntu Space|
|:---------                     | :----------: | :----------: |
|Option 1 (shrink desired=939659)|No — keeps Windows|	~917 GB|
|Option 2 (Erase disk)	         |Yes — wipes everything|	~1,835 GB|

Earlier, if you don't need Windows —  Option 2.  
But if you want to keep Windows as a backup, go with Option 1:  
shrink desired=939659  
Then install Ubuntu on that freed space.  
_____________________________________________________________________________________________________________________  
•	**Keep Windows** (just in case) → use Option 1  
•	**Wipe Windows completely** (full disk for Ubuntu) → use Option 2  
917 GB is still very generous for CUDA/AI development, so Option 1 is not a bad choice even if you end up never using Windows again.  
Don't format anything in diskpart! Leave the unallocated space as-is.    
The Ubuntu installer will handle all the partitioning and formatting automatically.  
____________________________________________________________________________________________________________________  
What to do now 
1.	Exit diskpart 
   `exit`  
2.	Create Ubuntu USB (if you haven't yet)  
•	Download Ubuntu 24.04 LTS from ubuntu.com  
•	Flash with Rufus → select GPT partition scheme  
•	Use 8GB+ USB stick  
3.	Before rebooting — note this down  
•	Disable Secure Boot in BIOS  
•	Disable Fast Boot in BIOS  
•	These two cause the most NVIDIA/CUDA driver problems on Ubuntu  
4.	Boot from USB → spam F2, F12, or Del at startup  
5.	In Ubuntu installer  
•	Choose "Install Ubuntu alongside Windows"  
•	It will automatically detect the 917 GB unallocated space  
•	Select that space for Ubuntu  
____________________________________________________________________________________________________________________  
**Partition layout Ubuntu will create automatically**  
|Partition	|Size	|Purpose|
|:--------- | :----------: | :----------: |
|EFI	    |~512 MB	     |Boot|
|swap       |	~16 GB      |	RAM overflow|
|root	    |Remaining ~900 GB|	Everything else|
You don't need to manually set this up — the installer handles it.  
You should disabl boot. There are two options in "Secure Boot Mode", Standard and Custom. You should select **"Standard"**    
Why **"Standard":  
•	**Standard** = uses Microsoft's trusted certificates — works fine for Ubuntu  
•	**Custom** = for manually importing your own certificates — not needed here  
___________________________________________________________________________________________________________________
**Full BIOS checklist for MSI before you boot Ubuntu USB** (I use MSI hardware)   
While you're in BIOS, make sure:    
|Setting	   |What to set|  
|:--------- | :----------: |
| Secure Boot Mode| **Standard**  |
| Secure Boot |	**Disabled**  |
| Fast Boot	| **Disabled**  |
| Boot mode	| **UEFI** (not Legacy/CSM)|  
**How to disable Secure Boot on MSI**   
1.	Go to **Settings → Security → Secure Boot**    
2.	Set Secure Boot to **Disabled**     
3.	Press **F10** to save and exit    
Then plug in your Ubuntu USB and boot from it — on MSI you can hit **F11** at startup for the one-time boot menu to select your USB drive directly without changing boot order permanently. (**Note**: MIS is F11)  
  
If you don't see "Fast Boot" option, but you have the following set up:  
Full Screen Logo Display: Enable    
Bootup NumLock State: On    
POST Beep: Disable    
FIXED BOOT ORDER Priorities:    
Boot Option #1: UEFI Hard Disk: Windows Boot Manage    
Boot Option #1: UEFI CD/DVE   
Boot Option #3 UEFI USB Hard disk    
    
Change Boot Order to prioritize USB    
Change your boot order to:    
Priority	Set to    
Boot Option #1	UEFI USB Hard Disk    
Boot Option #2	UEFI Hard Disk: Windows Boot Manager    
Boot Option #3	UEFI CD/DVD    
This makes your PC boot from the Ubuntu USB first.    
_____________________________________________________________________________________________________________________  
Fast Boot on MSI — where to find it  
Look in these locations:  
•	Settings → Advanced → Power Management Setup → Fast Boot  
•	OC → CPU Features → Fast Boot  
•	It may also be called "Windows 10 WHQL Support" on some MSI boards  
If you genuinely can't find it, don't worry — it's not critical. Ubuntu will likely install fine without disabling it.  
_____________________________________________________________________________________________________________________  
After changing boot order  
1.	Press F10 to save and exit  
2.	Make sure your Ubuntu USB is plugged in  
3.	PC will reboot and boot from USB automatically  
4.	Ubuntu installer will launch  
  
Boot Option #4 UEFI USB CD/DVD  
Boot Option #5: UEFI USB Key: UEFI: USB disk 3.0  
Boot Option #6: UEFI USB Floopy  
  
You already have your Ubuntu USB recognized!
Change Boot Option #1 to your USB  
Set it to:  
|Priority	|Set to|
|:--------- | :----------: |
|Boot Option #1	|**UEFI USB Key: UEFI: USB disk 3.0** | 
|Boot Option #2	|UEFI Hard Disk: Windows Boot Manager | 
|Others	|leave as is | 
**"UEFI USB Key: UEFI: USB disk 3.0"** is your Ubuntu USB drive.  
______________________________________________________________________________________________________________________  
Then  
1.	Press **F10** to save and exit  
2.	PC will reboot and boot directly into Ubuntu installer  
3.	You're good to go!  
______________________________________________________________________________________________________________________  
**What you'll see in the Ubuntu installer**    
•	**"Try or Install Ubuntu"** → press Enter  
•	Wait for desktop to load  
•	Click **"Install Ubuntu"**  
•	Choose "Install Ubuntu alongside Windows Boot Manager"  
•	It should detect your 917 GB unallocated space automatically  
  
If you have OC Explore Mode options Expert and Normal.
That's not Fast Boot. OC Explore Mode is different:  
|Option |	What it does  |
|:--------- | :----------: |
|**Normal**	|Shows basic overclocking options|  
|**Expert** |	Shows advanced overclocking options  |
This is only for CPU/RAM overclocking settings — nothing to do with boot speed or Ubuntu.  
___________________________________________________________________________________________________________________  
Leave it on Normal and don't worry about Fast Boot  
On your MSI board, Fast Boot either:  
•	Doesn't exist on your specific model  
•	Is not enabled by default  
Either way, it won't stop Ubuntu from installing. Many people install Ubuntu successfully without disabling Fast Boot.  
____________________________________________________________________________________________________________________  
You're good to proceed  
Just focus on what matters:  
•	Secure Boot → Disabled  
•	Boot Option #1 → UEFI USB Key: USB disk 3.0  
•	Press F10 to save and exit  
That's all you need. Plug in your Ubuntu USB if you haven't already and save & exit BIOS.  
  
Follow these steps in the installer  
1.	Language → Select English (or your preference) → Continue  
2.	Keyboard layout → Continue  
3.	Connect to internet → Skip for now (install offline, update later)  
4.	What to install  
•	Choose "Ubuntu" (full install, not minimal)  
•	Check "Install third-party software for graphics" — this is important for NVIDIA drivers!  
5.	Installation type  
•	Choose "Install Ubuntu alongside Windows Boot Manager"  
•	It should show your 917 GB unallocated space  
6.	Username & password → set these up  
7.	Click Install → wait ~15-20 minutes  
____________________________________________________________________________________________________________________  
The most important checkbox 
"Install third-party software for graphics and Wi-Fi hardware"  
This pre-installs NVIDIA drivers which saves you a lot of hassle for CUDA later.  
____________________________________________________________________________________________________________________  
On the "Installation type" screen, as that's the most critical step where you want to make sure it's not set to erase Windows,
if you want to keep Windows.  
Partitioning was handled automatically by the installer — you don't need to do anything there.
______________________________________________________________________________________________________________________  
Set Ubuntu as default boot (GRUB setup) 
When you reboot, you'll see the GRUB menu with Ubuntu and Windows options. To make Ubuntu the default: 
Open Terminal in Ubuntu and run: 
bash 
`sudo nano /etc/default/grub`
  
Find this line:  
GRUB_DEFAULT=0  
- `0` = Ubuntu (first option) — **already default**  
- If Windows is booting first, change it to `0`  
  
Also check this line:  
GRUB_TIMEOUT=10  
Change 10 to 5 to reduce the wait time to 5 seconds.  
Save and apply:  
bash  
# Press Ctrl+X → Y → Enter to save in nano #
`sudo update-grub`
____________________________________________________________________________________________________________________  
Verify your partitions were created correctly  
bash  
`lsblk`
You should see something like:   
|Partition	| Size	| Mount|
|:--------- | :----------: |:----------: |
|sda1	      |100 MB	       |/boot/efi   | 
|sda2       |	~900 GB      |	/    |
|swap       |	~16 GB	     |swap   |
**Next step — NVIDIA/CUDA setup**      
Run this to check if NVIDIA drivers are working:   
bash  
nvidia-smi    
  
My Partition Layout  
|Partition	|Size	|Purpose|
|:--------- | :----------: |:----------: |
|nvme0n1p1	|100MB	|EFI/Boot|
|nvme0n1p2	|128MB	|Windows MSR|
|nvme0n1p3	|917.8GB	|**Windows C: drive**|
|nvme0n1p4	|800MB	|Windows Recovery|
|nvme0n1p5	|26.5GB	|BIOS Recovery|
|nvme0n1p6	|917.6GB	|**Ubuntu** / (your root)|
Ubuntu is installed on **nvme0n1p6** with 917.6GB — perfect.
___________________________________________________________________________________________________________________  
**No Swap — Add a Swapfile instead**  
No swap partition is fine. Create a swapfile (easier and flexible):  
bash  
**Create 16GB swapfile**   
`sudo fallocate -l 16G /swapfile`  
`sudo chmod 600 /swapfile`  
`sudo mkswap /swapfile`  
`sudo swapon /swapfile`  

**Make it permanent**
`echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab`

**Verify**
`free -h`  
__________________________________________________________________________________________________________________    
Next — Check NVIDIA drivers  
bash  
`nvidia-smi`  
  
Should I use Swap = 32 GB instead of 16GB?
For CUDA/AI development, 32GB swap is better. Here's why:  
Recommendation: 32GB Swap  
Your Use Case	Recommended Swap  
General desktop	8 GB  
Software development	16 GB  
CUDA/AI/ML development	32 GB  
Large model training	64 GB  
Why 32GB for AI/ML  
•	Large models (LLMs) can overflow GPU VRAM into system RAM, then into swap  
•	PyTorch/TensorFlow can use significant RAM during training  
•	You have 917 GB of space — 32GB swap is only 3% of your disk, well worth it  
___________________________________________________________________________________________________________________  
Create 32GB Swapfile  
bash  
**Create 32GB swapfile**
`sudo fallocate -l 32G /swapfile`  
`sudo chmod 600 /swapfile`  
`sudo mkswap /swapfile`  
`sudo swapon /swapfile`  
  
**Make it permanent after reboot**
`echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab`  
  
**Verify**
`free -h`  
You should see 32GB in the swap row.  
__________________________________________________________________________________________________________________  
RAM and swap:  
•	16GB RAM → 32GB swap  
•	32GB RAM → 32GB swap  
•	64GB RAM → 32GB swap is still fine for AI work  

My GPU Summary  
	
|GPU|	NVIDIA GeForce RTX 5080|
|:--------- | :----------: |
|VRAM|	16GB|
|Driver	|590.48.01 |
|CUDA	|13.1 |
|Temp	|33°C  (very cool)|



   
_____________________________________________________________________________________________________________  


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





