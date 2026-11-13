<pre>
recommending setting up the SSH key on the Linux desktop before renting the Lambda H100 PCIe instance.
Check whether we already have an SSH key
On the RTX 5080 Linux desktop, open a terminal:
ls -la ~/.ssh

Step 1. Check whether we already have an SSH key
We might see files such as:
id_ed25519
id_ed25519.pub
id_rsa
id_rsa.pub

If developer already have an SSH key, developer can reuse it. However, It recommendind creating a dedicated 
SSH key for Lambda to keep the cloud experiments organized.

Step 2. Generate the Lambda SSH key
Run the following command on the Linux desktop:
ssh-keygen -t ed25519 -f ~/.ssh/lambda_h100 -C "lambda-h100"
We will see:
Generating public/private ed25519 key pair.
Enter passphrase (empty for no passphrase):
Recommending entering a passphrase to protect the private key.
Next, we will see:
Enter same passphrase again:
Enter the same passphrase again and press Enter.
After that, we should see output similar to:
The identification has been saved in
/home/xxx/.ssh/lambda_h100

The public key has been saved in
/home/xxx/.ssh/lambda_h100.pub

The key fingerprint is:
SHA256:...

Step 3. Verify the new SSH keys
Run:
ls -la ~/.ssh
We should now see three files:
authorized_keys
lambda_h100
lambda_h100.pub
Their purposes are:
lambda_h100 — Private key
Keep this file on the desktop. Never upload it to Lambda or share it with anyone.

lambda_h100.pub — Public key
This is the key we will copy into the Lambda Cloud account.
Step 4. Display The public key
Run:
cat ~/.ssh/lambda_h100.pub
We will see something like:
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... lambda-h100
    
Copy the entire line.
Then go to the Lambda Cloud console, navigate to SSH Keys, and add The public key.
Once we have generated the SSH keys, run ls -la ~/.ssh again to verify the two new files. We do not need 
to rent the H100 yet.
Output confirms that everything looks correct:
File              Permissions          Status
lambda_h100       -rw------- (600)     Private key — correct
lambda_h100.pub   -rw-r--r-- (644)     Public key — correct
authorized_keys   -rw------- (600)     Leave it unchanged

Step 1. Display the public key
On the Linux desktop, execute:
cat ~/.ssh/lambda_h100.pub

We should see a single line that looks similar to this:
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... lambda-h100

Select and copy the entire line, including ssh-ed25519 at the beginning and lambda-h100 at the end.
In the Ubuntu terminal, we can use Ctrl+Shift+C to copy the selected text.
Important: Copy only the contents of lambda_h100.pub. Never upload or share the contents of the private key file, 
lambda_h100.
Step 2. Add the public key to Lambda
Open the Lambda Cloud console and sign in.
1. Navigate to SSH Keys.
2. Click Add SSH Key.
3. Choose the option to add an existing public key.
4. Enter the following information:
Example — Add SSH Key
Key Name lambda-h100

Public Keyssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... lambda-h100
Paste the actual complete public key here, not this example.

5. Save the SSH key.
Lambda officially supports adding an existing public key and using it to access the GPU instances.

Step 3. Verify the preparation
Once Lambda displays the key named lambda-h100, we will have completed the SSH setup.
Step 4. Later, when we rent the H100
When we launch the single H100 PCIe instance, select the saved SSH key named lambda-h100.

Lambda will provide an IP address for the server. Then we can connect using:
ssh -i ~/.ssh/lambda_h100 ubuntu@SERVER_IP

Replace SERVER_IP with the actual Lambda server's IP address.

For now, the next step is simply to execute:
cat ~/.ssh/lambda_h100.pub

We can also begin configuring your GPU instance first.
1. Click Launch instance in Lambda Cloud.
2. Select 1 × H100 PCIe (80 GB).
3. Select the region and operating system image.
4. When prompted for an SSH key, choose to add the existing public key.
5. Paste the contents of lambda_h100.pub and name it lambda-h100.
6. Review the configuration and confirm the launch.

The important distinction: Clicking "Launch instance" to begin the configuration wizard does not mean we have 
already started renting the GPU. We can add the SSH key during this configuration process, 
before the final launch.
Once the H100 instance is running, Lambda will provide an IP address. The can then connect 
from the Linux desktop:
ssh -i ~/.ssh/lambda_h100 ubuntu@SERVER_IP
Replace SERVER_IP with the actual Lambda IP address.

Entering ssh key name, using the following name for the SSH key in Lambda:
lambda-h100

Fill in Lambda's SSH key form
Field             What to enter
SSH Key Name      lambda-h100
Public Key        Contents of ~/.ssh/lambda_h100.pub

If the goal is to rent one NVIDIA H100 PCIe GPU, connect to it using SSH, run the Hopper programs, and collect 
Nsight profiling reports.
We do not need multiple GPUs, a cluster of servers, or an H100 SXM.
The Lambda setup, one screen at a time.
Step 1. Click Instances
Go to the Lambda Cloud dashboard.
Click Instances in the navigation menu, then click Launch instance.
    
Step 2. Select the GPU
Look for the following option:
1 × NVIDIA H100 PCIe
Hopper architecture — SM90a  $3.29/hour
(Currently listed public rate, before tax)
GPU memory80 GB     CPU26 vCPUs
System RAM225 GiB   Local storage1 TiB SSD

Connect to the new H100
Once the instance is running, Lambda will show its public IP address.
On the Linux desktop, run:
ssh -i ~/.ssh/lambda_h100 ubuntu@YOUR_SERVER_IP
Replace YOUR_SERVER_IP with the actual IP address shown in the Lambda dashboard.

It is possible developer receive the connecting information:
<img width="913" height="1018" alt="image" src="https://github.com/user-attachments/assets/e3a83db6-fa8d-49ac-9d63-48e7aa7bb6bf" />

After connecting, the first command should be:
nvidia-smi
Then verify the profiling tools:
ncu --version
nsys --version

Recommendation:
Click the dropdown arrow on the right and check whether Lambda Stack 24.04 is available.
Operating system      Recommendation
Lambda Stack 24.04    Preferred, if available, if the desktop uses Ubuntu 24.04
Lambda Stack 22.04    Also fine for running and profiling Hopper programs

<img width="913" height="637" alt="image" src="https://github.com/user-attachments/assets/bdf3db63-5615-461c-8658-c5a9e2a65427" />
Select the first option: “Don't attach a filesystem.” It is the black button in the screenshot.
    
One important note: the CUDA Toolkit is preinstalled, confirmed the exact Nsight Compute and Nsight 
Systems versions installed in this particular image. Once we connect through SSH, we will check:
nvidia-smi
nvcc --version
ncu --version
nsys --version

SCP the executable to Lambda
On your Linux desktop, navigate to the project directory and execute:
scp -i ~/.ssh/lambda_h100 \
    ./build/wgmma_sm90 \
    ubuntu@SERVER_IP:~/

Verify the GPU and executable:
nvidia-smi
ls -lh ~/wgmma_sm90

Step 4. Execute the Hopper program
On Lambda:
cd ~
chmod +x wgmma_sm90
./wgmma_sm90

We can use the default GEMM dimensions. No command-line arguments are required.
Make sure the program executes successfully before profiling.

Step 5. Generate the Nsight Compute report
First, check that Nsight Compute is installed:
ncu --version

Then execute:
ncu \
    --set full -f \
    --kernel-name regex:gemm_device \
    --launch-skip 1 \
    --launch-count 1 \
    --export ./wgmma_sm90 \
    ./wgmma_sm90


This generates:
wgmma_sm90.ncu-rep

Step 6. Generate the Nsight Systems report
Check the version:
nsys --version

Then execute:
nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --output=wgmma_sm90 \
    ./wgmma_sm90

This generates:
wgmma_sm90.nsys-rep

wgmma_sm90 executable will work on H100 SXM5 too, assuming it was compiled for sm_90a and has compatible 
library dependencies.
Both H100 PCIe and H100 SXM5 support WGMMA, TMA, TMA multicast.
We do not need to change the CUDA source code or recompile specifically for SXM5.

Here is the cost comparison using the prices:
Rental duration     H100 PCIe ($3.29/hr)        H100 SXM5 ($4.29/hr)      Extra cost
30 minutes          $1.65                        $2.15                     $0.50
1 hour              $3.29                        $4.29                     $1.00
2 hours             $6.58	                     $8.58                     $2.00
3 hours             $9.87                        $12.87                    $3.00

nvidia-smi
Wed Nov 11 10:11:59 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:06:00.0 Off |                    0 |
| N/A   26C    P0             69W /  700W |       0MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|

~$ chmod +x ~/wgmma_sm90
~/wgmma_sm90
CUTE_GEMM:     [378214.7]GFlop/s  (0.5678)ms

Next step: Install Nsight Compute and Nsight Systems
Since the earlier commands showed that ncu and nsys were missing, let's address that next.

On the Lambda SSH terminal, execute:
apt-cache policy cuda-nsight-compute-12-8 cuda-nsight-systems-12-8


If both packages show available candidate versions, install them:
sudo apt update
sudo apt install -y \
    cuda-nsight-compute-12-8 \
    cuda-nsight-systems-12-8


Then verify:
ncu --version
nsys --version

hostname
whoami
echo "$SSH_CONNECTION"
nvidia-smi -L

If the installation succeeds, we can generate the first reports:

Nsight Compute:
ncu --set full \
    --launch-count 1 \
    -f -o wgmma_sm90 \
    ./wgmma_sm90

Nsight Systems:
nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    -o wgmma_sm90 \
    ./wgmma_sm90

Locate the installed Nsight tools:
dpkg -l | grep -i nsight
Then:
find /usr/local /opt -type f \
    \( -name ncu -o -name nsys \) \
    2>/dev/null

If neither executable is found, update the package list:
sudo apt update
apt-cache policy cuda-nsight-compute-12-8 cuda-nsight-systems-12-8

If both packages now show an available candidate, install them:
sudo apt install -y \
    cuda-nsight-compute-12-8 \
    cuda-nsight-systems-12-8

If APT still cannot locate them, Lambda may not have NVIDIA's CUDA repository configured. NVIDIA documents how to enable its Ubuntu 24.04 CUDA repository using the cuda-keyring package. 
Installation Guide for Linux 12.8 documentation


Run these commands on Lambda only:
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    -O /tmp/cuda-keyring.deb

sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt update

sudo apt install -y \
    cuda-nsight-compute-12-8 \
    cuda-nsight-systems-12-8

Finally, check:
ncu --version
nsys --version

cd ~

nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    -f true \
    -o wgmma_sm90 \
    ./wgmma_sm90


Then verify the report:
ls -lh ~/wgmma_sm90.nsys-rep
If successful, we have the first Hopper GPU execution timeline ready for analysis.

Step 2. Get Nsight Compute working
First, check the two standard NVIDIA installation locations on Lambda:
find /opt/nvidia /usr/local/cuda-12.8 \
    -type f -name ncu 2>/dev/null

If nothing appears, run:
sudo apt update
apt-cache policy cuda-nsight-compute-12-8

If the package has an available candidate, install it:
sudo apt install -y cuda-nsight-compute-12-8

If installation succeeds but ncu is still not found, check again:
find /opt/nvidia /usr/local/cuda-12.8 \
    -type f -name ncu 2>/dev/null


Once ncu is available, the initial profiling command will be:
ncu --set full \
    --launch-count 1 \
    -f -o wgmma_sm90 \
    ./wgmma_sm90

This should generate wgmma_sm90.ncu-rep, provided performance-counter access is permitted on the instance.

Step 1. Verify Nsight Compute
On the Lambda SSH terminal, execute:
/usr/local/cuda-12.8/bin/ncu --version
If that works, add it to the current terminal's PATH:
export PATH=/usr/local/cuda-12.8/bin:$PATH

Verify:
ncu --version
Step 2. Generate the first Nsight Compute report
Since wgmma_sm90 already runs successfully, execute:
cd ~

ncu \
    --set full \
    --launch-count 1 \
    -f \
    --export wgmma_sm90 \
    ./wgmma_sm90

This should generate:
/home/ubuntu/wgmma_sm90.ncu-rep

wgmma_sm90 program is working correctly on the H100. However, Nsight Compute cannot access the GPU hardware 
performance counters with the current user permissions.
This is a permissions problem, not a problem with the CUDA code or executable.

The first thing I recommend is running Nsight Compute with sudo. We do not need to recompile or reinstall anything.
Step 1. Try Nsight Compute as root
On the Lambda H100 SSH terminal, execute:
cd ~

sudo /usr/local/cuda-12.8/bin/ncu \
    --set full \
    --launch-count 1 \
    -f \
    --export wgmma_sm90 \
    ./wgmma_sm90

If the Lambda instance gives root access to the GPU performance counters, this should generate:
/home/ubuntu/wgmma_sm90.ncu-rep

NVIDIA's official troubleshooting guide recommends using sudo on Linux when performance counters are restricted 
to administrators

Step 2. Check whether the report was generated
After the command completes, run:
ls -lh ~/wgmma_sm90.ncu-rep
If the file exists and profiling completed successfully, make sure the regular ubuntu account can access it:
sudo chown ubuntu:ubuntu ~/wgmma_sm90.ncu-rep

User attachment
<img width="954" height="791" alt="image" src="https://github.com/user-attachments/assets/0649ea8a-6584-4bd9-9938-1b45aaf519bf" />
erase data on instance
Before proceeding, verify that these files are safely stored on the RTX 5080 Blackwell SM120 desktop(not Lambda datacenter):
~/hopper-reports/wgmma_sm90.ncu-rep
~/hopper-reports/wgmma_sm90.nsys-rep

We can check from the desktop terminal:
ls -lh ~/hopper-reports/
If both files are present and we have verified they open correctly, then:
Enter erase data on instance in the confirmation box.
Click Terminate instances.
Verify that the instance is no longer running in the Lambda dashboard.
Termination permanently deletes the files stored on that Lambda instance.

The SSH key and locally compiled CUDA executables will remain on the desktop.

xxx@xxxxxx:~/hopper-reports$ ls
wgmma_sm90.ncu-rep  wgmma_sm90.nsys-rep
Both profiling reports have been successfully downloaded to the RTX 5080 desktop.

The output confirms:
~/hopper-reports/
├── wgmma_sm90.ncu-rep
└── wgmma_sm90.nsys-rep
Next step: Terminate the Lambda H100
Because both files are on the desktop, we're almost ready to terminate the instance.
First, I recommend verifying that the reports open:
ncu-ui ~/hopper-reports/wgmma_sm90.ncu-rep
nsys-ui ~/hopper-reports/wgmma_sm90.nsys-rep
If both reports open successfully, return to the Lambda termination window.
Enter the exact phrase: erase data on instance
Click Terminate instances.
Verify that the instance is no longer running.
This permanently deletes the files on Lambda and stops GPU rental charges once termination completes.

Based on the Lambda H100 session this morning, I used at least approximately 15–16 minutes. The actual billable 
time will be slightly longer.

Here are the timestamps from the terminal outputs on November 11, 2026:
Massachusetts time (EST)
H100 verified with nvidia-smi: 5:11 AM
Nsight Systems report created: 5:22 AM
Nsight Compute report created: 5:27 AM
Estimated rental cost
The H100 SXM5 costs $4.29/hour.
Estimated cost
20 minutes: $1.43
30 minutes: $2.15
40 minutes: $2.86
60 minutes: $4.29

For the exact duration and cost, check the Lambda dashboard's billing or usage history after termination.

<img width="700" height="350" alt="image" src="https://github.com/user-attachments/assets/80b52127-0a76-4a56-bce4-16182e6e1bc6" />


</pre>
