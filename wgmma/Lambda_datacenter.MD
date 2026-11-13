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

Step 2. Generate your Lambda SSH key
Run the following command on your Linux desktop:
ssh-keygen -t ed25519 -f ~/.ssh/lambda_h100 -C "lambda-h100"
You will see:
Generating public/private ed25519 key pair.
Enter passphrase (empty for no passphrase):
Recommending entering a passphrase to protect your private key.
Next, you will see:
Enter same passphrase again:
Enter the same passphrase again and press Enter.
After that, you should see output similar to:
Your identification has been saved in
/home/xxx/.ssh/lambda_h100

Your public key has been saved in
/home/xxx/.ssh/lambda_h100.pub

The key fingerprint is:
SHA256:...

Step 3. Verify your new SSH keys
Run:
ls -la ~/.ssh
You should now see three files:
authorized_keys
lambda_h100
lambda_h100.pub
Their purposes are:
lambda_h100 — Private key
Keep this file on your desktop. Never upload it to Lambda or share it with anyone.

lambda_h100.pub — Public key
This is the key you will copy into your Lambda Cloud account.
Step 4. Display your public key
Run:
cat ~/.ssh/lambda_h100.pub
You will see something like:
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... lambda-h100
Copy the entire line.
Then go to the Lambda Cloud console, navigate to SSH Keys, and add your public key.
Once you have generated your SSH keys, run ls -la ~/.ssh again to verify the two new files. You do not need 
to rent the H100 yet.
Output confirms that everything looks correct:
File              Permissions          Status
lambda_h100       -rw------- (600)     Private key — correct
lambda_h100.pub   -rw-r--r-- (644)     Public key — correct
authorized_keys   -rw------- (600)     Leave it unchanged

Step 1. Display your public key
On your Linux desktop, execute:
cat ~/.ssh/lambda_h100.pub

You should see a single line that looks similar to this:
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... lambda-h100

Select and copy the entire line, including ssh-ed25519 at the beginning and lambda-h100 at the end.
In your Ubuntu terminal, you can use Ctrl+Shift+C to copy the selected text.
Important: Copy only the contents of lambda_h100.pub. Never upload or share the contents of your private key file, 
lambda_h100.
Step 2. Add your public key to Lambda
Open the Lambda Cloud console and sign in.
1. Navigate to SSH Keys.
2. Click Add SSH Key.
3. Choose the option to add an existing public key.
4. Enter the following information:
Example — Add SSH Key
Key Namelambda-h100

Public Keyssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... lambda-h100
Paste your actual complete public key here, not this example.

5. Save the SSH key.
Lambda officially supports adding an existing public key and using it to access your GPU instances.

Step 3. Verify your preparation
Once Lambda displays your key named lambda-h100, you will have completed the SSH setup.
Step 4. Later, when you rent your H100
When you launch your single H100 PCIe instance, select your saved SSH key named lambda-h100.

Lambda will provide an IP address for your server. Then you can connect using:
ssh -i ~/.ssh/lambda_h100 ubuntu@SERVER_IP

Replace SERVER_IP with your actual Lambda server's IP address.

For now, your next step is simply to execute:
cat ~/.ssh/lambda_h100.pub

You can also begin configuring your GPU instance first.
1. Click Launch instance in Lambda Cloud.
2. Select 1 × H100 PCIe (80 GB).
3. Select your region and operating system image.
4. When prompted for an SSH key, choose to add your existing public key.
5. Paste the contents of lambda_h100.pub and name it lambda-h100.
6. Review the configuration and confirm the launch.

The important distinction: Clicking "Launch instance" to begin the configuration wizard does not mean you have 
already started renting the GPU. You can add your SSH key during this configuration process, 
before the final launch.
Once your H100 instance is running, Lambda will provide an IP address. You can then connect 
from your Linux desktop:
ssh -i ~/.ssh/lambda_h100 ubuntu@SERVER_IP
Replace SERVER_IP with your actual Lambda IP address.

Entering ssh key name, using the following name for your SSH key in Lambda:
lambda-h100

Fill in Lambda's SSH key form
Field             What to enter
SSH Key Name      lambda-h100
Public Key        Contents of ~/.ssh/lambda_h100.pub

If your goal is to rent one NVIDIA H100 PCIe GPU, connect to it using SSH, run your Hopper programs, and collect 
Nsight profiling reports.
You do not need multiple GPUs, a cluster of servers, or an H100 SXM.
The Lambda setup, one screen at a time.
Step 1. Click Instances
Go to your Lambda Cloud dashboard.

Click Instances in the navigation menu, then click Launch instance.
Step 2. Select your GPU
Look for the following option:
1 × NVIDIA H100 PCIe
Hopper architecture — SM90a  $3.29/hour
(Currently listed public rate, before tax)
GPU memory80 GB     CPU26 vCPUs
System RAM225 GiB   Local storage1 TiB SSD


Connect to your new H100
Once the instance is running, Lambda will show its public IP address.
On your Linux desktop, run:
ssh -i ~/.ssh/lambda_h100 ubuntu@YOUR_SERVER_IP
Replace YOUR_SERVER_IP with the actual IP address shown in your Lambda dashboard.

After connecting, your first command should be:
nvidia-smi
Then verify the profiling tools:
ncu --version
nsys --version

Recommendation:
Click the dropdown arrow on the right and check whether Lambda Stack 24.04 is available.
Operating system      Recommendation
Lambda Stack 24.04    Preferred, if available, if your desktop uses Ubuntu 24.04
Lambda Stack 22.04    Also fine for running and profiling Hopper programs

One important note: the CUDA Toolkit is preinstalled, confirmed the exact Nsight Compute and Nsight 
Systems versions installed in this particular image. Once you connect through SSH, we will check:
nvidia-smi
nvcc --version
ncu --version
nsys --version

SCP your executable to Lambda
On your Linux desktop, navigate to your project directory and execute:
scp -i ~/.ssh/lambda_h100 \
    ./build/wgmma_sm90 \
    ubuntu@SERVER_IP:~/

Verify your GPU and executable:
nvidia-smi
ls -lh ~/wgmma_sm90

Step 4. Execute your Hopper program
On Lambda:
cd ~
chmod +x wgmma_sm90
./wgmma_sm90

You can use the default GEMM dimensions. No command-line arguments are required.
Make sure the program executes successfully before profiling.



</pre>
