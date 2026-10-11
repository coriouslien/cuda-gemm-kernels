<pre>
Below is a complete procedure. The main idea is to do all the preparation on my own machine first, so the
paid time on the H100 is spent only on building, running and profiling (nsys and ncu).

A few details below I could not confirm from Lambda’s own pages today: the exact button names, the billing
increment, and the current price. I mark those. Check them on Lambda’s site when I sign up, and give an 
update later.

Part A: Before renting (free, on my machine, RTX5080 Blackwell, sm120)

1. Create a Lambda account
Sign up at lambda.ai (the Cloud / GPU instances section) and add a payment method. On-demand instances are
billed while they run.
https://cloud.lambda.ai/
Where to choose the GPU
On lambda.ai, sign up or log in.
Go to the cloud console: https://cloud.lambda.ai, then the Instances page (https://cloud.lambda.ai/instances).
Click “Launch instance”.
A list of GPU types appears, by model name, for example “1x H100 (80 GB PCIe)”. Pick one, then the region,
then the SSH key.

(This is from a third-party guide to Lambda’s console, not Lambda’s own documentation; the page layout may
differ slightly.)
</pre>
Which model is which architecture
|GPU model in<br> the list	|Architecture	|Can it run wgmma_sm90.cu?|
|:-------------------|:----------------------|:------------------------|
|H100 PCIe|	Hopper, SM90|	Yes. Choose this one.|
|H100 SXM	|Hopper, SM90|	Yes (faster memory, higher price)|
|GH200	|Hopper GPU + ARM<br> Grace CPU|	The GPU is right, but the ARM CPU complicates setup.<br> Avoid.|
|B200/GB200|Blackwell, SM100|	No. SM100 uses tcgen05, not WGMMA. A kernel<br> compiled for sm_90a can’t run there.|
|A100,A10,<br>A6000|	Ampere, SM80/86|	No. No WGMMA.|
<pre>
So for your tutorial: H100 PCIe, or H100 SXM if PCIe is unavailable.

If an option is greyed out or missing, it’s temporarily sold out in every region. Try again later, or check another region.

Three details from the same guide that affect your plan
The default image is Ubuntu 22.04 with CUDA 12.x (“Lambda Stack”). Ubuntu 22.04’s CMake is 3.22, so keep cmake_minimum_required(VERSION 3.18) in your CMakeLists.txt. Don’t raise it to 3.24 as I suggested earlier, or the configure step may fail there.
Billing is per minute, and there is no stop or pause, only terminate. An idle instance keeps costing money until you terminate it.
Local data is deleted on terminate. Copy your results back with scp before terminating, or attach a persistent filesystem at launch time.

Check the CUDA version once you’re connected (nvcc --version). sm_90a needs CUDA 12.0 or newer.
</pre>
Sources:
<sub>[Lambda Labs GPU Cloud – Hermes Agent docs(third-party guide)](https://hermes-agent.nousresearch.com/docs/user-guide/skills/optional/mlops/mlops-lambda-labs)</sub><br>
<sub>[H100 PCIe on Lambda Labs – GPUPerHour](https://gpuperhour.com/providers/lambda-labs)</sub><br>
<pre>    
2. Create an SSH key if you don’t already have one:

ssh-keygen -t ed25519 -C "lambda"          # accept the default path ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub                  # copy this public key

In the Lambda dashboard, find the SSH keys page and add this public key. Never upload the private key (the file without .pub).

3. Write down the CUTLASS commit you use locally, so you build the same code on the H100:

bash
cd ~/cuda-learning/cutlass && git log -1 --format=%H

4. Make your CMakeLists.txt portable. The H100 machine may run an older Ubuntu with an older CMake. If you raised cmake_minimum_required to 3.24 as I suggested, it may fail there. Keep it at 3.18 (it worked for you), or check the remote version first (step 9).

5. Prepare one script that does everything on the H100. Save as run_on_h100.sh in your project folder:
#!/usr/bin/env bash
set -euo pipefail
COMMIT=<paste your CUTLASS commit hash>

# 1. Get CUTLASS at the same commit you use locally
cd ~
[ -d cutlass ] || git clone https://github.com/NVIDIA/cutlass.git
cd cutlass && git checkout "$COMMIT" && cd ~

# 2. Build both configurations
cd ~/sm90
mkdir -p logs results
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake -S . -B build-debug   -DCMAKE_BUILD_TYPE=Debug
cmake --build build-release 2>&1 | tee logs/build-release.log
cmake --build build-debug   2>&1 | tee logs/build-debug.log

# 3. Run both (default arguments = NT, 5120 x 5120 x 4096)
./build-release/wgmma_sm90 2>&1 | tee results/run-release.txt
./build-debug/wgmma_sm90   2>&1 | tee results/run-debug.txt

# 4. Timeline profile (nsys)
nsys profile -o results/nsys-release --force-overwrite true ./build-release/wgmma_sm90

# 5. Kernel profile (ncu): one launch of gemm_device, after skipping 2 warm-up launches
NCU=$(command -v ncu)
sudo "$NCU" --set full -k regex:gemm_device --launch-skip 2 -c 1 \
     -o results/ncu-release -f ./build-release/wgmma_sm90
sudo "$NCU" --set full -k regex:gemm_device --launch-skip 2 -c 1 \
     -o results/ncu-debug   -f ./build-debug/wgmma_sm90

Adjust the paths: this assumes your project folder will be ~/sm90 on the remote machine and that CUTLASS sits next to it as ~/cutlass. Your CMakeLists.txt uses ../cutlass/include, so the two folders must be siblings. If your CMakeLists also refers to ../include, copy that folder too.

6. Test the build locally once more. If cmake --build succeeds for sm_90a on your 5080, the same project will build on the H100.

Part B: Rent the instance

7. Launch. In the dashboard, choose to launch an instance, then:

Instance type: 1× H100 (PCIe). A third-party price tracker currently lists it at about $2.49/hour; confirm the price on Lambda’s pricing page. If PCIe isn’t available, 1× H100 SXM works too, at a higher price.
Avoid GH200. It’s also Hopper, but its CPU is ARM (aarch64). Your x86 tooling and any copied binaries won’t run, and setup differs.
Region: any with availability.
SSH key: select the key from step 2.
Filesystem (optional): Lambda offers persistent filesystems in some regions. Without one, everything on the instance is deleted when you terminate it, which is fine if you copy your results back (step 13).

8. Wait until the status shows it running, then copy the instance’s IP address from the dashboard.

Part C: Work on the instance

9. Connect and check the environment:
ssh ubuntu@<IP>
nvidia-smi                 # should show the H100
nvcc --version             # needs CUDA 12.x for sm_90a
cmake --version
which ncu nsys
Lambda’s default image usually includes the CUDA toolkit and Nsight tools, but verify, don’t assume. If something is missing, tell me what nvcc --version and which ncu print and we’ll decide what to install.

10. Copy your project. From a terminal on your own machine:
scp -r ~/cuda-learning/cute_hopper/sm90 ubuntu@<IP>:~/
11. Run the script on the instance:
cd ~/sm90 && bash run_on_h100.sh
About sudo ncu: by default, reading GPU performance counters needs administrator rights. A Lambda forum thread shows users get the permission error without sudo, and that ncu works with sudo. The permanent fix (a driver setting plus a reboot) isn’t worth the risk on a rented machine, so the script uses sudo.

12. Look at the results while you’re still connected:
grep -i -E "time|gflop|tflop|pass|fail" results/run-*.txt
grep -E "C7510|registers" logs/build-*.log

Check that the Debug build shows C7510 and Release doesn’t, as on your machine.

Part D: Copy results back, then terminate

13. Download everything (from your own machine):
scp -r ubuntu@<IP>:~/sm90/results ~/cuda-learning/cute_hopper/sm90/results-h100
scp -r ubuntu@<IP>:~/sm90/logs    ~/cuda-learning/cute_hopper/sm90/logs-h100
scp -r ubuntu@<IP>:~/sm90/results ~/cuda-learning/cute_hopper/sm90/results-h100
scp -r ubuntu@<IP>:~/sm90/logs    ~/cuda-learning/cute_hopper/sm90/logs-h100
</pre>
|Step|	Rough time|
|:----|:-------------|
|Launch and environment check|	5–10 min|
|Clone CUTLASS, copy project, build both	|10–15 min|
|Runs + nsys + 2 × ncu --set full	|10–20 min|
|Checking results, copying back	|10 min|
|Total	about 1 |hour|

<pre>
At roughly $2.50/hour that’s a few dollars. Budget for 2 hours in case something needs fixing. Check Lambda’s pricing page for the billing increment.

What this gives you

For the first time, a measured answer to the question your write-up raises: how much does C7510 serialization actually cost? Run time and TFLOP/s of Debug vs Release on the same H100, plus ncu’s view of tensor-core utilization in each. That’s part 2 of your notes.
</pre>
<small>Source:</small><br>
<sub>[H100 PCIe on Lambda Labs – GPUPerHour](https://gpuperhour.com/providers/lambda-labs)</sub><br>
<sub>[Running Nvidia NSight – permissions issue (Lambda forum)](https://deeptalk.lambda.ai/t/running-nvidia-nsight-permissions-issue/4433)</sub><br>
<sub>[NVIDIA: ERR_NVGPUCTRPERM solutions](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute)</sub><br>




