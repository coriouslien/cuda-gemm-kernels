**Launch config (40,40,1)×(128,1,1), duration 1.33ms.**<br>
**GPU Speed Of Light Throughput**
<img width="1826" height="1024" alt="GPU_Speed_Of_Light_Throughput" src="https://github.com/user-attachments/assets/bfaaba08-7462-411b-9214-cc057348f4d2" />
<pre>
NCU flags High Throughput — over 80% compute utilization. The bottleneck is clearly the SM compute pipe, 
not memory bandwidth. To go further you need to shift work away from the most saturated unit.
Compute (SM) Throughput: 82.19% — very high, the kernel is compute-bound
Memory Throughput: 63.48%
The compute units are "busy", but they are largely busy stalling (specifically, Wait Stalls averaging 4.5 
cycles per instruction). Because the kernel requests too much shared memory per block, occupancy is crushed
down to ~8%. The SM doesn't have enough active warps to swap between to hide the latency of its compute 
operations. Warp State Statistics and Occupancy charts reveal that the SMs are highly inefficient.
The kernel is currently bound by compute latency, but the root cause preventing the GPU from hiding that
latency is the severe shared memory occupancy limit.
  
L1/TEX: 43.55%, L2: 63.48% — moderate cache utilization
DRAM: 11.11% — very low, meaning data is being served mostly from L2, not DRAM
</pre>
**Memory Workload Analysis**
<img width="1826" height="1024" alt="Memory_workload_Analysis" src="https://github.com/user-attachments/assets/cfdac2c1-37a3-47f7-9d5c-a261b5467b77" />
<pre>
Excellent L2 caching but poor L1 efficiency. L1/TEX Hit Rate is 2.83%, extremely low, almost no L1 reuse. 
The kernel achieves a very high L2 Hit Rate of 97.20%, data is almost entirely served from L2, meaning most
memory requests are served quickly without going to main memory.
  
The near-zero L1 hit rate combined with a 97% L2 hit rate, the access pattern is bypassing L1 entirely 
(consistent with LDGSTS.E.BYPASS — global-to-shared async copies that intentionally skip L1) and the working 
set fits comfortably in L2. DRAM at 11% confirms L2 is absorbing almost everything.
</pre>
**Scheduler Statistics**
<img width="1826" height="1024" alt="Scheduler_Statistics1" src="https://github.com/user-attachments/assets/05a3e3c5-68f8-498e-b31d-3c03fde9287c" />

<img width="1826" height="1024" alt="Scheduler_statistics2" src="https://github.com/user-attachments/assets/a2772075-2cda-4776-862c-809396dbd3db" />
<pre>
This is a critical bottleneck:
Active Warps Per Scheduler: 1.00 (hardware max: 12)
Eligible Warps Per Scheduler: 0.16
Issued Warp Per Scheduler: 0.16
No Eligible: 84.32% — scheduler finds no eligible warp 84% of cycles, poor issue slot utilization. Each scheduler checks its pool of warps every cycle, but for this workload, 84.32% of cycles result in "No Eligible" warps to issue.
One or More Eligible: 15.68%
Issue Slot Utilization: Est. Speedup 17.81%
  
This is severely under-occupied. The scheduler only has 1 active warp out of a possible 12, and of that 1, it can only issue 16% of the time. The other 84% of cycles the scheduler is completely idle — it has nothing to issue. This directly explains the 1.33ms duration. With only 1 active warp per scheduler there is virtually no latency hiding whatsoever.
</pre>
**Warp State Statistics**
<img width="1826" height="1024" alt="Warp_State_Statistics1" src="https://github.com/user-attachments/assets/c88085f7-0734-40c2-908f-689e5072e0ab" />

<img width="1826" height="1024" alt="Warp_State_Statistics" src="https://github.com/user-attachments/assets/406f4f77-5436-4f5c-b8d1-c16d788a6822" />
<pre>
Warp Cycles Per Issued Instruction: 6.38 — each issued instruction costs 6.38 cycles on average
Avg. Active Threads Per Warp: 32 — full warp utilization, no divergence at the thread level
Avg. Not Predicated Off Threads: 22.81 — some predication reducing effective threads
Stall Wait: dominant (~4.5+ cycles per instruction, 70.5% of total stall cycles)
Stall Long Scoreboard: significant
Stall MIO Throttle: present
Stall Short Scoreboard: minor
Thread Divergence Est. Speedup: 23.61% — notable, predication is killing ~9 threads per warp

Two key findings here. First, Stall Wait dominates at 70.5% — this is a fixed-latency execution dependency
stall, meaning instructions are waiting on prior instructions to complete before they can issue. 
In a highly optimized kernel this stall is expected and hard to eliminate — the only cure is more warps to 
hide it with. Second, Thread Divergence at 23.61% estimated speedup is significant — 32 active threads but 
only 22.81 not predicated off means roughly 9 threads per warp are being masked out by predication. 
This is likely your boundary condition handling at tile edges.
</pre>

<img width="1826" height="1024" alt="Occupancy1" src="https://github.com/user-attachments/assets/885047bc-ee7d-48f5-9709-a7df38e05010" />
<pre>
Shared memory is the hard occupancy ceiling — Block Limit Shared Mem: 1. Only 1 block can fit per SM due to
shared memory usage. With (128,1,1) = 4 warps per block, you get exactly 4 active warps per SM, which at
SM120's 48 warps/SM maximum gives 4/48 = 8.33% occupancy. This matches achieved exactly.
</pre>

<img width="1826" height="1024" alt="Occupancy2" src="https://github.com/user-attachments/assets/36d7a927-858e-4546-a75b-bb9fa7a86f08" />
<pre>
completely flat at ~8% regardless of register count, registers are NOT the limiter here despite being
168/thread
</pre>
<img width="1826" height="1024" alt="Occupancy3" src="https://github.com/user-attachments/assets/503f3a95-559f-429a-a9ab-afebf3647604" />
<pre>
occupancy rises with block size up to ~384 threads then hard-drops at 416, the shared memory wall hits
</pre>
<img width="1826" height="1024" alt="Occupancy4" src="https://github.com/user-attachments/assets/1c7762fd-df9e-479a-8537-b8f2ef961d43" />
<pre>
sharp step-down at ~31KB then again at ~51KB — this is the SM120 shared memory partitioning at work.
</pre>
<img width="1826" height="1024" alt="Occupancy" src="https://github.com/user-attachments/assets/ec0116a6-0909-4650-9b7c-6668bfff7733" />
<pre>
flat until 240 barriers then drops — your barrier count is within limits
</pre>






