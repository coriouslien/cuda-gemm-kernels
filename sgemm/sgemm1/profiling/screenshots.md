launch config (40,40,1)×(128,1,1), duration 1.33ms.

<img width="1826" height="1024" alt="GPU_Speed_Of_Light_Throughput" src="https://github.com/user-attachments/assets/bfaaba08-7462-411b-9214-cc057348f4d2" />
<pre>
NCU flags High Throughput — over 80% compute utilization. The bottleneck is clearly the SM compute pipe, not memory bandwidth. To go further you need to shift work away from the most saturated unit.
Compute (SM): 82.19% — very high, the kernel is compute-bound
Memory: 63.48% — substantial but secondary to compute
L1/TEX: 43.55%, L2: 63.48% — moderate cache utilization
DRAM: 11.11% — very low, meaning data is being served mostly from L2, not DRAM
</pre>
<img width="1826" height="1024" alt="Memory_workload_Analysis" src="https://github.com/user-attachments/assets/cfdac2c1-37a3-47f7-9d5c-a261b5467b77" />

<img width="1826" height="1024" alt="Scheduler_Statistics1" src="https://github.com/user-attachments/assets/05a3e3c5-68f8-498e-b31d-3c03fde9287c" />

<img width="1826" height="1024" alt="Scheduler_statistics2" src="https://github.com/user-attachments/assets/a2772075-2cda-4776-862c-809396dbd3db" />

<img width="1826" height="1024" alt="Warp_State_Statistics1" src="https://github.com/user-attachments/assets/c88085f7-0734-40c2-908f-689e5072e0ab" />

<img width="1826" height="1024" alt="Warp_State_Statistics" src="https://github.com/user-attachments/assets/406f4f77-5436-4f5c-b8d1-c16d788a6822" />

<img width="1826" height="1024" alt="Occupancy1" src="https://github.com/user-attachments/assets/885047bc-ee7d-48f5-9709-a7df38e05010" />

<img width="1826" height="1024" alt="Occupancy2" src="https://github.com/user-attachments/assets/36d7a927-858e-4546-a75b-bb9fa7a86f08" />

<img width="1826" height="1024" alt="Occupancy3" src="https://github.com/user-attachments/assets/503f3a95-559f-429a-a9ab-afebf3647604" />

<img width="1826" height="1024" alt="Occupancy4" src="https://github.com/user-attachments/assets/1c7762fd-df9e-479a-8537-b8f2ef961d43" />

<img width="1826" height="1024" alt="Occupancy" src="https://github.com/user-attachments/assets/ec0116a6-0909-4650-9b7c-6668bfff7733" />






