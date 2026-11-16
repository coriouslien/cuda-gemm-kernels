<pre>
Detailed Nsight Compute analysis — Hopper wgmma_sm90.cu
The first 20 screenshots

We can reconstruct how the GEMM uses the H100, identify where execution stalls, and form specific hypotheses about 
why the kernel does not achieve peak Tensor Core performance.

The preliminary conclusion is:
The kernel performs FP16 WGMMA correctly, but its performance appears limited partly by memory-access latency and 
insufficient warp-level latency hiding—not by saturated HBM bandwidth.

There is also evidence of inefficient global stores. We need to investigate those separately rather than assume they 
explain all the lost performance.

Three things throughout this analysis:
- Measured: What Nsight Compute actually reports.
- Derived: What we can calculate from those measurements.
- Hypothesis: A possible bottleneck that requires further evidence.
Step 1. Understand precisely what we profiled
The screenshots show the following launch.
Property	             Value
GPU                   NVIDIA H100 80GB HBM3
Kernel                gemm_device
Grid dimensions       (40, 40, 1)
Block dimensions      (128, 1, 1)
Kernel duration       651.87 μs
Elapsed cycles        951,535
SM frequency          1.45 GHz

Let's translate this into CUDA execution terminology.
1.1 Number of CTAs
The  grid contains:
40 x 40 x 1= 1600  CTAs

Each CTA has 128 CUDA threads:
128/32= 4  warps per CTA

Therefore, the launch comprises:
1600 x 4= 6400 CTA-level warps
Those are the warps launched over the entire kernel, not the number resident simultaneously on the GPU.
Because this tutorial uses one 128-thread warpgroup per CTA, each CTA has exactly one warpgroup.
 
1.2 Verify the GEMM dimensions
NVIDIA's wgmma_sm90.cu defaults are:
int m = 5120;
int n = 5120;
int k = 4096;

Its default CTA tile dimensions are:
auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int<64>{};

These agree with the measured (40,40,1) grid:
5120/128=40

Thus, each CTA calculates one 128 x 128  output tile, accumulating contributions from 64 K-tiles. 
The source uses three shared-memory pipeline stages. 

1.3 Calculate the total floating-point operations
For GEMM:
C=A x B

Using the standard convention of two FLOPs per multiply-add:
FLOPs=2MNK

Substituting your dimensions:
=2(5120)(5120)(4096)
=214,748,364,800 FLOPs

That is approximately 214.75 GFLOPs for one GEMM execution.
Step 2. Calculate your measured performance
Your first screenshot reports:
t=651.87 mu s

Converting to seconds:
t=0.00065187 s

Now calculate throughput:
P=FLOPs/t
P=214,748,364,800/0.00065187
P approx 329.43  TFLOP/s

Nsight Compute throughput 329.43 TFLOP/s
Profiled execution time 651.87 microseconds
The standalone program reported:
CUTE_GEMM: [378214.7] GFlop/s (0.5678) ms
That represents approximately 378.21 TFLOP/s.
Measurement	               Time	             Throughput
Standalone benchmark	      567.8 μs	         378.21 TFLOP/s
Nsight Compute capture	    651.87 μs	        329.43 TFLOP/s
 
The profiled launch took approximately 14.8% longer than the earlier standalone average. This does not necessarily 
mean the profiling tool slowed the kernel itself by that amount. The captures can differ in cache conditions, 
clock frequency, launch selection, and profiling/replay behavior.

For performance optimization, use the unprofiled benchmark to compare actual runtime, and Nsight Compute to 
understand hardware behavior. NVIDIA documents that collecting counters may require multiple kernel 
replay passes.

1. GPU Speed of Light Throughput 
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/98974c2d-5772-4ded-965e-0236c73e5a56" />
2. GPU Speed of Light Throughput
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/8c770961-e5d1-48e3-acbe-b89fca78a7dd" />
3. GPU Speed of Light Throughput
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/f0f68aa2-a4cd-49f2-aad9-131f2750cc68" />
4. GPU Speed of Light Throughput
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/f3a321dc-7f3d-4388-82ae-5eba4816c98c" />

Metric	                             Value
Compute (SM) Throughput             42.29%
Memory Throughput                   72.77%
L1/TEX Cache Throughput             82.82%
L2 Cache Throughput                 48.23%
DRAM Throughput                     16.41%

These numbers are extremely important.
3.1 What does Compute (SM) Throughput = 42.29% mean?
1). What does 42.29% mean?
Nsight Compute reports:
Compute (SM) Throughput [%] = 42.29
Nsight Compute compares achieved utilization against the relevant peak hardware throughput.
A value of 42.29% means the reported compute-throughput metric reaches approximately 42% of its reference maximum.
It does not mean that only 42.29% of the SMs are running.
This measures how close the busiest relevant SM compute resource is to its peak sustained throughput, 
averaged over the kernel's execution.
It does not mean only 42.29% of the GPU's SMs are active. NVIDIA defines this as an aggregate throughput metric derived
from underlying hardware counters. 
The FP16 WGMMA operation counter also reports 42.29% of its dense HGMMA peak, which confirms that Tensor Core 
throughput is the relevant performance measure for this kernel.
 
2). Is 42.29% good?
As a rough learning-oriented guide—not an official NVIDIA grading scale:
SM throughput       General interpretation
Below 20%           Low; investigate underutilization
20–40%              Moderate
40–60%              Reasonable, with optimization potential
60–80%              Strong utilization
Above 80%           Very high utilization

These ranges aren't universal. A memory-bound kernel can be extremely well optimized while having low SM throughput.
For the compute-intensive WGMMA GEMM, however, higher Tensor Core throughput is desirable.
 
3). Why is your throughput not higher?
The earlier Nsight Compute measurements provide useful evidence.
Metric                      the result	    What it tells us
Compute (SM) Throughput	    42.29%	        Considerable compute headroom
Theoretical Occupancy	      12.50%	        Few resident warps
Eligible Warps/Scheduler	   0.15	          Very few ready instructions
No Eligible Warp	          85.12%	         Schedulers frequently cannot issue
L1/TEX Throughput	         82.82%	         High internal memory activity
DRAM Throughput	           16.41%	         HBM bandwidth is not saturated
 
The important relationship is:
The two questions are different:
- How many SMs are doing work?
- How close are the compute pipelines to their maximum throughput?
The reported value concerns throughput, not the percentage of SMs assigned work.

42.29% Compute (SM) Throughput is a reasonable result for your first Hopper WGMMA experiment, but it is not 
particularly high for a highly optimized Tensor Core GEMM. There is substantial room for improvement.
The important relationship is:
96 KiB shared memory per CTA
           |
           V
Only 2 resident CTAs per SM
           |
           V
12.5% theoretical occupancy
           |
           V
Limited ability to hide memory latency
 
Tensor Cores may not receive work continuously
This is a plausible explanation supported by the occupancy and stall metrics, although the exact limiting instruction 
still needs source-level investigation.
4). What throughput should you aim for?
The captured kernel achieved:
Current performance 329.4           Dense HGMMA peak utilization 42.29%
 TFLOP/s                            of measured reference peak

Instead of immediately targeting 80% or 90%, trying to improve the kernel to 50–60% of the dense HGMMA peak 
while maintaining correctness.
That is an experimental goal, not a prediction that the tutorial can necessarily reach it.
We could begin by comparing two versus three pipeline stages, examining long-scoreboard stalls, and investigating the 
output-store access pattern. Measure execution time as well as throughput, because optimizing one metric does not 
guarantee a faster kernel.
 
5). One important lesson
High occupancy and high SM throughput are not the same thing.
A kernel can have 12.5% occupancy yet achieve substantial Tensor Core throughput because each WGMMA instruction performs 
a large amount of arithmetic asynchronously.

The task as a GPU performance engineer is not simply to maximize occupancy. It is to keep the hardware performing 
useful work with as little waiting as possible.
The conclusion: the 42.29% is a credible starting point, not a poor result. The interesting part is that the Nsight 
report already provides clues about why it is 42.29% and which experiments might improve it.

3.2 What does Memory Throughput = 72.77% mean?
Memory Throughput summarizes activity across important memory-system resources.
We should not interpret it as:
The H100 is using 72.77% of its maximum HBM bandwidth.
The actual DRAM throughput is only 16.41%.
This is the first important distinction:
The memory system is significantly utilized, but external HBM bandwidth is not saturated.

3.3 Compare the memory levels
L1/TEX     82.82%
L2 Cache   48.23%
DRAM       16.41%
Reported throughput relative to each unit's respective peak. These are not percentages of one common bandwidth.
Notice the ordering:
82.82% > 48.23% > 16.41%
This suggests that internal memory activity is more significant than external memory bandwidth pressure.
It does not yet prove that L1 is the final bottleneck. For that, we need the Memory Workload Analysis and 
scheduler-stall information.
3.4 Interpret NVIDIA's High Memory Throughput warning
Nsight Compute states that memory is more heavily utilized than compute and recommends examining memory 
access patterns.
This is a diagnostic clue.
The investigation should be:
1. Is DRAM bandwidth saturated? No.
2. Is the L1/TEX subsystem heavily utilized? Yes.
3. Are there inefficient memory accesses? The later screenshots suggest yes, especially global stores.
4. Are warps waiting for memory operations? Yes, strongly.
The fourth observation is especially important because it connects memory behavior to lost compute performance.

Step 4. Understand the Tensor Core Roofline
Screenshots 2–4 — Tensor Core Operations Roofline and operation tables
We selected the correct chart: Roofline Tensor Core.
4.1 The mathematics behind the roofline
The fundamental roofline equation is:

P less or equal min(P_peak,B_peak x I)
Where:
Symbol              	Meaning
P	                   Achieved computational throughput
P_peak	              Theoretical peak compute throughput
B_peak	              Peak memory bandwidth
I	                   Arithmetic intensity, in operations per byte
 
The chart's vertical axis is computational performance, while the horizontal axis is arithmetic intensity.
The diagonal lines represent memory-bandwidth limitations. The horizontal lines represent compute-throughput ceilings.
The Tensor Core chart includes different memory-hierarchy levels, so multiple points or roofs can describe the 
same kernel using different measured traffic.

4.2 Why are there three colored points?
The screenshot shows three measured points at approximately the same vertical position but different arithmetic 
intensities.
The hierarchical roofline analyzes computation relative to different memory traffic levels, principally L1, L2, and DRAM.
The calculation is conceptually:
I_level= Tensor Core operations / Bytes transferred at that memory level
For example:
I_L1 = Operations / L1 traffic 
I_DRAM = Operations / DRAM traffic
 
The kernel's compute throughput is the same in these representations, but the denominator changes.
That is why the points share essentially the same height while appearing at different horizontal locations.
Important: These are different views of one kernel, not three separate kernels.

4.3 Why is the measured point below the roofline?
The measured Tensor Core rate is approximately:
329.43 TFLOP/s
The important question is whether the distance to a particular ceiling represents unused compute throughput, 
memory bandwidth limitation, or instruction and dependency latency.

The other counters show:
- Moderate Tensor Core pipeline utilization.
- Low issue-slot utilization.
- Very few eligible warps.
- Large long-scoreboard stalls.

Together, these suggest that Tensor Cores are not being supplied with work continuously enough to approach their 
dense-operation ceiling.
This is more informative than simply saying the roofline point is below the maximum.
 
4.4 Understand the Op:hgmma Src:fp16 Sparsity:off row
The third screenshot contains: (GPU Speed of Light Throughput)
Counter                                  Value
Number of operations                     214,748,364,800
Operations per cycle                     228,673.44
Operations per second, in gigaops/s      329,433.33
Peak utilization                         42.29%
Peak operations per cycle                540,672

The peak utilization mathematically: 
228,673.44 / 540,672 x 100 = 42.29%
This is excellent confirmation: The Nsight Compute throughput is based on Hopper FP16 WGMMA operations.
Notice the exact operation count:
214,748,364,800=2MNK
The profiler's Tensor Core operation count agrees with the expected GEMM workload.
That establishes consistency between the algorithm and what the hardware counter reports.
 
4.5 Why does the general Src:fp16 row show 21.15%?
The fourth screenshot reports:
Metric                       Value
FP16 operations              214,748,364,800
Achieved operations/cycle    228,673.44
Peak operations/cycle        1,081,344
Peak percentage              21.15%

1. FP16 operations: How much mathematical work was performed?
2. Achieved operations/cycle: How quickly did the GPU perform that work, measured 
in operations per clock cycle?
3. Peak operations/cycle: How quickly could the GPU theoretically perform this type of work?
4. Peak percentage: What fraction of that theoretical maximum did the kernel achieve?

1. FP16 operations: How much mathematical work was performed?
2. Achieved operations/cycle: How quickly did the GPU perform that work, measured in operations per clock cycle?
3. Peak operations/cycle: How quickly could the GPU theoretically perform this type of work?
4. Peak percentage: What fraction of that theoretical maximum did the kernel achieve?
 
2). FP16 operations = 214,748,364,800
Your GEMM computes:
C=A x B
Its matrix dimensions are:
M=5120, N=5120, K=4096
For a matrix multiplication, the standard floating-point operation count is:
FLOPs = 2MNK
Why multiply by 2?
Each multiply-accumulate contains:
- One multiplication: a x b
- One addition: c+a x b
By the usual GEMM counting convention, that's two floating-point operations.
Therefore:
2 x 5120 x 5120 x 4096
= 214,748,364,800 operations
That is approximately 214.75 billion floating-point operations.
The FP16 operations metric counts the mathematical work attributed to this FP16 Tensor 
Core operation class. It does not mean that the GPU executed 214.75 billion individual 
WGMMA instructions.
One WGMMA instruction represents many arithmetic operations.
3). Achieved operations/cycle = 228,673.44
This measures the average rate at which your H100 performs the counted operations, expressed
per clock cycle across the GPU.
The result is:
228,673.44 operations/cycle

Conceptually, that means:
During the measured kernel execution, the H100 performed the equivalent of approximately 228,673
FP16 floating-point operations per GPU clock cycle.
This is a throughput measurement, not the number of operations performed by one CUDA thread or
one SM.
Why can one cycle contain so many operations?
Because the H100 has many SMs, and each SM contains Tensor Core hardware capable of performing 
many multiply-accumulate operations in parallel.

Imagine the H100 as a factory with many workers processing calculations simultaneously.
The result measures the output of the whole factory, normalized by clock cycles.

4). Peak operations/cycle = 540,672
This is the reference maximum throughput for the dense FP16 HGMMA operation type on the H100.
Your report shows:
540,672 operations/cycle

The H100 has 132 SMs. Dividing the GPU-wide peak by the number of SMs gives:
540,672/132=4096
Thus, the peak corresponds to:
4096 FP16 operations/cycle/SM
And across all 132 SMs:
132 x 4096=540,672

This is a hardware throughput ceiling for the selected operation category, not a promise that
every kernel will reach it.
NVIDIA's profiler defines the peak using its sustained-throughput model for the relevant
operation and hardware. 

5). Peak percentage = 42.29%
Now we can calculate the most important number.
The formula is:
Peak Percentage=
((Achieved Operations/Cycle)  / (Peak Operations/Cycle) ) x 100
Substitute the measurements:
(228,673.44 / 540,672) x 100
=42.29%

The H100 dense FP16 Tensor Core utilization
42.29%           57.71% headroom
228,673 operations/cycle achieved540,672 peak

That means the WGMMA kernel achieved approximately 42.29% of the reference dense FP16 HGMMA
throughput ceiling.

The remaining 57.71% is theoretical headroom, not guaranteed performance improvement. Real 
kernels have memory operations, synchronization, dependencies, and other overhead.
 
6). Why does another FP16 row show only 21.15%?
We may have noticed another row in the screenshot:
Metric	Dense HGMMA FP16 row	General FP16 row
FP16 operations	214,748,364,800	214,748,364,800
Achieved operations/cycle	228,673.44	228,673.44
Peak operations/cycle	540,672	1,081,344
Peak percentage	42.29%	21.15%

Why the difference?
The general FP16 category uses a higher reference peak that includes the capability for sparse Tensor Core 
operations, which can achieve twice the arithmetic rate of dense operations.

The kernel performs dense FP16 WGMMA:
Op: hgmma
Src: fp16
Sparsity: off

Therefore, for evaluating the current kernel, 42.29% is the more meaningful comparison. 
NVIDIA explains how sparsity affects these peak percentages. 

7). Connect these numbers to TFLOP/s
The profiler also reports approximately:
329.43 TFLOP/s
This is another way to express achieved computational throughput.
- Operations/cycle measures work per GPU clock cycle.
- Operations/second measures work per second.
- TFLOP/s expresses trillions of floating-point operations per second.
Conceptually:
Operations/second
= (Operations/cycle) x (Cycles/second)
The profiler accounts for measured clock behavior and counter normalization when producing its per-second metrics.
 
8). Check your understanding
Imagine another GPU kernel has:
- Achieved operations/cycle = 300,000
- Peak operations/cycle = 600,000
 
What is its peak percentage?
(300,000/600,000) x 100=50%
FP16 operations tells us how much mathematical work was done. Achieved operations/cycle tells us
how quickly the GPU did it. Peak operations/cycle tells us the reference hardware capability. 
Peak percentage compares the two rates.

For the wgmma_sm90 experiment, that comparison is 42.29%, and the next optimization task is to 
understand what prevents the Tensor Cores from being supplied with work more continuously.
 
Compare its peak with the non-sparse HGMMA peak:
1,081,344=2 x 540,672
The general FP16 maximum used for this row is twice the dense HGMMA FP16 peak, reflecting
the higher supported  sparse Tensor Core operation rate.
The kernel uses:
Op:hgmma Src:fp16 Sparsity:off
 
4.6 Why do FP32 and FP64 show 0%?
The roofline analysis message says the workload achieved 0% of the FP32 and FP64 peaks.
That is expected for an FP16 Tensor Core GEMM.
The actual matrix multiplication is performed by HGMMA, not ordinary FP32 or FP64 arithmetic pipelines.
This is not a performance problem.

 5. PM Sampling
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/4f7ef757-21d5-40ab-82ae-1f90e9a0c949" />
Step 5. Analyze PM Sampling
Screenshot 5 — PM Sampling timeline
The report shows:
Measurement	Value
Maximum sampling interval	1.50 μs
Maximum buffer size	48 MB
Pass groups	2
Dropped samples	0

PM Sampling lets you examine how performance metrics change during the kernel's lifetime, rather than using only a whole-kernel average.

Your timeline is particularly interesting.
5.1 The kernel has a sustained computation phase
The SM Active Cycles and Tensor Pipe traces remain elevated through most of the kernel.

This indicates that the kernel is not spending most of its duration waiting for GPU work to begin.

There is sustained execution.
5.2 There are periodic decreases in Tensor Pipe activity
The Tensor Pipe trace shows several dips.

A possible explanation is that the mainloop periodically waits for a data-transfer or synchronization dependency.

However, a dip alone does not identify the cause.

We would need to correlate the location of these dips with source-level stall samples or additional pipeline measurements.

The profiler collected two pass groups, so counters from different passes should not automatically be treated as perfectly synchronized to individual instructions.
5.3 Why is DRAM activity comparatively small?
The DRAM throughput trace is much lower than the Tensor Pipe trace.

This agrees with your earlier finding that DRAM throughput reaches only 16.41% of its peak.

It is further evidence that external HBM bandwidth is unlikely to be the primary limiting resource for this kernel.
5.4 What I would investigate
Rather than optimizing from the dips alone, I would first identify the top stalled SASS instructions in the Source tab.

That will give us much stronger evidence about whether the delays come from copying A/B tiles, waiting for WGMMA, or executing the epilogue.

Step 6. Analyze Compute Workload Analysis
Screenshots 6–7 — Compute Workload Analysis and pipeline utilization

This section measures the scheduling and execution resources inside the streaming multiprocessors.

Your important numbers are:
Metric	Measured value
Executed IPC Elapsed	0.52
Executed IPC Active	0.59
Issued IPC Active	0.59
SM Busy	48.13%
Issue Slots Busy	14.86%
Tensor (All), active cycles	Approximately 48.1%


 
Step 6. Analyze Compute Workload Analysis
Screenshots 6–7 — Compute Workload Analysis and pipeline utilization

This section measures the scheduling and execution resources inside the streaming multiprocessors.

Your important numbers are:
Metric	Measured value
Executed IPC Elapsed	0.52
Executed IPC Active	0.59
Issued IPC Active	0.59
SM Busy	48.13%
Issue Slots Busy	14.86%
Tensor (All), active cycles	Approximately 48.1%

6. Compute Workload Analysis
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/001ab819-58ad-4e76-b451-a5cbf1d03ef8" />
7. Compute Workload Analysis
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/2b9ede8b-2aba-4769-9163-348fd6f80cf0" />
 
Step 6. Analyze Compute Workload Analysis
Screenshots 6–7 — Compute Workload Analysis and pipeline utilization
This section measures the scheduling and execution resources inside the streaming multiprocessors.
The important numbers are:
Metric	Measured value
Executed IPC Elapsed	0.52
Executed IPC Active	0.59
Issued IPC Active	0.59
SM Busy	48.13%
Issue Slots Busy	14.86%
Tensor (All), active cycles	Approximately 48.1%

6.1 Understand IPC
IPC means instructions per cycle.
The reported executed IPC values summarize instruction execution at the relevant profiler aggregation level. 
They do not represent GEMM FLOPs per cycle.
This distinction matters because one WGMMA instruction performs many arithmetic operations.
A relatively small instruction count can still generate hundreds of TFLOP/s.
6.2 Understand SM Busy = 48.13%
SM Busy describes how busy the SM execution machinery is during the sampled period.
It does not mean that only 48.13% of the H100's SMs participate in the kernel.
It is a utilization measure, not a count of participating SMs.
6.3 Understand Issue Slots Busy = 14.86%
This is one of the most significant findings in the report.
A warp scheduler has opportunities to issue instructions. If no resident warp is eligible, the corresponding 
opportunity goes unused.
The issue-slot utilization is:
14.86%
This suggests substantial periods in which schedulers cannot issue an instruction.
The Scheduler Statistics section gives direct confirmation.
 
6.4 Why is Tensor Pipe utilization only approximately 48%?
The charts identify the Tensor FP pipeline as the dominant compute pipeline.
That is correct for a WGMMA GEMM.
But it is not near saturation.
The report says the Tensor pipeline is well utilized but should not be the bottleneck.
This suggests that something other than pure Tensor Core execution capacity is preventing higher throughput.
A likely contributor is the rate at which data and dependencies become ready for the next WGMMA work.
 
6.5 Why does the Tensor (Warp Group) instruction-issue bar look small?
The left chart measures pipeline active-cycle utilization; the right chart measures executed instruction 
throughput relative to peak instruction-issue rates.
These are different quantities.
WGMMA is an asynchronous, warpgroup-level operation. One issued instruction can initiate a large amount 
of Tensor Core arithmetic.
Therefore, the Tensor Warp Group instruction-issue bar should not be equated directly with Tensor Core arithmetic utilization.
Do not conclude that the Tensor Core is unused simply because the WGMMA instruction-issue bar is short.

8. Memory Workload Analysis 
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/948b4c7e-133f-4df5-9806-58c76b9f479b" />
9. Memory Workload Analysis
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/6adac94a-0bbb-41c8-b51d-17dca6c43330" />
10. Memory Workload Analysis
 <img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/f28b181e-afd2-4380-a4d7-58e8653f1b5e" />
 
Step 7. Analyze Memory Workload Analysis
Screenshot 8 — Memory Workload Analysis summary

This is one of the most useful sections of the report.
The measurements are:
Metric	Value
Memory Throughput	550.17 GB/s
Mem Busy	72.77%
Max Bandwidth	47.74%
Mem Pipes Busy	26.44%
L1/TEX Hit Rate	9.18%
L2 Hit Rate	79.75%
L2 Compression Success Rate	0%

Nsight Compute distinguishes utilization of memory units, communication bandwidth, and memory-instruction 
pipelines. All three can limit memory performance. 

7.1 What does Memory Throughput 550.17 GB/s mean?
This is the reported device-memory throughput.
To estimate total HBM traffic over the measured kernel:
550.17 x 10^9 x 0.00065187
approximat 358.64 x 10^6  bytes
Therefore, the capture corresponds to approximately:
358.6 MB of  device-memory traffic
This is a derived value using the reported average bandwidth and kernel duration.
 
7.2 Why is Mem Busy 72.77% but Max Bandwidth only 47.74%?
They represent different types of resource pressure.
- Mem Busy: Activity/utilization of memory-system hardware.
- Max Bandwidth: Utilization relative to the maximum communication bandwidth among relevant memory paths.
- Mem Pipes Busy: Utilization of memory instruction-processing pipelines.
The values show that memory-system hardware can be busy even when the highest possible data-transfer 
rate is not being reached.
For example, inefficient transactions or latency dependencies can occupy resources without transferring 
data at maximum efficiency.
 
7.3 L1/TEX Hit Rate = 9.18%
A conventional interpretation is that most relevant L1/TEX requests do not produce L1 cache hits.
However, the kernel uses asynchronous global-to-shared-memory copies.
For those operations, cache policies and data-movement behavior differ from ordinary scalar loads, so a 
low L1 hit rate is not automatically a design error.
We should investigate the traffic, not simply maximize the L1 hit-rate number.
 
7.4 L2 Hit Rate = 79.75%
This is much higher.
It suggests that a large fraction of requests reaching L2 can be served there rather than requiring 
external device-memory access.

That makes sense for a tiled GEMM:
- Multiple CTAs consume overlapping regions of A or B.
- Repeated computation can benefit from cached matrix tiles.
- The L2 cache may service reused data efficiently.
A high L2 hit rate helps explain why the HBM bandwidth is not saturated.
 
7.5 L2 compression success = 0%
Nsight Compute flags low compression success and estimates a potential speedup.
That is a profiler heuristic, not a guaranteed improvement.

The kernel's primary purpose is matrix multiplication. Do not redesign its numerical representation 
merely to make the compression percentage higher.
We should prioritize actual data dependencies and inefficient transactions first.

Step 8. Understand the global-store warning
Screenshots 8–10 — Global store access pattern and memory chart
The screenshot contains this warning:
L1TEX Global Store Access Pattern — Estimated Speedup: 36.38%
The explanatory text states that, on average, only 16 of the 32 bytes in a transferred sector are useful.
That corresponds to:
Sector utilization = 16 / 32
50%

8.1 What does 50% sector utilization mean?
Imagine the hardware transfers a 32-byte sector, but the memory instruction needs only 16 bytes of 
data within that sector.
The unused bytes still contribute to the transferred sector.
This is less efficient than an access pattern that uses the complete sector.
It is not equivalent to saying that every thread stores 16 bytes.
The warning describes aggregate sector efficiency.
 
8.2 Why might the output stores be inefficient?
The GEMM uses:
axpby(alpha, tCrC, beta, tCgC);
This writes the accumulator fragment back to the output tensor.
The efficiency of those stores depends on how the logical output tile is partitioned among threads and
how that partition maps to contiguous memory addresses.
For example, if neighboring threads write locations that are not adjacent in memory, one warp may touch 
more sectors than necessary.

A simplified example:
<img width="1206" height="321" alt="image" src="https://github.com/user-attachments/assets/4b099d0e-a43a-48c4-aa24-4be34b794c2a" />
These diagrams illustrate the principle, not the exact pattern of the kernel.
 
8.3 Does the 36.38% estimated speedup mean the kernel can become 36.38% faster?
No.
It is an estimated local optimization opportunity produced by Nsight Compute's diagnostic rules.
It is not proof that changing the stores will make the complete kernel 36.38% faster.
The total kernel also spends time copying A/B tiles, performing MMA, synchronizing warps and
processing dependencies.

The correct next investigation is the Source Counters section, especially the SASS STG instructions 
associated with output storage.

Step 9. Trace the memory movement through the source code
Screenshots 9–10 — L1/TEX, L2, DRAM, shared memory and TMA paths
The memory chart shows substantial activity through the L1/TEX and shared-memory paths.
This connects directly to how wgmma_sm90.cu moves A and B.
 
9.1 The program uses cp.async, not TMA
The default tutorial uses copy atoms similar to:
TiledCopy copyA = make_tiled_copy(
    Copy_Atom<
        SM80_CP_ASYNC_CACHEALWAYS<uint128_t>,
        TA>{},
    ...);

The generated SASS includes LDGSTS, which represents asynchronous global-to-shared-memory copying.
The Instruction Statistics chart also contains substantial LDGSTS activity.

This is the expected data path:
Global memory: A and B
     |
     V
cp.async / LDGSTS
     |
     V
Shared memory: sA and sB
     |
     V
WGMMA / HGMMA
     |
     V
Accumulator: tCrC
     |
     V
Epilogue / global stores
     |
     V
Global memory: C

NVIDIA's tutorial explicitly constructs asynchronous copy atoms, then uses WGMMA to consume the shared-memory tiles. 

9.2 Why is TMA activity zero?
The memory chart shows zero TMA instructions.
That is expected. 
This program is wgmma_sm90.cu, not wgmma_tma_sm90.cu.
Both use Hopper WGMMA, but the latter uses the Hopper Tensor Memory Accelerator to move tiles.
The current program uses the earlier cp.async mechanism to stage data in shared memory.
This distinction will become extremely valuable when we compare the two tutorials.
     
9.3 Why is shared-memory traffic large?
WGMMA reads input tiles from shared memory, and cp.async writes new tiles into shared memory.
Both operations generate substantial internal traffic.
We should not confuse shared-memory bytes with HBM bytes.
The same tile may be consumed several times from on-chip memory while generating far more arithmetic 
operations than its initial global-memory transfer required.
That reuse is the central idea behind tiled GEMM.
     
9.4 What do we still need to check?
The memory chart is a high-level representation.
To identify whether shared memory itself is inefficient, we need to examine the detailed shared-memory 
table for:
- Wavefronts per request.
- Bank conflicts.
- Actual shared-memory transaction counts.

Memory Tables Analysis:
1)
<img width="1818" height="1095" alt="image" src="https://github.com/user-attachments/assets/831f52e2-ca60-47c1-ab61-10582209b690" />
2)
<img width="1818" height="1095" alt="image" src="https://github.com/user-attachments/assets/81b6a809-f72b-46c2-b1cc-9fa1b9385a37" />
3)
<img width="1818" height="1095" alt="image" src="https://github.com/user-attachments/assets/1bb01f68-d2b9-41c5-ae16-f9417aa2a9b7" />
4)
<img width="1818" height="1095" alt="image" src="https://github.com/user-attachments/assets/aeaa2e4e-d05d-4741-a6b5-d4cbb5ad0b78" />
5)
<img width="1818" height="1095" alt="image" src="https://github.com/user-attachments/assets/d99d35cc-5f30-46f7-9c78-3bbcb098e16f" />
Nov 15
The wgmma_sm90.cu kernel moves matrix data through shared memory, L1/TEX, L2, and HBM.
The most important finding is that Nsight Compute reports 4,810,514 shared-memory bank-conflict events in the Shared Store From Global Load path. However, that number alone does not prove that bank conflicts are a major performance bottleneck.

In fact, another measurement in your screenshot suggests we should investigate the counter carefully before modifying your CuTe shared-memory layout.
1. Start with your Shared Memory table
The first screenshot shows:
Access type                   Instructions Requests   Wavefronts   % Peak  Bank conflicts
Shared Load	                  0	           0	         0	           0%	     0
Shared Load Matrix	           0	           0	         0	           0%	     0
Shared Store	                 0	           0	         0	           0%	     0
Shared Store Matrix           0	           0	         0	           0%	     0
Shared Store From Global Load	6,758,400	   6,758,400	 27,033,600	  5.45%	  4,810,514
Shared Atomic	                0	           0	         0	           0%	     0
Other	                        —	           —	         85,524,949	  68.99%	 0
Total	                        6,758,400	   6,758,400	 112,558,549	 74.44%	 4,810,514  

NVIDIA defines the columns as counts of warp-level instructions, requests, memory-service 
wavefronts, throughput utilization, and reported bank-conflict events.

2. Why are ordinary Shared Load and Shared Store both zero?
You might initially think:
My kernel uses shared memory extensively. Why does Nsight Compute say zero shared loads and stores?

The answer is that your kernel primarily uses two specialized mechanisms:
- cp.async to move input data from global memory directly into shared memory.
- WGMMA to consume input operands from shared memory.
These are not necessarily counted as ordinary LDS and STS instructions.
Data movement in your kernel
Global memory (A and B)
      |
      V
cp.async -> LDGSTS
Shared Store From Global Load
      |
      V
Shared memory (sA and sB)
      |
      V
WGMMA -> HGMMA
Asynchronous matrix operand consumption
      |
      V
Accumulator registers (tCrC)

Thus, zero ordinary Shared Load instructions does not mean the Tensor Cores never read shared memory.

For an SS-mode WGMMA, operand data is consumed from shared memory through descriptors.
This is the first important lesson from your table.

3. Understand Instructions = 6,758,400
Your table reports:
Shared Store From Global Load
Instructions = 6,758,400


This counts executed SASS instructions at warp granularity.

In your kernel, the relevant instruction is LDGSTS, generated from the cp.async copying mechanism.

One warp-level instruction can transfer data on behalf of many threads.

Therefore:

6,758,400 instructions does not mean 6,758,400 bytes or 6,758,400 individual thread loads.

It is a count of executed warp-level copy instructions.

We can calculate how many such instructions each CTA executes on average.

Your launch has 1,600 CTAs:
\[
\frac{6,758,400}{1600}
=\boxed{4224\text{ warp-level copy instructions/CTA}}
\]

That is the average for this launch. The total includes the copy work performed throughout the GEMM mainloop.
4. Understand Requests = 6,758,400
Your table shows the same number of requests:
\[
\text{Instructions}=6,758,400
\]

\[
\text{Requests}=6,758,400
\]

For this memory path, each executed warp-level copy instruction accounts for one shared-memory request.

This is expected.

An instruction and a request are not the same thing as a wavefront.

A request may require multiple wavefronts to service all the data involved.

That distinction is particularly important for your kernel because each lane may transfer 16 bytes.

5. Understand Wavefronts = 27,033,600
Your screenshot shows:
\[
\text{Wavefronts}=27,033,600
\]

Let's calculate the average wavefronts per request:
\[
\frac{27,033,600}{6,758,400}
=\boxed{4.0}
\]

This is an extremely interesting result.
Why might one request need four wavefronts?
Your CuTe copy atom uses:
SM80_CP_ASYNC_CACHEALWAYS<uint128_t>


A uint128_t represents 128 bits, or 16 bytes.

Assuming a full warp participates:
\[
32\text{ threads}\times16\text{ bytes}
=512\text{ bytes}
\]

The shared-memory service mechanism processes requests in bounded transaction-sized portions.

For a straightforward 512-byte warp request, four 128-byte wavefronts may be needed:
\[
\frac{512}{128}=\boxed{4}
\]

Illustration: One 512-byte warp copy
Wave 1   Wave 2   Wave 3   Wave 4
128 B    128 B    128 B    128 B
Four wavefronts can be necessary for a wide access even when the shared-memory bank mapping is conflict-free.
NVIDIA confirms that wide shared-memory accesses are split into multiple wavefronts without necessarily causing bank conflicts. 
NVIDIA Developer Forums

What this tells us
Your measured value is:
\[
\boxed{4\text{ wavefronts/request}}
\]

That is consistent with the expected four wavefronts for a 16-byte-per-lane copy.

Therefore, four wavefronts per request is not, by itself, evidence of bank conflicts.

This brings us to the most subtle part of your report.

6. Analyze Bank Conflicts = 4,810,514
Your screenshot reports:
\[
\boxed{4,810,514\text{ bank-conflict events}}
\]

All of those events appear under:
Shared Store From Global Load


The ordinary shared-load and shared-store rows show zero.
6.1 What is a shared-memory bank conflict?
Hopper shared memory is organized into 32 banks.

For ordinary 32-bit accesses, successive 32-bit words map to successive banks.

A simplified mapping is:
\[
\text{Bank}=
\left(\frac{\text{Byte Address}}{4}\right)\bmod32
\]

Suppose two threads access different 32-bit words mapping to the same bank in the same serviced access.

The bank may be unable to satisfy both simultaneously, requiring serialization.

That is a bank conflict.

NVIDIA documents this bank organization and the resulting serialization. 
NsightCompute

6.2 Calculate reported conflicts per request
Your measurements are:
\[
\text{Bank Conflicts}=4,810,514
\]

\[
\text{Requests}=6,758,400
\]

Therefore:
\[
\frac{4,810,514}{6,758,400}
\approx\boxed{0.712}
\]

This is approximately 0.71 reported conflict-counter events per request.

But be careful: it does not mean 71.2% of requests experienced a bank conflict. Counter events are not necessarily a binary one-conflict-or-no-conflict count per request.

It also does not directly measure a 71.2% performance penalty.
6.3 Why the result deserves closer investigation
Here is the interesting combination:
Metric                            Result
Copy requests                     6,758,400
Copy wavefronts                   27,033,600
Wavefronts per request	Exactly    4
Reported bank conflicts           4,810,514

If every reported bank conflict represented an additional serialized wavefront beyond the four expected from the wide copy, we might expect the measured wavefront count to exceed the four-per-request baseline.

But it does not.

That means we should not interpret the bank-conflict counter as a direct count of additional wavefronts.
6.4 An important NVIDIA profiler caveat
NVIDIA engineers have documented that some shared-memory conflict counters also count arbitration or stalled-cycle events rather than exclusively counting genuine address-based bank conflicts.

An NVIDIA developer-tools engineer recommends comparing actual shared-memory wavefronts with their ideal counterparts to distinguish genuine excess serialization from counter artifacts. 
NVIDIA Developer Forums


This distinction matters directly to your experiment.

We have a nonzero reported conflict counter, but the copy-path wavefront count is exactly four per request—the expected baseline for this access width.

That is not sufficient to prove the 4.81 million counter events represent 4.81 million performance-damaging bank conflicts.

My present conclusion: There may be conflicts or arbitration pressure in the LDGSTS destination path, but we have not established that ordinary shared-memory bank conflicts are a major bottleneck.

7. What is the mysterious “Other” row?
Your Shared Memory table contains:
Metric	                Other
Instructions	           —
Requests	               —
Wavefronts	             85,524,949
% Peak	                 68.99%
Bank Conflicts	         0
This row accounts for most of the measured shared-memory wavefront activity.

Calculate its proportion:
\[
\frac{85,524,949}{112,558,549}\times100
\approx\boxed{76.0\%}
\]

So approximately 76% of the reported shared-memory wavefronts are classified as Other.

That is important.

Your kernel uses WGMMA, which consumes substantial shared-memory operand data through specialized hardware paths.

WGMMA-related activity is a plausible contributor to this category, but the table does not break down Other sufficiently to attribute all 85.5 million wavefronts to HGMMA.

Likewise, the zero in the Bank Conflicts column does not prove that every specialized access represented by Other is conflict-free.

The useful conclusion is:

Most of your reported shared-memory traffic is not represented by ordinary shared-load or shared-store instructions.

The Other row's 68.99% peak utilization also shows why internal shared-memory activity deserves attention.

8. Interpret Total Wavefronts and % Peak
Your table reports:
\[
\text{Total Wavefronts}=112,558,549
\]

And:
\[
\text{Total Peak Utilization}=74.44\%
\]

The total wavefronts are the sum of:
\[
27,033,600+85,524,949
=\boxed{112,558,549}
\]

The 74.44% is a throughput utilization metric relative to the relevant shared-memory service capacity.

It is not the percentage of shared memory capacity allocated.

Your kernel's 96 KiB per CTA is a capacity measurement.

The 74.44% in this table is a throughput measurement.

Those are different concepts.

Your shared-memory capacity restricts how many CTAs can reside on an SM, while your shared-memory throughput reflects how actively the memory subsystem services requests.

Both can affect performance, but in different ways.

9. Analyze the L1/TEX Cache table
Screenshots 1–3 — L1/TEX Cache
Your second screenshot shows:
Access type                 Instructions Requests	 Wavefronts	 Sectors	    Bytes
Global Load To Shared Store	6,758,400	   6,758,400	11,696,305	 108,134,400	3,460,300,800
Global Store	               819,200	     819,200	  2,001,676	  3,291,461	  105,326,752
Total	                      7,577,600	   7,577,600	13,697,981	 111,425,861	3,565,627,552

9.1 Why are there so many sectors in the copy path?
Your global-to-shared copy path transfers:
\[
108,134,400\text{ sectors}
\]

A sector is 32 bytes.

Therefore:
\[
108,134,400\times32
=\boxed{3,460,300,800\text{ bytes}}
\]

The average sectors per copy request are:
\[
\frac{108,134,400}{6,758,400}
=\boxed{16}
\]

Therefore, average bytes per request:
\[
16\times32=\boxed{512\text{ bytes}}
\]

That matches our earlier calculation:
\[
32\text{ threads}\times16\text{ bytes/thread}
=512\text{ bytes}
\]

This is a strong consistency check.

Your source uses a 16-byte asynchronous copy atom, and the observed global-to-shared requests average 512 bytes of sector traffic per warp-level request.
9.2 Why is the global-copy traffic 3.46 GB?
Your GEMM performs many K iterations.

Each CTA repeatedly loads A and B tiles into shared memory.

Across all 1,600 CTAs, those copies generate substantial traffic through the L1/TEX path.

But 3.46 GB of traffic at this level does not mean that the H100 fetched 3.46 GB from external HBM.

Some data can be served by L2 or other cache behavior.

That explains why we must distinguish traffic at each memory level.
9.3 Analyze global stores
Your table shows:
\[
819,200\text{ global-store requests}
\]

and:
\[
105,326,752\text{ sector bytes}
\]

Your output matrix contains:
\[
5120\times5120
=26,214,400\text{ FP16 elements}
\]

At two bytes per element:
\[
26,214,400\times2
=\boxed{52,428,800\text{ useful output bytes}}
\]

Compare that with L1/TEX sector traffic attributed to global stores:
\[
\frac{105,326,752}{52,428,800}
\approx\boxed{2.01}
\]

This is consistent with the earlier diagnostic warning that only about half of the transferred store-sector bytes are useful.

Your output epilogue is a real candidate for improving global-store transaction efficiency.

However, that does not establish that output stores are the dominant source of the kernel's execution time.

10. Analyze the L2 Cache table
Screenshots 3–5 — L2 Cache

Your L2 table includes these values:
Source of L2 traffic	Requests    Sectors     Hit rate
L1/TEX Load	         21,231,782  85,888,800  82.15%
L1/TEX Store         1,765,310   3,142,860   100%
L1/TEX Total         22,997,092  ~89.03M     83.06%
L2 Fabric Total      5,718,336   11,024,880  63.83%
GPU Total            28,768,382  100,299,659 80.66%

10.1 Why is the L2 hit rate high?
For traffic reaching L2 from L1/TEX, the reported load hit rate is:
\[
\boxed{82.15\%}
\]

Your tiled GEMM reuses matrix data across many CTAs.

For example, different output tiles in the same row of C need the same A rows, while different output tiles in the same column need the same B columns.

That creates opportunities for L2 reuse.

The measured hit rate is consistent with substantial cache reuse, although it does not tell us exactly which tiles were reused or which earlier launches contributed to cache contents.
10.2 Why does L2 show a different total hit rate?
Your earlier Memory Workload Analysis summary reported approximately 79.75% L2 hit rate.

The detailed table contains several hit rates because it separates different L2 request sources.

The 82.15% value applies specifically to the L1/TEX Load row.

The 80.66% GPU Total value includes other traffic classes.

Do not treat these numbers as conflicting measurements of one identical request population.
10.3 Does a high L2 hit rate mean no memory bottleneck?
No.

An L2 hit avoids an external-memory access, but it still has a latency.

A warp may have to wait for data even when that data comes from L2.

Thus, high L2 reuse can coexist with significant memory-dependency stalls.

This is relevant because your earlier Warp State Statistics showed high Long Scoreboard stalls.

11. Understand L2 Cache Eviction Policies
Your screenshot shows that most traffic uses the Normal eviction policy.

In particular, the L1/TEX Load row reports approximately:
Normal:   84,413,468 sectors
Hit rate: 82.15%


The policy categories describe cache-retention priorities associated with L2 accesses.

These are not separate physical L2 caches.

The First, Last, Normal, and related columns describe policy classifications.

For your workload, the important observation is that most L2 traffic follows the normal policy.

I would not spend time changing cache-eviction policy yet. Your current report points more strongly toward latency hiding, shared-memory resource allocation, and the copy/synchronization pipeline.

12. Analyze Device Memory — Actual HBM traffic
Screenshot 5 — Device Memory

Your screenshot provides:
Operation	Sectors      Bytes         Throughput
Load      9,756,896    312,220,672   478.96 GB/s
Store     1,450,648    46,420,736    71.21 GB/s
Total     11,207,544   358,641,408   550.17 GB/s
Let's verify the total:
\[
312,220,672+46,420,736
=358,641,408\text{ bytes}
\]

Your profiled kernel duration was 651.87 μs.

Therefore:
\[
B=\frac{358,641,408}{0.00065187}
\]

\[
\boxed{B\approx550.17\text{ GB/s}}
\]

This agrees with the profiler's reported memory throughput.

Your report also shows:
\[
\text{DRAM Throughput}=16.41\%
\]

That is further evidence that the kernel is not saturating the H100's external-memory bandwidth.
Compare memory traffic at different levels
Global-to-shared L1/TEX sector traffic    3.46 GB
Total HBM traffic                         0.359 GB

Different measurement points and traffic classes; the bars illustrate scale, not a directly calculated cache-hit rate.
This is a valuable illustration of why tiled GEMM can perform enormous amounts of arithmetic without continuously fetching all of its operand data from HBM.

13. Are shared-memory bank conflicts your primary bottleneck?
Here is how I would combine your new measurements with the earlier report.
Evidence                              Observation    Interpretation
Shared-memory bank-conflict counter   4.81 million   Requires investigation
Copy-path wavefronts/request	         Exactly 4      Consistent with 16-byte/lane accesses
Shared Memory Other utilization       68.99%         Significant specialized shared-memory activity
Total shared-memory utilization       74.44%         Substantial pressure on the shared-memory subsystem
L1/TEX throughput                     82.82%         High internal memory-system utilization
Long Scoreboard stalls               ~8.3 cycles/instruction	 Important memory-related dependency waits
Eligible warps/scheduler             0.15            Insufficient ready work
DRAM throughput                      16.41%          External bandwidth not saturated
One crucial point from NVIDIA's profiling guidance: shared-memory serialization is more closely associated with Short Scoreboard and MIO Throttle stalls, whereas Long Scoreboard concerns L1/TEX memory dependencies. NVIDIA recommends examining those specific stall metrics and memory bank throughput before treating reported conflicts as a significant limiter. 
NVIDIA Developer Forums


Your earlier report showed Long Scoreboard as the dominant stall reason. This does not directly establish that shared-memory bank conflicts cause it.

So my current diagnosis is:

The kernel has substantial shared-memory activity and a nonzero bank-conflict counter, but the evidence does not yet establish that actual bank conflicts are its main performance problem.

I would not change Swizzle<3,4,3> simply because the counter is nonzero. That swizzle is part of a carefully constructed CuTe layout compatible with the chosen WGMMA operations.

14. The next step: Verify actual versus ideal shared-memory wavefronts
This is the most useful next investigation, and it can be attempted using your existing report.

In Nsight Compute:
1. Select the Source tab.
2. Switch to the SASS view.
3. Look for the source-counter columns L1 Wavefronts Shared, L1 Wavefronts Shared Ideal, and L1 Wavefronts Shared Excessive, if available.
4. Examine the LDGSTS instructions associated with your cp.async copies.
5. Compare actual and ideal wavefronts for those instructions.
These metrics are useful because a wide memory instruction may legitimately require multiple wavefronts without conflicts. We want to identify excessive wavefronts beyond the ideal access pattern, not simply count every wavefront beyond the first. 
NVIDIA Developer Forums+1


For your wide copy, the simplified baseline is:
\[
\text{Ideal Wavefronts/Request}\approx4
\]

And your aggregate measurement is:
\[
\text{Actual Wavefronts/Request}=4
\]

That agreement is encouraging, but a source-level comparison would be stronger evidence. It would also help resolve whether the aggregate bank-conflict counter is reflecting true serialization or additional hardware arbitration events.

If the needed source metrics were not collected, the existing report may not be able to show that comparison; it would then require a future capture.
Final conclusion
You have now collected and examined the three measurements we originally needed:
Original question                     What your screenshots establish
How many shared-memory requests?      6,758,400 in the copy path
How many wavefronts?                  27,033,600 copy-path wavefronts, exactly four per request
Are bank conflicts reported?          Yes — 4,810,514 events, but their actual performance 
                                      impact is not established
The biggest lesson is:

Wavefronts per request must be interpreted relative to the width and access pattern of the memory instruction. A 16-byte-per-thread copy can require four wavefronts even when the memory layout is conflict-free.

Your report now gives us a much more concrete performance question than before: Are the reported bank-conflict events causing extra shared-memory serialization, or are they partly reflecting other arbitration activity?

That is the distinction I would investigate before attempting to optimize shared-memory layout.

 


 
Nov 15


 ??????
 
The a suffix identifies architecture-specific features that are not guaranteed to work on later GPU architectures.
The WGMMA instruction family is one of the reasons for targeting sm_90a.
We should not think of it as just a newer spelling of sm_90.
H100 — Hopper                  sm_90a compatible
RTX 5080 — Blackwell           Cannot execute SM90a WGMMA
This is precisely why the desktop RTX 5080 Blackwell SM120 can compile the executable but cannot run that 
 Hopper WGMMA kernel.
The compiled machine code is for the target architecture, not the GPU doing the compilation.
 
3.4 Why include both PTX and machine code?
The build retains the architecture-specific PTX in addition to generating the sm_90a machine code.
This can be useful for inspecting the intermediate representation and for compatible driver JIT scenarios.
However, the compute_90a PTX is not a general forward-compatible fallback for your Blackwell RTX 5080. 
Architecture-specific PTX features are restricted to compatible targets.

 3.5 --generate-line-info: Essential for your Nsight studies
The compiler command contains:
--generate-line-info
This generates device-code source-line information.

That is especially useful because we want to connect:
CUDA C++ source
       |
       V
PTX
       |
       V
SASS
       |
       V
Profiler metrics

The earlier Nsight Compute screenshots displayed source references such as wgmma_sm90.cu:235 and axpby.hpp:90.
That is consistent with having useful line-number information in the binary.
One important distinction: line information does not necessarily embed the full source text. For source-level 
browsing on a different computer, the source files may still need to be accessible or embedded during profiling.

NVIDIA recommends line information rather than disabling device optimization with -G for performance profiling. 
 
3.6 -Xptxas=-v and --resource-usage
These two options explain why your build prints detailed resource information.
-Xptxas=-v
--resource-usage
For the H100, the driver can use the native sm_90a machine code embedded in your executable.
 

 


 

 


11. Scheduler Statistics
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/fc442123-9e32-414d-ad24-78221f809cf7" />

 12. Scheduler Statistics
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/07bbe929-ac2a-4b86-8e00-443fa1be0ea5" />
Step 10. Analyze Scheduler Statistics
Screenshots 11–12 — Scheduler Statistics and Warps Per Scheduler

This is probably the most revealing part of your current report.

Your numbers:
Metric	Value
Active Warps Per Scheduler	1.99
Eligible Warps Per Scheduler	0.15
Issued Warp Per Scheduler	0.15
No Eligible	85.12%
One or More Eligible	14.88%

10.1 What is an active warp?
An active warp is resident on a scheduler. It has been allocated GPU resources and has not finished executing.

But an active warp does not necessarily have an instruction ready to issue.
10.2 What is an eligible warp?
An eligible warp has an instruction ready to execute, with the required dependencies and execution resources available.

A warp may be active but not eligible because it is waiting for:
- A memory dependency.
- Another instruction to finish.
- A synchronization barrier.
- An available execution pipeline.
10.3 Understand 1.99 active versus 0.15 eligible warps
Imagine that a scheduler has two resident warps.

Most of the time, neither warp can issue its next instruction.

Your measured result is approximately:
\[
\text{Active warps}=1.99
\]

\[
\text{Eligible warps}=0.15
\]

That tells us the warps are present, but they are frequently unable to make forward progress at the instruction-issue stage.
10.4 Understand No Eligible = 85.12%
This is the key result.
\[
\boxed{85.12\%}
\]

For approximately 85% of scheduler cycles, no resident warp is eligible.

This explains why your issue slots are busy only around 14.86% of the time.
\[
100-85.12=14.88\%
\]

The near agreement with the measured issue-slot utilization is not accidental.

Your warp schedulers are often idle because they have no ready instructions—not because there is no work remaining in the GEMM.
10.5 Why are there only about two active warps per scheduler?
Now we connect this result to the tutorial's shared-memory allocation.

The default code uses three pipeline stages:
auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int<64>{};
auto bP = Int<3>{};


Each shared-memory A tile contains:
\[
128\times64=8192\text{ FP16 elements}
\]

At two bytes per element:
\[
8192\times2=16,384\text{ bytes}
\]

Three pipeline stages require:
\[
3\times16,384=49,152\text{ bytes for A}
\]

B requires the same amount.

Therefore:
\[
\text{Shared memory per CTA}
\approx49,152+49,152
\]

\[
\boxed{98,304\text{ bytes}=96\text{ KiB}}
\]

This calculation assumes the expected layout storage footprint; the precise allocation should be confirmed in Launch Statistics.

An H100 has 228 KiB of shared memory per SM. 
Hopper Tuning Guide 13.5 documentation


Therefore, considering shared memory alone:
\[
2\times96=192\text{ KiB}
\]

Two CTAs can fit.

But:
\[
3\times96=288\text{ KiB}
\]

Three cannot fit.

This strongly suggests that the three-stage shared-memory allocation limits residency to two CTAs per SM.

Each CTA has four warps:
\[
2\times4=8\text{ resident warps per SM}
\]

H100 SMs have four warp schedulers, giving:
\[
8/4=\boxed{2\text{ warps per scheduler}}
\]

That agrees remarkably well with your measured value of 1.99.
10.6 Estimate theoretical occupancy
H100 supports up to 64 concurrent warps per SM. 
Hopper Tuning Guide 13.5 documentation


With eight resident warps:
\[
\text{Occupancy}=\frac{8}{64}\times100
\]

\[
\boxed{12.5\%}
\]

This is a strong inference, not a replacement for the profiler's Occupancy section.

It tells us why relatively few warps are available to hide latency.

But low occupancy is not automatically bad for Tensor Core GEMM: large shared-memory tiles can enable high data reuse, and asynchronous Tensor Core operations can achieve substantial throughput with relatively few warps.

The important problem is the combination of low residency and low eligibility.

 13. Warp State Statistics
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/698167a0-609a-4620-8b1f-28f53b61ab14" />
14.  Warp State Statistics
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/95dc1779-539f-40e8-bc2e-1571e4a9c881" />
15.  Warp State Statistics
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/2f5103d4-5284-4af4-8c61-c713309a4496" />

Step 11. Analyze Warp State Statistics
Screenshots 13–15 — Warp State Statistics and stall chart

Your screenshot reports:
Metric	Value
Warp Cycles Per Issued Instruction	13.36
Warp Cycles Per Executed Instruction	13.36
Average Active Threads Per Warp	32
Average Not Predicated Off Threads	31.37
Dominant stall	Long Scoreboard
Long Scoreboard wait	Approximately 8.3 cycles per issued instruction

11.1 Understand Warp Cycles Per Issued Instruction = 13.36
This is an average warp-state metric.

It is not a literal latency measurement for each SASS instruction.

Nsight Compute accumulates warp states and normalizes them by issued instructions to understand the average number of cycles spent in different states.

Your result is:
\[
\boxed{13.36\text{ warp cycles per issued instruction}}
\]

That is consistent with a kernel experiencing significant stalls.
11.2 Long Scoreboard = 8.3 cycles
Your report says approximately 8.3 cycles are spent waiting for long-scoreboard dependencies per issued instruction.

Calculate the fraction:
\[
\frac{8.3}{13.36}\times100
\]

\[
\boxed{62.1\%}
\]

That is a substantial fraction of the total average warp cycles per issued instruction.

This is our strongest evidence of an important latency-related bottleneck.
11.3 What exactly is a Long Scoreboard Stall?
A long scoreboard tracks dependencies involving the L1/TEX memory subsystem.

Consider:
float value = A[index];
float result = value * 2.0f;


The multiplication cannot use value until the load has produced the data.

The warp may be waiting because the dependency has not resolved.

Nsight Compute records a long-scoreboard stall when a warp is waiting for such a memory dependency.

NVIDIA's documentation explicitly distinguishes memory latency from memory bandwidth saturation. 
NsightCompute


For your kernel:
- DRAM utilization is low.
- Long-scoreboard stalls are high.
- Eligible warp count is very low.
That combination points strongly toward insufficient memory-latency hiding.
11.4 Does Long Scoreboard mean LDGSTS is definitely the culprit?
Not yet.

The consumer instruction where a stall is sampled may not be the instruction that originally initiated the memory access.

We must identify the dependency chain.

Your asynchronous A/B copies are plausible contributors, but the screenshot does not prove which particular instruction causes the most waiting.

Use Source Counters and the SASS dependency information to locate the relevant instructions.
11.5 What does Stall Barrier mean?
Your chart also shows a substantial Barrier stall category.

A Barrier stall means a warp is waiting for synchronization involving other warps in its CTA.

Your tutorial includes:
__syncthreads();


and asynchronous synchronization constructs.

We need to inspect exactly which SASS instructions account for these stalls.

A significant barrier stall does not automatically mean warp divergence or an incorrect synchronization design.
11.6 What does Stall GMMA mean?
This category relates specifically to GMMA/WGMMA-related waiting conditions.

Your chart shows a comparatively small GMMA stall contribution.

That suggests GMMA-specific issue stalls are not the dominant sampled wait condition.

However, it does not mean WGMMA completion latency is irrelevant to the overall pipeline.
11.7 Why are average active threads per warp approximately 32?
Your report shows:
Avg. Active Threads Per Warp: 32
Avg. Not Predicated Off Threads Per Warp: 31.37


That is encouraging.

Most warp instructions execute with nearly all their threads participating.

This makes severe thread divergence unlikely to be the dominant performance problem.

It also reinforces that your main issue is probably not ordinary branch divergence.

Step 12. Connect the stalls to your actual CUDA C++ mainloop
This is where the Nsight analysis becomes particularly useful.

Your wgmma_sm90.cu mainloop contains the following sequence, simplified from NVIDIA's source: 
GitHub

// Stage the next A/B tiles
copy(copy_a, ...);
copy(copy_b, ...);
cp_async_fence();

// Wait for asynchronous copies
cp_async_wait<0>();

// Prepare and issue WGMMA
warpgroup_fence_operand(tCrC);
warpgroup_arrive();

cute::gemm(
    mma,
    tCrA(_,_,_,k_pipe_read),
    tCrB(_,_,_,k_pipe_read),
    tCrC
);

warpgroup_commit_batch();

// Wait for WGMMA completion
warpgroup_wait<0>();

warpgroup_fence_operand(tCrC);


There are two particularly interesting points.
12.1 cp_async_wait<0>()
This waits for all prior asynchronous copy groups issued by the calling thread to complete.

That is a conservative synchronization point.

Even though the code uses three shared-memory stages, waiting for all outstanding copy groups can reduce the ability to overlap copying with computation.

For example:
<img width="1072" height="321" alt="image" src="https://github.com/user-attachments/assets/e048d63a-7986-47e0-8488-b52c14491268" />

 A more deeply overlapped implementation may allow the next copy stage to proceed while the current MMA batch executes, using synchronization only where the needed operands or memory buffers require it.

This is a hypothesis worth testing against the source and SASS, not a suggestion to remove waits indiscriminately.

Removing necessary synchronization would make the kernel incorrect.
12.2 warpgroup_wait<0>()
This waits until no committed WGMMA batches remain outstanding.

It ensures that the asynchronous MMA work has completed before the code proceeds to operations that depend on completion.

But waiting after each K-tile can limit the opportunity to have multiple MMA batches in flight.

A more aggressive design may overlap MMA work across iterations if accumulator dependencies and shared-memory lifetime rules permit it.

You cannot safely remove or move this wait without reconsidering those dependencies.
12.3 Why this matters for your report
Your profiler shows:
- Long scoreboard: high.
- Barrier stalls: noticeable.
- Eligible warps: only 0.15 per scheduler.
- Tensor pipeline: not saturated.
The conservative waiting strategy is a plausible contributor to this combination.

The next investigation is to identify exactly where the waiting occurs and whether it can be hidden by restructuring the pipeline.
 
16. Instruction Statistics
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/f28dfd2e-e9eb-4be7-94e1-367fe9322e68" />
17. Instruction Statistics
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/496ab5d4-a958-4458-ae1b-52a0f5209553" />
18. Instruction Statistics
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/313a47bb-d8a9-489b-aa32-2edc157d5b1d" />
19. Instruction Statistics
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/2a6a9258-6fb0-41c9-b31c-c630f075f834" />
20. Instruction Statistics
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/d91c052a-1f3d-435c-a126-4bd4cb6bd21c" />
Step 13. Analyze Instruction Statistics
Screenshots 16–20 — Instruction Statistics and Executed Instruction Mix

Your final screenshots show the low-level SASS instruction mix.

The summary reports:
Metric	Count
Executed Instructions	64,723,200
Issued Instructions	64,733,000

These totals are close, which is normal for this workload.
13.1 What does the Executed Instruction Mix tell us?
The most prominent SASS instruction groups include:
Opcode	General function	Connection to your kernel
UMOV	Move a value to a uniform register	Kernel bookkeeping and address-related values
IADD3	Integer addition	Address and index calculations
LDGSTS	Asynchronous global-to-shared memory copy	Loading A/B tiles with cp.async
UIADD3	Uniform integer addition	Loop/index/address bookkeeping
HGMMA	Hopper warpgroup matrix multiplication	Actual Tensor Core GEMM
IMAD	Integer multiply-add	Index and address calculations
PRMT	Byte permutation	Data or value manipulation
LDS	Shared-memory load	Shared-memory-related operations
STG	Global-memory store	Writing C
WARP* instructions	Warpgroup coordination	WGMMA synchronization

These are broadly consistent with your source code.
13.2 Why is there so much integer arithmetic?
A GEMM kernel performs more than matrix multiplications.

It must also:
- Determine which tile each CTA owns.
- Calculate addresses for A, B and C.
- Advance K-tile loop counters.
- Manage pipeline stage indices.
- Generate addresses for asynchronous loads and stores.
- Coordinate synchronization.
IADD3, UIADD3, IMAD, and related instructions implement much of this work.

Integer instructions are therefore not surprising in a highly optimized Tensor Core kernel.

However, excessive address-generation overhead can reduce the rate at which useful work is issued.

Your issue-slot utilization is low, but we cannot conclude from opcode counts alone that address calculations are the dominant problem.
13.3 Why is HGMMA an important result?
The instruction mix confirms that the kernel is actually executing Hopper's warpgroup Tensor Core instruction family.

This matters because you compiled the executable on your RTX 5080 but executed it on H100.

Your Nsight report provides hardware-level evidence that the intended Hopper matrix-multiplication instructions executed.
13.4 One HGMMA instruction does much more than one FLOP
Your source uses an MMA atom corresponding to a \(64\times64\times16\) operation.

Its mathematical work is:
\[
2\times64\times64\times16
\]

\[
=\boxed{131,072\text{ FLOPs}}
\]

per logical MMA operation of that shape.

That is why we must not interpret the HGMMA SASS instruction count as a conventional scalar FLOP count.

Furthermore, Nsight's executed-instruction counters are generally measured at warp-level granularity, while WGMMA is a coordinated warpgroup operation.

The profiler's Tensor Core operations counter is the better source for determining total matrix arithmetic.
13.5 What about LDGSTS?
This is particularly important for your next project.

Your current kernel uses cp.async, reflected by LDGSTS.

The wgmma_tma_sm90.cu tutorial uses TMA instead.

One of the most valuable comparisons will be:
Current WGMMA tutorial	Future WGMMA + TMA tutorial
Thread-driven cp.async copies	TMA-driven tile copies
LDGSTS activity	TMA activity
cp.async completion waits	Transaction-barrier coordination
Explicit copy layout	TMA descriptors and partitioning
WGMMA computation	WGMMA computation

This is not a claim that TMA will always be faster. It is a way to examine how changing the data movement mechanism affects memory behavior, synchronization, and Tensor Core utilization.

Step 14. Put all the evidence together
We now have enough measurements to construct a useful working diagnosis.
Observation	Measurement	Interpretation
Dense FP16 Tensor Core throughput	42.29% of peak	Substantial compute headroom
L1/TEX throughput	82.82%	High internal memory activity
DRAM throughput	16.41%	HBM not saturated
L2 hit rate	79.75%	Considerable on-chip caching
Active warps per scheduler	1.99	Low resident-warp population
Eligible warps per scheduler	0.15	Most warps often unable to issue
No eligible warp	85.12%	Severe lack of ready work
Long-scoreboard stall	~8.3 cycles/instruction	Significant memory dependency latency
Global-store sector utilization	~50%	Potentially inefficient output writes
TMA instructions	0	Expected for this tutorial

My current performance diagnosis
Confirmed: WGMMA is functioning correctly.

The reported FP16 HGMMA operation count matches the expected GEMM FLOPs.



Strong evidence: Memory-dependency latency is not being hidden sufficiently.

High long-scoreboard stalls and very low scheduler eligibility support this conclusion.



Likely contributor: Low CTA residency.

The three-stage pipeline consumes approximately 96 KiB shared memory per CTA, consistent with two CTAs per SM.



Possible optimization: Improve copy/computation overlap.

The existing cp_async_wait<0>() and warpgroup_wait<0>() synchronization warrants investigation.



Separate optimization candidate: Improve output-store transaction efficiency.

The profiler reports that only about half of each global-store sector is useful.



Step 15. What should you investigate next?
I would prioritize these experiments in this order.
Experiment A — Identify the instructions responsible for Long Scoreboard stalls
Open your existing report in Nsight Compute.

Select the Source tab and examine SASS along with the warp-stall sampling columns.

Look for instructions associated with LDGSTS, waits, and other memory dependencies.

Remember that stall samples can appear on the consumer instruction rather than the original load.
Experiment B — Examine the Occupancy section
We need to verify our shared-memory residency calculation.

Look for:
- Shared memory per CTA.
- Registers per thread.
- Theoretical occupancy.
- Achieved occupancy.
- Occupancy limiter.
This will tell us whether shared memory is the controlling residency resource or whether register usage also constrains it.
Experiment C — Test pipeline stages
You could compare a two-stage and three-stage version of wgmma_sm90.cu.

The tradeoff is interesting:
Pipeline configuration	A/B shared memory per CTA	Possible residency effect
2 stages	Approximately 64 KiB	May allow more CTAs
3 stages	Approximately 96 KiB	Consistent with 2 CTAs per SM
4 stages	Approximately 128 KiB	May reduce residency further

These are approximate A/B storage requirements, excluding any additional allocations or alignment overhead.

More pipeline stages do not automatically mean better performance.

Additional stages can improve latency hiding but increase shared-memory consumption. Reducing stages can allow more resident CTAs but reduce copy/computation overlap.

Your profiler provides a concrete reason to study that tradeoff.
Experiment D — Compare wgmma_sm90 with wgmma_tma_sm90
When you next rent an H100, use the same problem dimensions, precision, output layout and benchmarking method wherever possible.

Then compare:
- Dense Tensor Core throughput.
- Long Scoreboard stalls.
- Eligible warps per scheduler.
- TMA and LDGSTS activity.
- Kernel execution time.
Because the tutorials differ in more than just one instruction, treat the comparison as a comparison of two implementations, not proof of the isolated benefit of TMA.

Step 16. What you have learned from this first report
You have already uncovered a meaningful GPU performance story:
1. The kernel performs approximately 214.75 billion arithmetic operations in 651.87 μs.
2. It achieves 329.43 TFLOP/s during the captured launch.
3. It reaches 42.29% of the dense FP16 HGMMA peak.
4. External HBM bandwidth is not saturated.
5. There are very few ready warps despite plenty of GEMM work remaining.
6. Memory-dependency stalls consume a substantial fraction of warp cycles.
7. Three-stage shared-memory buffering likely contributes to low warp residency.
8. Global output stores provide an additional optimization opportunity.
This is the sort of reasoning a GPU performance engineer should build from Nsight Compute evidence.

It is much more useful than looking only at the achieved TFLOP/s number.
What I want to examine in your remaining screenshots
The most valuable remaining sections would be Occupancy, Launch Statistics, Source Counters, any shared-memory bank-conflict metrics, and SASS-level stall attribution.

Those can help us decide whether the kernel is primarily limited by the copy pipeline, synchronization, residency, or another dependency.

Send your remaining screenshots in the next message. I'll continue the same detailed analysis, connect their measurements to the findings above, and revise any hypotheses that the new evidence contradicts.

The most important lesson from this first batch: high Tensor Core performance requires more than fast matrix instructions. It requires keeping the data, synchronization, and instruction schedules moving so the Tensor Cores receive useful work consistently.

**************************************************
What is preventing your H100 WGMMA kernel from keeping more warps active?

We previously suspected that shared-memory consumption was restricting occupancy. Your new report confirms that shared memory is the limiting resource, and it also reveals a separate tail effect and specific source-code locations worth investigating.
**************************************************
21. Launch Statistics
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/c2ac58b1-a228-4a10-a770-1968661bfec0" />
Step 17. Launch Statistics — Understand your kernel's hardware resource requirements
Screenshot 1 — Launch Statistics

The following values come directly from your screenshot.
Metric	Measured value
Grid Size	1,600 CTAs
Block Size	128 threads
Cluster Size	1 CTA
Registers Per Thread	114
Static Shared Memory Per Block	0
Dynamic Shared Memory Per Block	98.30 KB
Driver Shared Memory Per Block	1.02 KB
Shared Memory Configuration Size	200.70 KB
SMs	132
TPCs	66
Waves Per SM	3.03

17.1 Grid Size: 1,600
Your execution grid is:
dim3 dimGrid(40, 40, 1);


Therefore:
\[
40\times40=\boxed{1600\text{ CTAs}}
\]

Each CTA calculates one output tile of size \(128\times128\).

Your total output matrix is \(5120\times5120\):
\[
\frac{5120}{128}\times\frac{5120}{128}=40\times40
\]

This confirms that the full GEMM output is covered exactly, without partially filled edge tiles.
17.2 Block Size: 128 threads
Your kernel launches:
dim3 dimBlock(128, 1, 1);


Each CTA contains:
\[
128/32=4\text{ warps}
\]

On Hopper, four warps form one warpgroup.

This matters because WGMMA instructions operate cooperatively across the entire warpgroup.

Each CTA in your experiment consists of one warpgroup executing the WGMMA kernel.
17.3 Cluster Size: 1
Your report says:
Cluster Size = 1


This means each thread-block cluster contains one CTA.

It does not mean the GPU can run only one CTA per SM.

These are separate concepts:
- Cluster size: How many CTAs belong to one cluster.
- CTA residency: How many CTAs can run simultaneously on one SM.
Your kernel uses cluster size 1, but its resource configuration permits two resident CTAs per SM.

This is an important distinction for when you later study wgmma_tma_sm90.cu, where the cluster configuration may differ.
17.4 Registers Per Thread: 114
Each thread requires 114 registers, according to the launch statistics.

A simple register-count calculation gives:
\[
114\times128=14,592
\]

So each CTA requires at least approximately 14,592 32-bit register slots before accounting for allocation granularity.

The H100 has 65,536 32-bit registers per SM.

Ignoring register-allocation rounding:
\[
\left\lfloor\frac{65,536}{14,592}\right\rfloor=4
\]

That agrees with the Occupancy section's Register Block Limit = 4.

This is not the limiting resource for your current kernel, because shared memory restricts residency to two CTAs.
17.5 Dynamic Shared Memory: 98.30 KB
This is the most important launch statistic.

Your source defines:
auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int<64>{};
auto bP = Int<3>{};


Here, bP = 3 specifies three pipeline stages.

One A tile contains:
\[
128\times64=8192\text{ elements}
\]

Since FP16 uses 2 bytes:
\[
8192\times2=16,384\text{ bytes}
\]

One B tile has the same size.

Thus, one stage needs:
\[
16,384+16,384=32,768\text{ bytes}
\]

Three stages need:
\[
3\times32,768
=\boxed{98,304\text{ bytes}}
\]

That's exactly:
\[
98,304/1000=98.304\text{ decimal KB}
\]

or:
\[
98,304/1024=\boxed{96\text{ KiB}}
\]

Your profiler's 98.30 KB confirms the expected shared-memory allocation for the three-stage A/B pipeline.
17.6 What is Driver Shared Memory = 1.02 KB?
Nsight Compute also reports approximately 1.02 KB reserved for the driver per CTA.

This corresponds to roughly 1,024 bytes of additional shared-memory allocation.

Therefore, the total resource consumption is approximately:
\[
98,304+1,024
=\boxed{99,328\text{ bytes per CTA}}
\]

When calculating CTA residency, include this allocation rather than considering only the memory explicitly requested by the kernel.
17.7 Shared Memory Configuration Size = 200.70 KB
The report also shows:
\[
200.70\text{ KB}\approx200,704\text{ bytes}
\]

This is the configured shared-memory resource size that the occupancy calculation uses.

Now compare one, two, and three CTAs:
CTAs per SM	Shared memory required	Fits?
1	99,328 bytes	Yes
2	198,656 bytes	Yes
3	297,984 bytes	No

Mathematically:
\[
\left\lfloor\frac{200,704}{99,328}\right\rfloor
=\boxed{2\text{ CTAs}}
\]

This directly explains the shared-memory occupancy limit.

One useful detail: the configured shared-memory resource is not necessarily equal to the physical maximum shared-memory capacity advertised for the H100. Use the value reported for the actual kernel launch when analyzing its occupancy.
 
22. Occupancy
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/3e783ec4-5a02-4c02-91bf-29e80fac5655" />

23. Occupancy
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/c0a3a7c0-9d1b-4060-8e9f-8da508400fc3" />

24. Occupancy
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/9d173fac-27fe-4556-8bd6-16292e4d348c" />
25. Occupancy
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/be69d592-7800-45fc-965b-00f4d0b159dd" />
26. Occupancy
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/97d14bde-b577-49db-aa03-efc3b9d616e7" />
27. Occupancy
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/e4bb0513-bdfe-4501-a1e2-3611714fec67" />

Step 18. Occupancy — Your central performance limitation is confirmed
Screenshot 2 — Occupancy

Your report gives:
Metric	Value
Theoretical Occupancy	12.50%
Achieved Occupancy	12.42%
Theoretical Active Warps Per SM	8
Achieved Active Warps Per SM	7.95
Block Limit Registers	4
Block Limit Shared Memory	2
Block Limit Warps	16
Block Limit SM	32
Block Limit Barriers	32

18.1 Derive theoretical occupancy
The H100 supports a maximum of 64 resident warps per SM.

Your kernel can have:
\[
2\text{ CTAs per SM}\times4\text{ warps per CTA}
=8\text{ warps per SM}
\]

Therefore:
\[
\text{Theoretical occupancy}
=\frac{8}{64}\times100
\]

\[
\boxed{12.50\%}
\]

This is exactly the value Nsight Compute reports.
18.2 Derive achieved occupancy
The measured average is 7.95 active warps per SM.

Therefore:
\[
\text{Achieved occupancy}
=\frac{7.95}{64}\times100
\]

\[
\boxed{\approx12.42\%}
\]

Again, this matches the profiler.
18.3 Why are theoretical and achieved occupancy almost identical?
Compare:
\[
12.50\%\quad\text{versus}\quad12.42\%
\]

The difference is only 0.08 percentage points.

This means the kernel generally achieves the residency permitted by its resource allocation.

You do not have a large gap between theoretical and achieved occupancy. You have a low theoretical occupancy ceiling.

That is a very different diagnosis.

If theoretical occupancy were 50% but achieved occupancy were 12%, we would investigate why the hardware failed to sustain the possible residency.

In your case, the hardware is maintaining almost all the available residency.

The limitation is primarily imposed by the kernel's required resources.
18.4 Identify which resource actually limits occupancy
Your profiler provides several block limits:
<img width="1243" height="236" alt="image" src="https://github.com/user-attachments/assets/a5347776-3eda-4e41-a97f-1a5c90712654" />
The smallest block limit determines the maximum number of resident CTAs. Scale is relative to 32 CTA slots.
The calculation is:
\[
\text{Resident CTAs per SM}
=\min(4,2,16,32,32)
\]

\[
=\boxed{2}
\]

Shared memory is definitively the occupancy-limiting resource.

NVIDIA's occupancy documentation explains that registers, warps, shared memory, barriers and hardware block limits jointly determine residency. 
NsightCompute

18.5 How this changes our interpretation of your earlier Scheduler Statistics
In the first batch, we observed:
Active Warps Per Scheduler = 1.99
Eligible Warps Per Scheduler = 0.15
No Eligible = 85.12%


Now we understand why the active-warp count is close to two.

Your H100 has four warp schedulers per SM.

With eight resident warps:
\[
8/4=2\text{ warps per scheduler}
\]

So the reported 1.99 active warps per scheduler is exactly what we would expect.

The kernel's problem is not that blocks failed to launch.

It is that each scheduler has very few resident warps available to hide dependencies, and most of those warps frequently cannot issue.

Low available residency plus memory-related stalls is a more precise explanation of the observed underutilization.
 
28. GPU and Memory Workload Distribution
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/bbe15388-dcc8-459f-abea-eed2fe704c8c" />
 
29. GPU and Memory Workload Distribution
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/d553a2cc-2eb0-4bed-84ba-abc4955dc23a" />
 
30. GPU and Memory Workload Distribution
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/6dca578d-703f-452b-9585-3e067b72803b" />
 
31. GPU and Memory Workload Distribution
<img width="1858" height="1134" alt="image" src="https://github.com/user-attachments/assets/f86b1250-5fcd-483c-aa4a-631855410f62" />

 Step 19. Understand GPU and Memory Workload Distribution
Your first screenshot reports the following measurements.
Metric	Value
Average SM Active Cycles	825,167.17
Average SMSP Active Cycles	823,939.69
Average L1 Active Cycles	825,167.17
Average L2 Active Cycles	1,065,944.89
Average DRAM Active Cycles	280,188.60
Total SM Elapsed Cycles	123,961,854
Total SMSP Elapsed Cycles	495,847,416
Total L1 Elapsed Cycles	123,961,854
Total L2 Elapsed Cycles	89,041,760
Total DRAM Elapsed Cycles	68,290,304

19.1 What does “Active Cycles” mean?
Start with one SM.

An SM has active cycles when it has at least one warp resident and in flight.

Consider this simplified example:
<img width="1213" height="156" alt="image" src="https://github.com/user-attachments/assets/4bac66cb-0142-4957-a94b-3016e5670335" />
For this example:
\[
\text{Active Cycles}=7
\]

\[
\text{Elapsed Cycles}=10
\]

The SM's active-cycle fraction is:
\[
\frac{7}{10}\times100=70\%
\]

But here is the crucial distinction:

An active SM does not necessarily issue instructions every active cycle.

It can have resident warps that are all stalled.

This distinction is especially relevant to your kernel because your earlier Scheduler Statistics reported 85.12% of cycles with no eligible warp. An SM can count as active during those cycles.

NVIDIA confirms that sm__cycles_active counts cycles with at least one warp in flight. 
NVIDIA Developer Forums

19.2 What does Average SM Active Cycles = 825,167.17 mean?
Your H100 has 132 SMs.

The profiler measured active cycles for each SM and then calculated their average.

Conceptually:
\[
\text{Average SM Active Cycles}
=
\frac{\sum_{i=0}^{131}\text{ActiveCycles}_i}{132}
\]

The result is:
\[
\boxed{825,167.17\text{ cycles}}
\]

That means each SM was active for approximately 825,167 cycles on average during this kernel.

It does not mean every SM was active for exactly that duration.

Some SMs were active longer; others finished their assigned work sooner.
19.3 What does Total SM Elapsed Cycles mean?
Your screenshot reports:
\[
\boxed{123,961,854\text{ cycles}}
\]

This is aggregated across the GPU's SM counter instances.

It is not the duration of a single SM.

We can divide by 132:
\[
\frac{123,961,854}{132}
\approx939,105
\]

That gives approximately 939,105 elapsed cycles per SM counter instance.

Now calculate the fraction of time SMs were active:
\[
\frac{825,167.17}{939,105}\times100
\]

\[
\boxed{\approx87.9\%}
\]

This is a derived active-cycle fraction—not a reported SM arithmetic-throughput percentage.
Why is this different from 42.29% Compute Throughput?
Because they measure different things.
Measurement	Interpretation
~87.9% SM active-cycle fraction	SMs generally had resident work
42.29% Compute Throughput	Achieved compute throughput relative to its reference peak
12.42% Achieved Occupancy	Average resident warps relative to maximum warp capacity

These measurements are compatible.

Your H100 can have an SM active for most of the kernel while only eight warps reside there and many of them spend time waiting.

That is exactly why understanding these metrics separately matters.

Step 20. What is an SMSP?
Your screenshot includes:
Average SMSP Active Cycles = 823,939.69


An SMSP is an SM subpartition.

For your Hopper H100, each SM contains four subpartitions associated with warp scheduling and execution resources.

Conceptually:
One H100 Streaming Multiprocessor (SM)
SMSP 0

Warp scheduler and execution resources


SMSP 1

Warp scheduler and execution resources


SMSP 2

Warp scheduler and execution resources


SMSP 3

Warp scheduler and execution resources




Simplified organization, not a complete hardware block diagram.
With 132 SMs and four SMSPs per SM:
\[
132\times4=\boxed{528\text{ SMSPs}}
\]

Your workload table reports:
\[
\text{Total SMSP Active Cycles}=435,040,154
\]

Divide by the SMSP average:
\[
\frac{435,040,154}{823,939.69}\approx528
\]

This confirms that the profiler is aggregating the measurements across 528 SMSP counter instances.
Compare SM and SMSP active cycles
Unit	Average active cycles
SM	825,167.17
SMSP	823,939.69

These are very close.

This suggests that the four subpartitions of an SM generally remain active for similar portions of the execution.

But it does not establish that their instruction throughput is identical.

Activity and throughput are not the same measurement.

Step 21. Understand L1, L2, and DRAM Active Cycles
Your screenshot reports:
Hardware component	Average active cycles
L1	825,167.17
L2	1,065,944.89
DRAM	280,188.60

You might look at this table and conclude:
L2 has more active cycles than L1, so L2 must be the busiest or slowest component.

That conclusion would not be justified.
21.1 Why can't we directly compare these cycle counts?
SMs, L2 caches, and DRAM interfaces have different clock domains, resource organizations, and definitions of activity.

For example:
\[
1,065,944\text{ L2 cycles}
\]

and:
\[
825,167\text{ SM cycles}
\]

do not necessarily represent comparable amounts of real time or work.

The correct comparisons are generally:
- SM active cycles versus SM elapsed cycles.
- L2 active cycles versus L2 elapsed cycles.
- DRAM active cycles versus DRAM elapsed cycles.
Do not divide an L2 cycle count by an SM cycle count and interpret the result as a bandwidth or latency ratio.
21.2 How many L2 and DRAM counter instances are represented?
Your second screenshot includes the workload distribution totals.

For L2:
\[
\frac{85,275,591}{1,065,944.89}\approx80
\]

For DRAM:
\[
\frac{11,207,544}{280,188.60}=40
\]

Therefore, your report aggregates approximately:
Counter level	Reporting instances
SM	132
SMSP	528
L1	132
L2	80
DRAM	40

These are the counter-instance counts inferred from the reported sums and averages. They should not automatically be interpreted as the number of physical memory chips or independent memory channels.
21.3 What can we learn from these measurements?
The active-cycle data can help determine whether a unit was active throughout a large portion of the kernel.

But it does not tell us by itself whether the unit transferred data at maximum throughput.

For actual memory bandwidth pressure, return to your earlier Memory Workload Analysis and Speed of Light sections.

NVIDIA distinguishes activity counters from throughput counters because being active is not equivalent to operating at peak bandwidth. 
NsightCompute


Step 22. Understand the Workload Distribution table
Screenshot 2 — Average, Min, Max, and Sum

This table is particularly useful.
Component	Average active cycles	Minimum	Maximum
SM	825,167	804,757	936,301
SMSP	823,940	801,211	936,606
L1	825,167	804,757	936,301
L2	1,065,945	1,059,155	1,071,739
DRAM	280,189	278,208	282,200

Let's focus on the SM row.
22.1 What does Average mean?
It is the average number of active cycles across the 132 SMs.
\[
\boxed{\text{Average}=825,167}
\]

22.2 What does Min mean?
The least-active SM accumulated:
\[
\boxed{\text{Min}=804,757}
\]

That SM finished with fewer active cycles than the average SM.

This does not necessarily mean it was slower or faster at processing individual instructions; it may simply have received less work or completed its assigned CTAs earlier.
22.3 What does Max mean?
The most-active SM accumulated:
\[
\boxed{\text{Max}=936,301}
\]

This SM was active for substantially longer than the average.

Now calculate the difference between the most-active and least-active SMs:
\[
936,301-804,757
\]

\[
=\boxed{131,544\text{ cycles}}
\]

That is a meaningful workload-duration difference.
22.4 What does Sum mean?
Your screenshot reports:
\[
\boxed{\text{Sum}=108,922,066}
\]

This is the total number of active SM cycles across the 132 SM instances.

We can verify:
\[
\frac{108,922,066}{132}
\approx825,167.17
\]

Exactly the average reported by Nsight Compute.

The four columns are therefore:
- Average: Typical active-cycle count across instances.
- Min: Lowest active-cycle count.
- Max: Highest active-cycle count.
- Sum: Total across all instances.
These are statistics across different hardware instances, not values sampled at four different times.

Step 23. Understand the “SMs Workload Imbalance” warning
Screenshots 1–3 — SMs, SMSPs, and L1 Slices Workload Imbalance

Nsight Compute reports:
SMs Workload Imbalance
Est. Speedup: 10.43%


It also reports that the most-active SM is 11.87% above the average according to its warning calculation, while the least-active SM is 2.47% below average.

Let's understand why.
23.1 Example: Two SMs executing unequal workloads
Imagine two SMs:
<img width="1210" height="246" alt="image" src="https://github.com/user-attachments/assets/7ed37186-2ce8-4e08-a8d8-4f7029b767d3" />
Both SMs execute in parallel, but SM 1 finishes earlier. The overall kernel still waits for SM 0 to finish.
The kernel's completion time depends on the slowest outstanding work.

Ideally, we would distribute work so the SMs finish at approximately the same time.

That is the motivation behind the workload imbalance warning.
23.2 Apply this to your H100
<img width="1239" height="197" alt="image" src="https://github.com/user-attachments/assets/f0c33bef-43ae-4e98-a031-17c37e637d87" />
The most-active SM accumulated substantially more active cycles than the least-active SM.

That's evidence of unequal work duration across SMs.

It is not direct evidence that the most-active SM processed instructions more slowly.
23.3 Why does the profiler estimate 10.43% speedup?
The profiler recognizes that some SMs remain active longer than others.

If the work could be distributed more evenly, the longest-running SM might finish sooner, reducing the overall kernel duration.

But 10.43% is a heuristic estimate, not a demonstrated optimization result.

It does not mean changing one line of code will automatically make your kernel 10.43% faster.

Nor should you add it to other speedup estimates in the report.
23.4 Why might your GEMM have workload imbalance?
Your output matrix is \(5120\times5120\), and your CTA tile is \(128\times128\).

Therefore:
\[
40\times40=\boxed{1600\text{ CTAs}}
\]

Each CTA performs the same nominal GEMM tile computation.

Also, 5120 is divisible by 128, and 4096 is divisible by 64. There is no partially filled boundary tile in the default problem.

This makes unequal mathematical work at the matrix boundaries an unlikely explanation.

More plausible contributors include:
- CTA scheduling differences.
- Cache and memory latency differences among CTAs.
- Partial-wave or tail effects.
- Differences in when SMs become available to receive additional CTAs.
Your earlier Launch Statistics also reported a Tail Effect warning.

That supports investigating the last wave of CTAs, but we cannot determine the precise cause of the SM imbalance from this table alone.

An additional detail: the earlier reported Waves Per SM = 3.03 does not follow the simple calculation using 1,600 CTAs, 132 SMs and the measured two-CTA residency limit. That calculation gives \(1600/(132\times2)\approx6.06\). I would not rely on the earlier 25% tail-speedup estimate quantitatively without resolving this discrepancy.
23.5 How significant is this bottleneck?
I would treat it as a secondary optimization opportunity, not the first problem to solve.

Your previous report showed:
Performance concern	Evidence
Long-scoreboard stalls	Approximately 8.3 cycles per issued instruction
Eligible warps per scheduler	0.15
No eligible warp	85.12%
SM workload imbalance	Estimated local speedup 10.43%

The scheduler and warp-stall findings suggest a more fundamental inability to keep instructions ready for execution.

Improving workload distribution could help, but it might not fix the memory and synchronization dependencies that limit WGMMA throughput.

Step 24. Why are SMSP and L1 imbalance warnings almost identical?
Your screenshots show:
Warning	Estimated speedup
SMs Workload Imbalance	10.43%
SMSPs Workload Imbalance	10.55%
L1 Slices Workload Imbalance	10.43%

These results are closely related.

An SM includes multiple execution subpartitions and an associated L1/TEX subsystem.

If an SM finishes its assigned CTAs earlier than another SM, its SMSPs and associated cache structures may also stop experiencing activity earlier.

Therefore, the imbalance can appear in multiple hardware-unit measurements.

This does not mean there are three independent performance problems whose estimated improvements can be added together.

The SM, SMSP, and L1 warnings may largely reflect the same underlying uneven execution duration.
What about L2 and DRAM?
Notice that L2 and DRAM active cycles are much more evenly distributed across their reporting instances.

For L2:
\[
\text{Min}=1,059,155
\]

\[
\text{Max}=1,071,739
\]

Their spread is approximately 1.2% of the average.

For DRAM:
\[
\text{Min}=278,208
\]

\[
\text{Max}=282,200
\]

Their spread is approximately 1.4% of the average.

This suggests that the per-instance activity imbalance is much more pronounced among SMs and SMSPs than among L2 and DRAM reporting instances.

It does not prove that memory accesses are perfectly balanced or that memory latency is irrelevant.

Step 25. Analyze Source Counters
Screenshots 3–4 — Source Counters

I would definitely not skip this section either.

Source Counters helps connect your performance measurements to particular source lines and SASS instructions.

NVIDIA describes this section as collecting branch behavior, sampled warp stalls and source-level memory inefficiency information. 
NsightCompute

25.1 Branch Efficiency = 100%
Your report shows:
Metric	Value
Branch Instructions	428,800
Branch Efficiency	100%
Average Divergent Branches	0

This is encouraging.

It means the measured branch behavior does not show divergent branch targets as a significant problem.

Your kernel contains loops and conditional code, but its control flow can still be uniform among participating warp lanes.

Conclusion: Ordinary warp branch divergence is not a leading optimization target for this kernel.
25.2 Uncoalesced Global Accesses
Your screenshot reports:
Uncoalesced Global Accesses

1,638,400 excessive sectors
111,411,200 total sectors

Est. Speedup: 1.41%


The excessive-sector ratio is:
\[
\frac{1,638,400}{111,411,200}\times100
\]

\[
\boxed{\approx1.47\%}
\]

The displayed whole-number percentage is rounded in the profiler's warning.

This means the model attributes a relatively small fraction of total sectors to excessive global-memory transactions.

Here is the distinction:

An excessive sector is a transaction the hardware could potentially avoid with a more efficient memory access pattern.

A memory operation may request more 32-byte sectors than are ideally necessary for the useful data.

That can happen due to scattered addresses, poor alignment, or another inefficient access pattern.
25.3 Where do the excessive sectors come from?
Your screenshots identify repeated entries at:
axpby.hpp:90


This is extremely useful.

The axpby routine is associated with the GEMM's output epilogue, where the accumulator is combined with the existing output according to:
\[
C=\alpha AB+\beta C
\]

Consequently, the results suggest that some of the excessive global transactions arise in the output-update path.

The repeated entries at the same source line can correspond to different compiled SASS instructions associated with that line.

Each visible entry shows 12,800 excessive sectors.

This supports our earlier observation that the epilogue's global memory access pattern deserves investigation.

However, we must be careful: the 1.41% estimated speedup in Source Counters and the earlier 36.38% estimated speedup associated with the L1TEX store-access warning come from different local diagnostic models. They are not directly comparable whole-kernel speedup measurements.
25.4 Warp Stall Sampling — This is particularly important
Your screenshot shows the following hot locations:
Source location	Samples	Share shown
wgmma_sm90.cu:235...	26,368	50%
wgmma_sm90.cu:240...	10,392	20%
wgmma_sm90.cu:235...	1,087	2%
copy_sm80.hpp:64...	807	2%
copy_sm80.hpp:64...	800	2%

The first two entries are responsible for approximately 70% of the displayed warp-stall samples.

This is a more focused clue than the overall imbalance percentage.

But 50% of stall samples does not mean that the corresponding source line consumes 50% of the kernel's execution time.

Warp Stall Sampling is statistical sampling of instruction locations and their scheduler states. The percentage is a share of samples, not an elapsed-time percentage. 
NsightCompute


Another important detail: the first and third entries can map to different SASS instructions that share the same CUDA source line.
25.5 What should we investigate at lines 235 and 240?
We need the actual instructions associated with those lines.

The screenshot truncates the locations, so it does not conclusively identify the operations.

For example, if the relevant instruction is waiting for a memory result, we would investigate its data dependencies.

If it is a synchronization wait, we would investigate the synchronization schedule and whether independent work could overlap it.

If it is an instruction repeatedly executed in the inner loop, we would examine its issue frequency and dependence chain.

Do not change the code until you know which instructions are responsible for the stall samples.

Step 26. Connect all four screenshots to the earlier profiling results
We can now create a more complete performance diagnosis.
Observation	Evidence	Priority
Tensor Core throughput below peak	Dense HGMMA at 42.29%	Important
Low theoretical occupancy	12.5%, limited by shared memory	Important
Limited eligible warps	0.15 per scheduler	High
Long-scoreboard stalls	~8.3 cycles per issued instruction	High
Uneven SM active durations	804,757–936,301 cycles	Secondary
Excessive global transactions	1,638,400 sectors	Secondary
Branch divergence	100% branch efficiency	Low concern

How I would interpret the complete report
Your kernel is not simply suffering because some SMs finish earlier than others.

The more fundamental observation is that many resident warps cannot issue instructions while the kernel is executing.

Your shared-memory configuration limits the number of resident warps. Your scheduler metrics indicate a shortage of eligible instructions. Your warp-state metrics point to memory dependencies and synchronization as important sources of waiting.

The workload-distribution section adds another effect: not all SMs remain active for the same duration, potentially creating a tail in which some GPU resources become idle before the kernel finishes.

These effects can coexist.

Step 27. What should you study next?
I recommend the following order:
1. Source tab — inspect wgmma_sm90.cu around lines 235 and 240. Find the SASS instructions and stall reasons associated with the two largest sample counts.
2. Analyze the copy and synchronization pipeline. Correlate the high-stall instructions with cp.async copying, barrier operations, and WGMMA waits.
3. Investigate the epilogue at axpby.hpp:90. Determine why some global-memory accesses create excessive sectors.
4. Compare kernel timing after targeted changes. Evaluate whether reducing shared-memory stages or changing the copy schedule improves execution time without introducing correctness problems.
I would not optimize the SM workload imbalance first. It is worth understanding, but the Source Counters and warp-stall information offer more actionable starting points.
Final assessment of these screenshots
You should keep both “GPU and Memory Workload Distribution” and “Source Counters” in your study notes.

The important lessons are:
- Active cycles are not compute cycles. An SM with stalled resident warps is still active.
- Average, Min, and Max reveal differences between hardware instances. They do not directly measure instruction throughput.
- Workload imbalance can produce an execution tail. The slowest outstanding work determines when the kernel finishes.
- Nsight speedup estimates are hypotheses, not promises.
- Source Counters helps turn a general bottleneck diagnosis into an investigation of particular instructions.
Your next most useful screenshot would be the Source tab with the SASS instructions associated with wgmma_sm90.cu:235 and wgmma_sm90.cu:240. That would let us move from general profiling analysis to investigating the actual cause of the stalls in your Hopper code.

</pre>
