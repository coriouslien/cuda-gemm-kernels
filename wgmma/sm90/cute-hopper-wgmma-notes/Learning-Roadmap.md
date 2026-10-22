<pre>
A structured through the dense CuTe material: small chunks, in program order, one at a time. 
I move to the next chunk only when the current one is clear. All source code line numbers are 
shown in the file I attached. 
</pre>
|Chunk	|Lines	|Topic|
|:------|:------|:----|
|0	|whole file|	The big picture: what the program does, end to end (below)|
|1	|L467–581, L446–464	|main and gemm: host data, which kernel runs|
|2	|L352–380	|gemm_tn: problem shape, strides, CTA tile|
|3	|L382–384	|shared-memory layouts: tile_to_shape, the GMMA atom, swizzle|
|4|	L386–392	|the copy: make_tiled_copy, the TV layout|
|5	|L394, L412–443|	the MMA atom and the launch|
|6	|L18–26, L28–81	|SharedStorage, the kernel signature, global and shared tensors|
|7	|L83–100	|copy partitioning|
|8	L123–143	|MMA partitioning, descriptors, accumulators|
|9	|L178–251	|the main loop|
|10	|L253–258	|the epilogue|
|11	|—	|verify: print_layouts.cu, then the SASS|
<pre>
Chunk 0: The big picture
What the program computes

A matrix multiplication in FP16:

C = A · Bᵀ          A is M × K,  B is N × K,  C is M × N

In CuTe’s convention, both A and B are stored with K as their second mode, which is why the product is written with Bᵀ. 
Default sizes (L487–497): M = N = 5120, K = 4096.

How the work is split

C is cut into 128 × 128 tiles. Each thread block (CTA) computes one tile:

grid = (5120/128) × (5120/128) = 40 × 40 = 1600 CTAs
each CTA: 128 threads = 1 warpgroup

To compute its tile, a CTA walks along K in chunks of 64: 4096 / 64 = 64 loop iterations.

The path the data takes inside one CTA
global memory           shared memory               registers            global memory
A tile (128×64) ─┐
                 ├─cp.async─►  sA, sB              ─wgmma─►  tCrC         ─axpby─►  C tile
B tile (128×64) ─┘            (3 stages, swizzled)   (reads smem via   (128×128 accumulator,     (128×128)
                                                      descriptors)       spread over 128 threads)
        └──────────── repeated 64 times along K ────────────┘
The four parts of the file, in calling order
main (L467)                     host: make A, B, C; call gemm once; then time 100 calls
 └─ gemm (L449)                 host: pick NT or TN from transA/transB
     ├─ gemm_nt (L264)          host: build shapes, layouts, copy, MMA; launch kernel
     └─ gemm_tn (L356)          host: same, TN variant
         └─ gemm_device (L36)   device: the kernel — one CTA computes one 128×128 tile of C

The host functions (gemm_nt, gemm_tn) do nothing but describe: sizes, layouts, the copy pattern and the MMA instruction, all as CuTe types. 
The kernel uses those descriptions. Since almost everything is a compile-time type, the description is what appears in the long demangled 
kernel names in your build log.

I'm continuing my study of CUTLASS's Hopper tutorial wgmma_sm90.cu (second pass, chunk by chunk).

Setup
- Compiling (not running) for sm_90a on my RTX 5080/5050 (SM120), CUDA 12.9, Release build
  (-O3 -DNDEBUG via CMAKE_BUILD_TYPE). No C7510 warning in Release.
- ptxas: TN kernel 112 regs, NT kernel 114 regs, 1 barrier, 0 spills; also cub::EmptyKernel (from Thrust).

Confirmed facts so far
- Loads use cp.async (SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, LDGSTS), NOT TMA.
- CTA tile 128x128x64, 3 pipeline stages, 128 threads (1 warpgroup), 98,304 B dynamic smem.
- TN: Layout_K_SW128_Atom -> ((8,16),(64,1),(1,3)):((64,512),(1,0),(0,8192)), Swizzle<3,4,3>.
- Atom in bits (8,1024):(1024,1) -> upcast to half_t (8,64):(64,1).
- TN copy TV layout ((8,16),8):((128,1),16), tile (16,64): thread t copies row t/8, K = 8*(t%8)..+7.
- MMA: MMA_64x64x16_F16F16F16_SS, 16 wgmma per k-tile; tCrA/tCrB hold smem descriptors.
- Default run uses the NT kernel (transA='N', transB='T').
- Main loop: cp_async_wait<0>() and warpgroup_wait<0>() -> no overlap of load and compute.

Where I am
- Plan: 1 main/gemm, 2 gemm_tn shapes/strides, 3 smem layouts, 4 copy, 5 MMA+launch,
  6 SharedStorage/kernel tensors, 7 copy partition, 8 MMA partition, 9 main loop,
  10 epilogue, 11 print_layouts.cu + SASS.

Chunk 1: main (L467–581) and gemm (L446–464), the host code that sets everything up and decides which kernel runs. All line numbers refer to my
attached file.
Part A: main, top to bottom
1. Query the GPU (L469–477)
cudaGetDevice(&current_device_id);
cudaGetDeviceProperties(&props, current_device_id);
cudaError_t error = cudaGetDeviceProperties(&props, 0);
It asks for the properties of the current device, then again for device 0, overwriting the first result. With one GPU both are the same device, 
so it doesn’t matter. It’s just redundant code,.

2. Architecture check (L479–483)
if (props.major != 9) { std::cout << "This example requires ... 90a"; return 0; }
Only the major compute capability is checked. My RTX 5080/5050 report 12, so the program prints the message and exits here, before any kernel 
or memory allocation. That’s the source-level reason it can’t run on my machines.
It returns 0 (success) on purpose, per the comment at L481, so automated test suites don’t count it as a failure.

3. Compile-time guard (L485, L576–578)
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
   ... the whole GEMM test ...
#else
   std::cout << "CUTLASS_ARCH_MMA_SM90_SUPPORTED must be enabled ...";
#endif
     
This macro isn’t defined in this file. It comes from CUTLASS’s configuration header and, as far as I know, depends on the CUDA toolkit 
version being new enough for SM90. My build compiled the GEMM path, so it’s defined in my setup. To see the exact condition:

grep -n "CUTLASS_ARCH_MMA_SM90_SUPPORTED" -B3 -A3 ../cutlass/include/cutlass/arch/config.h
     
4. Problem size and transpose flags (L487–505)
int m = 5120;  if (argc >= 2) sscanf(argv[1], "%d", &m);
int n = 5120;  ...argv[2]
int k = 4096;  ...argv[3]
char transA = 'N';  ...argv[4]
char transB = 'T';  ...argv[5]

Defaults: M = N = 5120, K = 4096, transA = 'N', transB = 'T', so the default run is the NT kernel. Command-line arguments override them in 
that order:
./wgmma_sm90 5120 5120 4096 T N     # TN kernel
5. Types and scalars (L507–513)
using TA = cute::half_t;  using TB = ...;  using TC = ...;  using TI = ...;
TI alpha = TI(1.0f);
TI beta  = TI(0.0f);

Everything is FP16, including alpha and beta. The kernel computes C = alpha · A·Bᵀ + beta · C; with alpha = 1, beta = 0, that’s simply C = A·Bᵀ.

6. Host data (L515–522)
thrust::host_vector<TA> h_A(m*k);    // 5120 × 4096 halfs = 40 MiB
thrust::host_vector<TB> h_B(n*k);    // 40 MiB
thrust::host_vector<TC> h_C(m*n);    // 5120 × 5120 halfs = 50 MiB
for (...) h_A[j] = TA(int((rand() % 2) ? 1 : -1));   // random ±1
for (...) h_B[j] = ...;                               // random ±1
for (...) h_C[j] = TC(0);
A and B are filled with random +1 or −1; C with zeros. The arrays are flat: the 2-D layout comes from the leading dimensions in step 8.

h_A is in host memory (CPU RAM), not on the GPU. In CUDA, global memory means the GPU’s DRAM. h_A isn’t in global memory; its GPU copy is:
</pre>
|Line	|Object	|Where it lives|
|:----|:------|:-------------|
|L515	|h_A (thrust::host_vector<TA>)|	host memory (CPU RAM)|
|L524	|d_A (thrust::device_vector<TA>)|	global memory (GPU DRAM)|
<pre>
d_A = h_A (L524) allocates the GPU memory and copies the data over. The kernel only ever sees d_A, passed as the raw pointer d_A.data().get() 
at L555. The matrix is stored as a flat 1-D array of m·k elements; its 2-D shape comes later from ldA and the strides.
2. “TA is a type define”: L507: using TA = cute::half_t; is a type alias (the modern form of typedef), local to main. One detail: inside gemm,
gemm_nt and gemm_tn, TA is a template parameter with the same name, deduced from the pointer type half_t const*. It’s a different TA that ends 
up being the same type.

3. “Shorter code instead of new or cudaMalloc, not for performance”: convenience and not performance, but the comparison needs splitting:
</pre>
|Thrust type|	Replaces|	Memory|
|:----------|:--------|:------|
|thrust::host_vector|	new[] / std::vector|	host|
|thrust::device_vector|	cudaMalloc + cudaMemcpy + cudaFree|	GPU|
<pre>
host_vector has nothing to do with cudaMalloc; on the host side, a plain std::vector would work just as well. The real convenience is
device_vector, plus the one-line copies d_A = h_A (host → GPU) and cute_result = d_C (GPU → host, L560).

On performance: host_vector uses ordinary pageable memory, not pinned memory, so these copies aren’t the fastest possible. That doesn’t 
matter here, because they happen outside the timed loop.

4. “It is RAII”: Both types free their memory in their destructors (device_vector calls cudaFree), so main needs no cleanup code, even on 
early returns.
h_A is in host memory, not on the device. global memory means GPU memory, so h_A is simply in host memory, and its device copy d_A (L524) 
is the one in global memory.
  
7. Copy to the GPU (L524–526)
thrust::device_vector<TA> d_A = h_A;    // allocates GPU memory and copies
These Thrust lines are why Thrust (and therefore CUB, and the EmptyKernel in the ptxas log) is included.

8. FLOP count and leading dimensions (L528–549)
double gflops = (2.0*m*n*k) * 1e-9;

Each of the m·n outputs needs k multiplies and k adds: 2·m·n·k floating-point operations. For the defaults that’s about 214.7 GFLOP per GEMM.

The leading dimension is the distance in memory between consecutive columns (for column-major storage):
int ldA = 0, ldB = 0, ldC = m;
if (transA == 'N') ldA = m;  else if (transA == 'T') ldA = k;
if (transB == 'N') ldB = k;  else if (transB == 'T') ldB = n;
</pre>
|Flag	|ld	|Contiguous dimension	|Becomes stride (in gemm_nt/gemm_tn)|
|:----|:--|:--------------------|:----------------------------------|
|transA = 'N'|	ldA = m|	M	|dA = (1, ldA) (L279)|
|transA = 'T'|	ldA = k	|K	|dA = (ldA, 1) (L371)|
|transB = 'T'|	ldB = n|	N	|dB = (1, ldB) (L280)|
|transB = 'N'|	ldB = k|	K	|dB = (ldB, 1) (L372)|
|C           |	ldC = m	|M (column-major)	|dC = (1, ldC) in both|

<pre>
This is where the int in the demangled stride types (tuple<C<1>, int> / tuple<int, C<1>>) comes from: the leading dimension is only known 
at runtime.

Line 520:  for (int j = 0; j < m*k; ++j) h_A[j] = TA(int((rand() % 2) ? 1 : -1)); 
• It’s initialization: every element of A gets either +1 or −1, at random.
• It converts int to TA: (rand() % 2) ? 1 : -1 is an int, and TA(...) turns it into a half_t, using TA from L507 (using TA = cute::half_t;). 
The inner int(...) cast is redundant (the expression is already an int) but harmless.

rand() returns integers, not floats
rand() returns an int between 0 and RAND_MAX. On Linux/glibc, RAND_MAX is 2,147,483,647. Using it directly would fill A with large integers, 
not floats. 
overflow gives inf first, not NaN
FP16’s largest finite value is 65,504. Converting a larger number to FP16 gives inf (infinity), not NaN. NaN comes later, from arithmetic 
involving inf.
  
The bigger reason: the FP16 accumulator
Even values well below 65,504 would cause trouble, because this GEMM also accumulates in FP16: the MMA atom is F16F16F16, meaning C, 
A and B are all FP16 (from the build log’s demangled name, and L302/L394). Each output is a sum of K = 4096 products:
0 · inf   = NaN      (possible: rand() can return 0)
inf − inf = NaN
</pre>
|Input values|	Largest possible output	|Fits in FP16 (max 65,504)?|
|:-----------|:-------------------------|:-------------------------|
|±1	|4,096	|yes|
|up to 10|	4096 × 100 = 409,600	|no, can overflow to inf|
|rand() |(up to ~2·10⁹)	far beyond|	no, already inf at conversion|
<pre> 
With ±1, each product is ±1, so each output is a sum of 4096 random ±1 terms: a whole number, typically small (around ±64), never larger than 
4096 in magnitude.

The tutorial doesn’t say why its authors chose ±1. Two likely reasons: no overflow anywhere, and exact results. FP16 represents every integer
exactly up to 2048, and the typical sums here stay far below that, so the output is easy to check. (The program doesn’t actually check it, as 
noted in Chunk 1, but it could.)

A side note: rand() is never seeded (no srand call), so it produces the same sequence on every run, and A and B are identical each time I
run the program.
TA(...) converts the ±1, not rand()’s value. The expression works in three steps:
h_A[j] = TA( int( (rand() % 2) ? 1 : -1 ) );
              //  ① rand() % 2       → 0 or 1   (int)
              //  ② ? 1 : -1         → 1 or −1  (int)
              //  ③ TA(...)          → half_t 1.0 or −1.0
1. rand() returns a large int; % 2 keeps only its last bit, giving 0 or 1.
2. The ? : turns that into 1 or −1, still an int. rand()’s actual value is used only to choose between the two; it never reaches the matrix.
3. TA(...) converts that 1 or −1 to half_t.
TA(...) converts the chosen int (1 or −1) to half_t.
On why the explicit TA(...) is written at all: as far as I know, CUTLASS’s half_t constructor from int is declared explicit, 
so writing h_A[j] = 1; without it wouldn’t compile. I can check in the checkout:
grep -n "explicit half_t" ../cutlass/include/cutlass/half.h

TA(...) returns half_t, not float.
int( (rand() % 2) ? 1 : -1 )        // → int:    1 or −1
TA( int(...) )                      // → half_t: 1.0 or −1.0   (TA = cute::half_t, L507)
half_t and float are different types:
</pre>
|Type|	Size|	Precision|
|:---|:------|:--------|
|float|	32 bits|	FP32|
|half_t|	16 bits|	FP16, max 65,504|
<pre>
Both are floating-point formats, so it’s fine to call the result “a floating-point value”. But its type is half_t, which is why h_A is a 
vector of half_t and why the GEMM is an FP16 GEMM. As far as I know, CUTLASS’s half_t stores its 16 bits in a small struct and converts 
through float internally when constructed from an int, but what we get back is still a half_t.
TA(int( (rand() % 2) ? 1 : -1 )) return a half_t, an FP16 (half-precision) value: 1.0 or −1.0.

9. Run once (L551–560)
d_C = h_C;                                   // reset C to zeros
gemm(transA, transB, m, n, k, alpha, d_A.data().get(), ldA, ..., d_C.data().get(), ldC);
CUTE_CHECK_LAST();
thrust::host_vector<TC> cute_result = d_C;   // copy back

10. Time 100 runs (L562–574)
timer.start();
for (int i = 0; i < 100; ++i) gemm(...);
double cute_time = timer.seconds() / 100;
printf("CUTE_GEMM: [%6.1f]GFlop/s (%6.4f)ms\n", gflops / cute_time, cute_time*1000);
GPU_Clock comes from cutlass/util/GPU_Clock.hpp (L13). As far as I know it uses CUDA events, so it measures GPU time between start and stop. 
Each timed call goes through gemm_nt/gemm_tn again, including cudaFuncSetAttribute on every call, so a little host overhead is included in the 
measurement.

Part B: gemm, the dispatcher (L446–464)
if (transA == 'N' && transB == 'T') return gemm_nt(...);
else if (transA == 'T' && transB == 'N') return gemm_tn(...);
assert(false && "Not implemented");
Only two of the four combinations are implemented: NT and TN.

A consequence of the Release build: with -DNDEBUG, assert(false) disappears. So if I run with N N or T T, no kernel is launched, 
no error is shown, and the program still prints a timing, measuring nothing. In a Debug build, the same call would stop with an assertion 
failure. Something to remembe when I run it on an H100.

Also, gemm and both gemm_nt/gemm_tn accept a cudaStream_t stream parameter, but it’s never used in the launch (L338, L418), so everything runs 
on the default stream.

main
 ├─ check GPU: major must be 9 (your SM120 → exits here)
 ├─ M=N=5120, K=4096, transA='N', transB='T'  (default → NT kernel)
 ├─ A, B random ±1; C zeros; all FP16; copied to GPU with Thrust
 ├─ leading dimensions → become the runtime strides dA, dB, dC
 ├─ gemm once (result not verified)
 └─ gemm 100× timed → prints GFlop/s
gemm
 └─ NT → gemm_nt,  TN → gemm_tn,  anything else → nothing (in Release)

Line 368: auto prob_shape = make_shape(M, N, K);  
make_shape(M, N, K) is  shape M, N, K in CPU, no memory address, no data, no row or column major concept.  M, N, K provide three integer numbers.
• No memory address, no data: prob_shape only holds sizes. It doesn’t point to A, B or C.
• No row- or column-major concept: A shape says how many elements there are along each mode, never where they are in memory. 
That’s the job of the strides, defined right after it (L371–373):
auto dA = make_stride(ldA, Int<1>{});   // TN: K contiguous
Shape and stride are only combined later, inside the kernel, when make_tensor builds the tensors (L66–68).
Three integers: M, N, K. With the defaults: 5120, 5120, 4096.   
make_shape(M, N, K) returns a cute::tuple<int, int, int>: the first template argument of the kernel in your build log (tuple<int, int, int>). 
The three values are plain runtime ints (L365–367 convert m, n, k to int), so they’re dynamic. Compare L376–378, where bM, bN, bK are Int<128>,
Int<64>: those are static, compile-time constants, which is why they appear as C<128>, C<64> in the kernel’s name instead of int.
It’s created on the host, but it’s then passed to the kernel by value as a kernel argument (L434: prob_shape is the first argument to
launch_kernel_on_cluster). So the GPU receives its own copy of the three integers, as shape_MNK (L36). It’s only 12 bytes (3 × 4-byte int), 
so this costs nothing.
Inside the kernel, it’s split into the shapes of the three matrices (L66–68):
select<0,2>(shape_MNK)   // (M, K)  → shape of A
select<1,2>(shape_MNK)   // (N, K)  → shape of B
select<0,1>(shape_MNK)   // (M, N)  → shape of C
That’s how one (M, N, K) triple describes all three matrices.

Line 371: auto dA = make_stride(ldA, Int<1>{}); 
It is A matrix, ldA is leading dimention of Matrix A, stride 1, row major (it is better to call k major)
the stride has two parts, and the 1 belongs to the K mode.
A is an M × K matrix, and dA = (dM, dK) (the comment on L371) gives one stride per mode:
</pre>
|Mode	|Stride	|Meaning|
|M (row index)|	ldA	|moving to the next row skips ldA elements|
|K (column index)|	Int<1>{} = 1|	moving to the next column is the next element|
<pre>
So the element A(m, k) is at offset:
offset = m · ldA + k · 1 
Each row’s K elements are contiguous: row-major, when A is viewed as M × K.
The value of ldA: for TN, main sets ldA = k (L537–538, transA == 'T'), so with the defaults dA = (4096, 1). Each row is exactly 
K = 4096 elements long, with no padding.

Static vs dynamic: ldA is a runtime int, and Int<1>{} is a compile-time constant. That’s why the build log shows the type tuple<int, C<1>> for 
the TN kernel.

On the name: CuTe code and comments usually call this K-major (the K mode has stride 1) rather than row-major. The reason: “row-major” depends 
on which index we call the row. If we drew A as K × M instead, the same memory would be “column-major”. “K-major” names the contiguous dimension
directly, so it stays unambiguous. BLAS calls the same storage ‘T’ (transposed), which is where the “T” in TN comes from. All three names 
describe the same layout:
</pre>
|Name	|Convention|
|:----|:---------|
|row-major (A as M × K)|	ordinary matrix language|
|K-major	|CuTe/CUTLASS|
|‘T’	|BLAS (transA = 'T')|
<pre>
Line 376: auto bM = Int<128>{};
Line 377: auto bN = Int<128>{};
Line 378:   auto bK = Int< 64>{};
What the three numbers mean
They’re the CTA tile sizes: how much work one thread block (CTA) handles.
</pre>
|Name|	Value|	Meaning|
|:---|:------|:--------|
|bM	|128|	the CTA computes 128 rows of C|
|bN	|128|	… and 128 columns of C → a 128 × 128 tile of C per CTA|
|bK	|64	|per main-loop iteration, it processes a 64-wide slice of K|
<pre>
So each CTA loops K / bK = 4096 / 64 = 64 times, each time multiplying a 128 × 64 piece of A by
a 64 × 128 piece of Bᵀ.

A possible confusion to avoid: bM = 128 is not the number of threads. The thread count is set
separately, by the MMA (L414: dim3 dimBlock(size(tiled_mma)), one warpgroup = 128 threads). 
Both happen to be 128, but they’re unrelated: the tile is 128 × 128 = 16,384 elements of C,
shared among 128 threads.
•  Static, integer: Int<128> is CuTe’s compile-time integer (the same as cute::C<128>). 
•  The value 128 is part of the type itself, which is why the build log shows 
tuple<C<128>, C<128>, C<64>> for cta_tiler. The object has no runtime storage at all; it’s an
empty type.

Line 376: auto bM = Int<128>{};
Line 377: auto bN = Int<128>{};
Line 378:   auto bK = Int< 64>{};
Their values are simply known to the compiler. 
The practical effect is that every calculation involving them is done 
by the compiler: tile shapes, layout sizes, the 16 wgmma per k-tile, the 8 copies per thread,
the loop bounds of the inner loops. None of that costs instructions or registers at runtime.

Compare with M, N, K (L365–367): runtime ints, unknown until the program runs. When both meet,
the result is a runtime value. For example, the grid size ceil_div(m, bM) (L416) is computed at
runtime, because m is runtime, even though bM is static.

Line 379: auto cta_tiler = make_shape(bM, bN, bK);     
•  CTA = Cooperative Thread Array, CUDA’s formal name for a thread block.
•  bM, bN, bK are static integers, and make_shape combines them into one shape: (128, 128, 64).
(The variable is named cta_tiler)
•  Elements, not threads: 128, 128 and 64 count matrix elements.
•  The object cta_tiler like a cookie cutter. A shape that’s used to cut a bigger tensor into
tiles is called a tiler in CuTe. (M, N, K) isn’t one 3-D matrix. It describes three 2-D
matrices, and each uses only two of the three sizes. The same tiler is applied to each, with a
Step that picks which two modes to use (kernel, L72–74):
</pre>
|Line|	Matrix|	Step|	Modes used	|Tile cut out|
|:---|:-------|:----|:------------|:-----------|
|L72	|A (M × K)	|<_1, X, _1>	|M, K|	128 × 64|
|L73	|B (N × K)	|<X, _1, _1>	|N, K	|128 × 64|
|L74	|C (M × N)	|<_1, _1, X>	|M, N	|128 × 128|
<pre>  
X means “skip this mode”. So (128, 128, 64) is three tile sizes packed into one object, 
not one chunk of 128·128·64 elements.
only A and B go to shared memory, and C never does
• The A and B tiles (128 × 64 each) are copied into shared memory with cp.async, one k-tile per
loop iteration.
• The C tile (128 × 128) is never loaded into shared memory. Its values are accumulated in
registers (tCrC), and at the end they’re written directly from registers to global memory
(axpby, L257).
L379 auto cta_tiler = make_shape(bM, bN, bK): only describes; the cutting happens in the kernel
At L379, cta_tiler is just a description, three static sizes with no data. It’s passed to the
kernel (L434), and the actual cutting happens there in local_tile (L72–74). For A, the 
coordinate cta_coord = (blockIdx.x, blockIdx.y, _) (L71) picks which tile along M (blockIdx.x),
and the _ keeps all tiles along K. So gA has shape (128, 64, k): this CTA’s 128-row strip of A,
cut into k = 4096/64 = 64 pieces, one per main-loop iteration.
Summary:
cta_tiler is a static shape (128, 128, 64), used like a cookie cutter. In the kernel, local_tile
uses it to cut a 128 × 64 tile of A and of B (for each k-step) and one 128 × 128 tile of C for
each CTA. The A and B tiles are loaded into shared memory one k-step at a time; the C tile is
accumulated in registers and written directly to global memory. All sizes count elements, not
threads.
• The atom is 8 × 64, not 128 × 64. For half_t it’s 8 rows × 64 elements, each row 64 × 2 B =
128 bytes, so the atom is 1024 bytes. tile_to_shape then repeats it to fill the target shape
(Step 3 below).
• K is the contiguous dimension: In the atom (8,64):(64,1), K has stride 1. Viewed as an M × K
tile, that’s row-major (K-major, as discussed for dA).
• bM, bK, bP are static: so the whole resulting layout is a compile-time object. The kernel 
even requires that, at L49: static_assert(is_static<ASmemLayout>::value).
  
Line 383: auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
On the host, sA is a layout not a matrix: a mapping from coordinates (m, k, stage) to offsets.
It holds no memory and no data. It’s passed to the kernel as sA_layout (L37). Inside the kernel,
the name sA is reused for a different thing: a tensor = shared-memory pointer + this layout
(L80). Same name, two different objects.
L383 builds its own shape from bM, bK and bP: make_shape(bM, bK, bP) = (128, 64, 3). It doesn’t
use cta_tiler, which has no bP at all. The values only agree because both use the same bM and
bK, and the kernel checks that they do (L52, L54).

bP = 3 (L380) is the number of pipeline stages: three 128 × 64 slots for A in shared memory, 
so loads for future k-tiles can be in flight while one is being used. It’s the third mode of the
layout.

The layout fixes the size and arrangement at compile time, but no shared memory is set aside 
at L383. The actual reservation happens at launch:

• the size comes from SharedStorage (L413), via cosize_v of this layout (L24): 3 × 8192 halfs = 
48 KB for A;
• smem_size is passed as dynamic shared memory at launch (L418), after cudaFuncSetAttribute 
(L427–430).
it’s a software description of a hardware format

GMMA::Layout_K_SW128_Atom<TA> isn’t a hardware configuration. It’s a CuTe type that describes,
in software, one of the few shared-memory formats the Hopper wgmma instruction can read 
(K-major, 128-byte swizzle). Both sides must agree on it: CuTe applies the swizzle when the copy
writes shared memory, and the hardware applies the same swizzle when wgmma reads it.

About Blackwell: this GMMA:: atom is for Hopper (SM90). Datacenter Blackwell (SM100) uses its
own MMA instruction with its own atom types in CUTLASS (as far as I know, similar swizzle modes
under a different namespace). My RTX Blackwell SM120 GPUs have neither wgmma nor that
instruction, so these atoms don’t apply there.

• Contiguity: before the swizzle, each 128-byte row is fully contiguous along K. After the
swizzle, the row’s eight 16-byte chunks are permuted (physical chunk = logical chunk XOR (row
mod 8), as in the swizzle table from earlier). So elements are physically adjacent only within
each 16-byte chunk (8 halfs). That’s exactly why the copy moves 16 bytes per cp.async: one chunk
always stays intact.
Bank conflicts: the 128-byte swizzle is designed to spread the 8 rows of an atom across all
shared-memory banks. Whether this kernel is actually conflict-free can only be measured: on an
H100, ncu’s shared-memory bank-conflict counters would show it. Until then, say “designed to
avoid bank conflicts”.
Summary the whole line in one sentence
tile_to_shape takes the Hopper K-major 128-byte-swizzle atom (8 × 64 halfs = 1 KB) and repeats
it 16× along M, 1× along K and 3× along the pipeline mode, producing the static layout (128, 64,
3): three swizzled 128 × 64 stages for A in shared memory. The memory itself is reserved at
launch, as dynamic shared memory.

Resulting type, from my build log:
swizzle: Sw<3,4,3>

shape : ( (8,16)    , (64,1)    , (1,3)     )
stride: ( (64,512)  , (1,0)     , (0,8192)  )
mode  :   M           K           PIPE
Line 383: auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
The atom is 8 × 64. See below:
The atom is defined in bits, and the 8 × 64 comes from converting it to half (16 bits). 
I checked this in CuTe’s current cute/atom/mma_traits_sm90_gmma.hpp on main. It’s the same line
84 we looked at before.
using Layout_K_SW128_Atom_Bits =
  ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag,
                 Layout<Shape<_8,_1024>, Stride<_1024,_1>>>;
The plain layout part is (8, 1024) : (1024, 1):
• 8 rows
• 1024 bits per row = 128 bytes, contiguous (stride 1 along the row)
Step 2: convert to your element type (line 104)
template <class Type>
using Layout_K_SW128_Atom =
  decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW128_Atom_Bits{}));
    
upcast<N> regroups every N bits into one element. Here TA = half_t, so N = sizeof_bits<half_t> =
16:
row:    1024 bits / 16 = 64 halfs
rows:   8 (unchanged)
→ (8, 64) : (64, 1)
That’s the 8 × 64: 8 rows, each 64 halfs = 128 bytes, contiguous along K. That’s why it’s 
called K-major.
The same atom gives different element counts for other types, but always 128 bytes per row:
</pre>
|Type|	Bits|	Atom|
|:---|:-----|:----|
|half_t (this kernel)|	16|	8 × 64|
|FP8	|8	|8 × 128|
|float/TF32|	32|	8 × 32|
<pre>
Why 128 bytes and 8 rows
128 bytes per row: this is the “SW128” in the name, the 128-byte swizzle mode, the widest of 
the swizzle modes that wgmma descriptors support (the neighbouring lines 81–83 define the INTER,
SW32 and SW64 versions: 128, 256 and 512 bits per row). With bK = 64 halfs, one row of the A
tile is exactly 128 bytes, so one atom row spans the full bK.

8 rows: Swizzle<3,4,3> XORs 3 bits of the row number into the chunk bits. 3 bits means 2³ = 8
different patterns, so the pattern repeats every 8 rows:
8 rows × 128 bytes = 1,024 bytes = one full swizzle pattern
That’s the 1,024 bytes we found before, and the reason the buffers must be 1,024-aligned.
How tile_to_shape uses it (L383)
atom 8 × 64 → repeat 128 / 8 = 16 times along M → 128 × 64 (one stage)
                                                  → 3 stages: (128, 64, 3)

Along K, 64 / 64 = 1, so no repetition is needed.

Summary
</pre>
|Question|	Answer	|Source|
|:------|:---------|:------|
|Where is 8 × 64 defined?|	as (8, 1024) bits, line 84|	mma_traits_sm90_gmma.hpp|
|Why 64?|	1024 bits/16 bits per half = 64	|upcast, line 104|
|Why 128 B per row?|	the 128-byte swizzle mode (SW128)	|atom name, lines 81–84|
|Why 8 rows?|	Swizzle<3,…> uses 3 row bits → 8 patterns → 1,024 B|	Swizzle<3,4,3>|
<pre>
The 8 × 64 defined in cute/atom/mma_traits_sm90_gmma.hpp because I use
GMMA::Layout_K_SW128_Atom<TA>{}.
By writing GMMA::Layout_K_SW128_Atom<TA>, I choose the alias defined in that file (line 104).
That alias is built from the bit-level atom on line 84.
</pre>
|I write|	Line|	Shape for half_t|
|:------|:----|:----------------|
|GMMA::Layout_K_SW128_Atom<TA> (gemm_tn, L383)|	84 → 104|	8 × 64 (K-major)|
|GMMA::Layout_K_SW64_Atom<TA>	|83 → 102	|8 × 32|
|GMMA::Layout_MN_SW128_Atom<TA> (used in gemm_nt)|	MN version|	64 × 8 (MN-majo)|
<pre>

Two precisions:

1. The file defines it in bits. The 8 × 64 comes from your <TA>. Line 84 says 8 × 1024 bits. The 8 × 64 only appears because TA = half_t (16 bits), and upcast<16> turns 1024 bits into 64 elements. With a different TA, the same name gives a different shape (8 × 128 for FP8, for example).

2. A different atom name gives a different definition from the same file. For example:

  
Step 1: the definition, in bits (line 84)
The same as a table:
</pre>
|Mode|	Shape|	Stride|	Meaning|
|:---|:------|:-------|:-------|
|M	(8,16)	|(64,512)|	8 rows within an atom (64 elements apart) × 16 atoms<br> (512 elements apart) = 128 rows|
|K	|(64,1)	|(1,0)	|64 contiguous elements × 1 atom <br>(size 1, so its stride doesn’t matter) = 64|
|PIPE	|(1,3)|	(0,8192)|	size-1 placeholder × 3 stages, 8192 elements apart|
<pre>
GMMA:: atom is for Hopper (SM90), but “Blackwell” covers two different GPU families with
different instructions, and “MMA” is a general term, not a Blackwell-specific one.
</pre>
|GPU	|Compute capability|	Tensor-core instruction|	CuTe namespace / atoms|
|:----|:-----------------|:------------------------|:-----------------------|
|Hopper (H100)|	SM90|	wgmma.mma_async	|GMMA:: (this tutorial)|
|Datacenter Blackwell (B200)|	SM100|	tcgen05.mma|	UMMA::|
|Consumer Blackwell (my RTX 5080)|	SM120|	mma.sync (warp-level)|	ordinary warp-level MMA<br> atoms, the SM80_…-style ones you used in sgemm_sm80.cu|
<pre>
GMMA:: = Hopper, SM90. 
UMMA:: (not MMA::) = datacenter Blackwell, SM100, with UMMA::ScaleOut::Zeroin the Blackwell 
tutorial concepts.
My SM120 GPUs have neither wgmma nor tcgen05. They use the warp-level mma.sync path.

“MMA” by itself just means matrix multiply-accumulate. CuTe uses it generically for all
architectures: MMA_Atom, TiledMMA, make_tiled_mma (L394) work the same way whether the atom
inside is GMMA, UMMA or mma.sync. Only the atom changes.

To see the namespaces in your checkout:
grep -rln "namespace GMMA\|namespace UMMA" ../cutlass/include/cute/ 

• GMMA:: = Hopper (SM90)
• Datacenter Blackwell (SM100) uses UMMA::. Consumer Blackwell, like my RTX 5080 (SM120), 
uses neither: it uses the ordinary warp-level mma.sync atoms.
• “MMA” alone is the general term, used for every architecture (MMA_Atom, TiledMMA).


sA indicates it is shared memory Matrix A. We use cta_tile object to make shape in shared memory. bM, bK, and bP is integer, static, evaluate in compile-time, bM, bK, bP dimension carve out physical bound of shared memory in compile-time . GMMA::Layout_K_SW128_Atom<TA> is a configuration of specification hardware  for Hopper/Blackwell shared memory.
128 byte hardware swizzle, tell the CuTe to apply 128 byte swizzle. We get bank conflict free.
K is a contiguous dimension. The elements alone the K-dimension, sit right next to each other in physical memory, moving contiguous across column which means matrix A is store a row-major format in shared memory.    
The Layout_K_SW128_Atom is not 128x64, it is an atom 8x64, a tiny atom.
</pre>
