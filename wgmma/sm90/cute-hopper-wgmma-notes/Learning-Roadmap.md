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

Bank conflicts: the 128-byte swizzle is designed to spread the 8 rows of an atom across all
shared-memory banks. Whether this kernel is actually conflict-free can only be measured: on an
H100, ncu’s shared-memory bank-conflict counters would show it. Until then, say “designed to
avoid bank conflicts”
Conclusion: until I have ncu data from a real H100 run, “designed to avoid bank conflicts” is
the accurate statement. “Bank-conflict-free” is a claim about measured behavior.

I can predict before you measure. There are two separate shared-memory accesses here, 
and they differ in how well they can be predicted:
I cannot say it is bank conflict until I look up the NCU data, then I can know whether it 
is bank conflict free or not.
</pre>
|Access|	Who does it|	Predict by hand|
|:-----|:------------|:----------------|
|Writes by cp.async (L222–223)|	each thread writes its own 16 bytes|Yes. You know which thread<BR> writes which element (the copy TV layout)<br> and where it lands (layout + swizzle),<br> so you can work out each thread’s<br> bank and check for collisions|
|Reads by wgmma|the tensor-core hardware,<br> through the descriptor|	Not really. The hardware’s<br> internal access order isn’t documented in<br> enough detail. The swizzle mode is<br> chosen to match what it expects,<br> but only measurement confirms the result|
<pre>
So the full method is:
1. Predict: compute the banks for the cp.async writes of one warp, by hand or with a small host
program like print_layouts.cu.
2. Measure: on the H100, profile with ncu and check the shared-memory bank-conflict counts. 
The Memory Workload Analysis section reports them per access type. Exact metric names vary
between ncu versions, so check what the version lists rather than relying on a name developer
give from memory.
3. Compare: if prediction and measurement agree, developer understand the kernel; if they don’t,
that’s worth investigating, and worth writing about.
SW128 is intended to avoid bank conflicts, and only measurement confirms it for a given kernel.
The hardware doesn’t specify GMMA::Layout_K_SW128_Atom<TA>. The roles split three ways:
</pre>
|Who	|What|
|:----|:---|
|Hopper hardware|	supports a few shared-memory formats<br> for wgmma: no swizzle, 32-, 64- and 128-byte<br> swizzle, K-major or MN-major|
|CuTe|	describes each supported format as<br> a layout atom; GMMA::Layout_K_SW128_Atom<br> is the one for K-major,<br> 128-byte swizzle|
|The programmer<br> (here, the tutorial’s authors at L383)|	chooses which atom to use|
<pre>
Summary:
Hopper’s wgmma supports a few swizzled shared-memory formats; the tutorial chooses the K-major
128-byte one, described in CuTe by GMMA::Layout_K_SW128_Atom<TA>. SW128 is intended to avoid
bank conflicts, but that isn’t guaranteed; it has to be measured with ncu.
  
This is the same verification-first pattern as my C7510 investigation: a reasoned prediction,
then evidence from the tools. 

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

1. The file defines it in bits. The 8 × 64 comes from your <TA>. Line 84 says 8 × 1024 bits. 
The 8 × 64 only appears because TA = half_t (16 bits), and upcast<16> turns 1024 bits into 64
elements. With a different TA, the same name gives a different shape (8 × 128 for FP8, for
example).

2. A different atom name gives a different definition from the same file. For example:
The last row matters because the default run is NT, which uses the MN version. Its bits are 
laid out the other way around: 1024 bits along M (or N) by 8 along K. For half that’s 64 × 8,
with M contiguous. I inferred that from lines 75–77, where the INTER/SW32/SW64 MN atoms are 
(128 / 256 / 512, 8). Check the SW128 MN line in your copy; it should read Shape<_1024,_8>,
Stride<_1,_1024>.

GMMA:: is the namespace SM90::GMMA. That’s why it’s in the Hopper (sm90) traits file.

Is SW128 a better choice?
Usually yes, when the tile is wide enough to use it. SW128 isn’t better unconditionally. 
It’s the widest pattern, and it only fits when the tile’s contiguous dimension is at least 
128 bytes.

What the number means
The number is how many bytes one swizzle row covers, and therefore how many 16-byte chunks 
the XOR can shuffle:
</pre>
|Mode|	Row width|	16 B chunks shuffled|	Swizzle<B,4,3>|	Pattern repeats every|
|:---|:----------|:---------------------|:--------------|:---------------------|
|SW32	|32 B	|2|	Swizzle<1,4,3>|	256 B|
|SW64|	64 B	|4|	Swizzle<2,4,3>|	512 B|
|SW128|	128 B|	8|	Swizzle<3,4,3>|	1,024 B|
<prd>
The Swizzle<1/2/3,4,3> values are from lines 81–84 of the traits file. The “repeats every” 
column is 2^(B+4+3) bytes, the same rule we used for SW128.
Why SW128 is usually the best

Shared memory has 32 banks of 4 bytes, so 128 bytes covers every bank once. When 8 rows are 
read at the same column:

• SW128: the XOR moves each of the 8 rows’ chunks to a different position among 8. The 8 reads
spread across all 32 banks, with no conflict.
• SW64 / SW32: there are only 4 or 2 positions to spread across, so some rows land on the same
banks. That can cause conflicts, depending on the access pattern.

So a wider swizzle spreads accesses across more banks. Whether a particular wgmma access pattern
actually conflicts with SW64 is something I haven’t verified. That’s an ncu measurement on the
H100.

When SW128 doesn’t fit
The tile must be at least 128 B wide in its contiguous dimension. For K-major half, bK must be a
multiple of 64 (64 × 2 B = 128 B). This kernel has bK = 64, so SW128 fits exactly. With bK = 32 
(64 B) you’d need SW64, and with bK = 16 you’d need SW32.
Stricter alignment: the buffer must be 1,024-byte aligned for SW128, versus 512 or 256 for the
smaller modes.
TMA (next file): I believe the TMA box’s inner dimension can’t exceed the swizzle width, so the 
swizzle mode and the TMA box size must agree. Check that when you get to wgmma_tma_sm90.cu.
How CUTLASS chooses

CUTLASS’s Hopper GEMM builders choose automatically, as far as I remember: they pick the 
widest swizzle atom that evenly divides the tile’s contiguous dimension (SW128 if it fits,
otherwise SW64, then SW32, then INTER). The helper is called ss_smem_selector. Verify with:
grep -rn "ss_smem_selector" -A25 ../cutlass/include/cutlass/gemm/collective/builders/ | head -60

That preference order is the practical answer: SW128 is the default choice when the tile allows 
it.

Summary
</pre>
|Question|	Answer|
|:-------|:--------|
|Is SW128 better?|	Usually: it spreads 8 rows across all 32 banks|
|Always?	|No: it needs a ≥ 128 B contiguous tile dimension and 1,024 B alignment|
|This kernel|	bK = 64 halfs = 128 B → SW128 fits exactly|
|How production code chooses|	the widest swizzle that fits the tile<br> (ss_smem_selector, to verify)|
|How to confirm the benefit|	ncu shared-memory bank-conflict metrics on the H100|
</pre>

 auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
shape : ( (8,16) , (64,1) , (1,3) ) 
stride: ( (64,512) , (1,0) , (0,8192) ) 
mode :    M               K           PIPE
Above is not for atom piece, it is the whole shared-memory layout sA, the result after
tile_to_shape, covering all three
128 × 64 stages. The atom is much smaller:
atom:   shape (8, 64) : stride (64, 1)        ← 8 rows × 64 elements = 1 KB
The atom is still visible inside the full layout: in each mode, the first number of the pair
comes from the atom, and the second is how many times the atom is repeated:
</pre>
|Mode	|Full shape|	Full stride|	From the atom|	Repeats of the atom|
|:----|:---------|:------------|:--------------|:---------------------|
|M	|(8, 16)|	(64, 512)|	8 : 64 (atom’s rows)|	16 : 512 (16 atoms,<br> each 512 elements later)|
|K	|(64, 1)|	(1, 0)|	64 : 1 (atom’s columns)|	1 : 0 (only one atom along K)|
|PIPE	|(1, 3)|	(0, 8192)|	1 : 0 (the atom has no PIPE mode;<br> added as size 1)|	3 : 8192 (3 stages)|
<pre>
Reading only the first number of each pair gives back the atom: (8, 64) : (64, 1). 
The second numbers describe how tile_to_shape stacked copies of it to reach (128, 64, 3).

The atom is one level smaller still.
There are three levels, from smallest to largest:
</pre>
|Level|	Shape|	Stride|	Size|
|:----|:-----|:-------|:-----|
|Atom	|(8, 64)|	(64, 1)|	8 × 64 = 512 elements (1 KB)|
|One stage|	((8,16), (64,1))	|((64,512), (1,0))|	128 × 64 = 8192 elements (16 KB)|
|Full sA	|((8,16), (64,1), (1,3))	|((64,512), (1,0), (0,8192))|	3 stages = 24,576 elements<br> (48 KB)|
<pre>
The atom, aligned by mode:
shape : ( 8  , 64 )
stride: ( 64 , 1  )
mode  :   M    K

M: 8 rows, each 64 elements after the previous one.
K: 64 elements, contiguous.

one stage:   shape ( (8,16) , (64,1) )      → 8×16 = 128 rows,  64×1 = 64 columns
                       M        K

atom:        shape (   8    ,   64   )      → 8 rows, 64 columns
                       M        K

The atom has one number per mode: 8 for M and 64 for K. Its full description is:

atom:   shape ( 8  , 64 )
        stride( 64 , 1  )
                M    K

The pairs (8,16) and (64,1) only appear after tile_to_shape has repeated the atom: the first
number of each pair is the atom’s size (8, 64), and the second is the number of 
repeats (16, 1).
One atom 8x64 is a tiny chunk of 128 x 64.
One atom (8 × 64) is a small piece of one 128 × 64 stage. It takes 16 atoms, stacked along M, 
to make one stage:
128 × 64 stage
┌──────────── 64 (K) ────────────┐
│ atom 0   rows   0 –   7         │
│ atom 1   rows   8 –  15         │
│ atom 2   rows  16 –  23         │
│   …                             │   16 atoms × 8 rows = 128 rows (M)
│ atom 15  rows 120 – 127         │
└─────────────────────────────────┘

• Each atom is 8 × 64 halfs = 1 KB, with its own 128-byte swizzle applied inside.
• 16 atoms make one stage: 16 KB.
• sA has 3 stages: 48 KB in total.




  
Step 1: the definition, in bits (line 84)
The same as a table:
</pre>
|Mode|	Shape|	Stride|	Meaning|
|:---|:------|:-------|:-------|
|M	|(8,16)	|(64,512)|	8 rows within an atom (64 elements apart) × 16 atoms<br> (512 elements apart) = 128 rows|
|K	|(64,1)	|(1,0)	|64 contiguous elements × 1 atom <br>(size 1, so its stride doesn’t matter) = 64|
|PIPE	|(1,3)|	(0,8192)|	size-1 placeholder × 3 stages, 8192 elements apart|
<pre>
Shape and stride pair up by position: the first group of the shape goes with the first group 
of the stride, and so on, both at the outer level (M, K, PIPE) and inside each group.
  
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


Undersanding the following statement.
auto sA = tile_to_shape(
    GMMA::Layout_K_SW128_Atom<TA>{},
    make_shape(bM, bK, bP)
);
tile_to_shape does not allocate memory or load matrix data. It constructs a CuTe layout that
describes how shared-memory storage should be organized.
gemm_tn: bM = 128, bK = 64, bP = 3 → shape (128, 64, 3)
bM, bK, and bP
In the example, these values are:
auto bM = Int<128>{};
auto bK = Int<64>{};
auto bP = Int<3>{};  // Pipeline stages
Then: make_shape(bM, bK, bP)
constructs a three-dimensional shape:
(128,64,3)

P 3 pipeline stages
Three storage slots allow the kernel to pipeline loading and computation.
Conceptually, the layout describes three A tiles:
sA(:,:,0), sA(:,:,1), sA(:,:,2)
Each stage can hold a tile of (128 x 64) elements. These are three storage regions, not three
different matrix dimensions in the mathematical GEMM problem.

TA is FP16, the storage required for all three tiles is:
128 x 64 x 3 x 2 = 49,152 bytes
That is 48 KB for operand A, excluding other shared-memory storage.

Layout_K_SW128_Atom
This is a predefined shared-memory layout atom.
- K means the layout is K-major: the K dimension is contiguous in memory.
- SW128 means it uses a 128-byte shared-memory swizzle.
- Atom means a small, reusable layout pattern that can be tiled to construct a larger layout.

GMMA::Layout_K_SW128_Atom<TA>{}
The {} constructs an instance of the layout atom. Because the type is known at compile time, 
the compiler can resolve much of the layout structure during compilation.
fully static here: every value is an Int<>

What layout does the atom describe:
For the FP16 example with a 128-byte swizzle, the K-major layout atom can be understood as 
the two-dimensional layout: (8,64):(64,1)
This notation means:
- Shape (8,64): 8 elements in the M-like direction and 64 elements in K.
- Stride (64,1): advancing one position along the first dimension advances 64 elements; advancing one position along K advances one element.
The atom covers 8 rows and 64 columns. For FP16, its contiguous K extent occupies:
64 x 2=128 bytes
That matches the 128-byte swizzle mode.

The first dimension requires:
128/8=16
So the atom repeats 16 times along M. It already spans all 64 elements along K, so it does not
need to repeat along K.
Finally, it extends the result over the P dimension of size 3.

The text says CuTe prints (_128,_64,_3):(_64,_1,_8192). Your actual output was hierarchical. 
Per mode it was:

((_8,_16), (_64,_1), _3) : ((_64,_512), (_1,_0), _8192)
For a logical coordinate (m,k,p), the unswizzled element offset is:
offset(m,k,p) = 64m+k+8192p
For example:
-(0,0,0) has an unswizzled offset of 0.
-(0,1,0) has an unswizzled offset of 1.
-(1,0,0) has an unswizzled offset of 64.
-(0,0,1) has an unswizzled offset of 8192.
These offsets are in elements, not bytes. For FP16, multiply by 2 to obtain the corresponding
byte offset before considering the swizzle transformation.

The swizzle modifies the physical address mapping, so these expressions describe the underlying 
stride structure, not the final swizzled address for every element.

Line: 387 TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},  Layout<Shape<_16,_8>,Stride<_8,_1>>{},   Layout<Shape< _1,_8>>{});  

Create a tile copy object copyA base on the layout (shape and stride). It is not copied over
yet. this is an object.
SM80_CP_ASYNC, SM80 is Ampere architectural. The compiler uses Ampere architectural asynchronous
copy from global memory to shared memory without private registers.
uint128_t: each thread move 128 bit.
TA is a type, here we use half_t, FP16.
Share<_16,_8>, 16x8 threads. They are static.
Layout<Shape< _1,_8>>{}): value layout. Each thread contains 8 elements alone row.

• An object, not a copy yet: copyA only describes how to copy. The actual copies happen inside
the kernel, when copy(copy_a, …) is called (L193, L222).
• SM80 = Ampere: cp.async was introduced with Ampere (SM80) and is still available on Hopper,
which is why a Hopper kernel can use an “SM80” atom.
• Global → shared without going through registers: right. That’s the key property of cp.async:
the data never passes through the thread’s registers. In SASS it’s the LDGSTS instruction.
CACHEALWAYS selects the .ca variant, which caches in L1 and L2.
• uint128_t: 128 bits per thread, per instruction: each cp.async moves 16 bytes. (For a whole
128 × 64 A tile, each thread issues 8 of them.)
• TA = half_t, FP16. Its role here: it tells CuTe the element size, so it knows 128 bits = 8
halfs.
• Value layout Shape<_1,_8>: 8 elements along a row: right. Each thread owns 1 element 
along M × 8 consecutive elements along K: one row, 8 contiguous halfs, exactly one 16-byte chunk.

Stride<_8,_1> is row-major, not column-major
The thread layout (16,8):(8,1) maps a thread’s grid position (tm, tk) to its index:
t = 8·tm + 1·tk
The second mode (tk, along K) has stride 1: consecutive thread numbers go along a row. 
That’s row-major (CuTe’s LayoutRight). The source comment at L388 says the same: “16x8 k-major”.
</pre>
|Thread|	(tm, tk)|
|:-----|:---------|
|0, 1, 2, …, 7	|(0,0), (0,1), …, (0,7) → across row 0|
|8|	(1,0) → start of row 1|
<pre>
Column-major would be the default stride for this shape, (1,16): then thread 1 would sit below
thread 0, in row 1. That’s what the NT version uses (L296, no stride given).

_16, _8 are static, not const
_16 is CuTe’s short name for Int<16>: a compile-time integer, like bM. So “static” or 
“compile-time” is the right word. const in C++ means something different (a variable that 
can’t be modified at runtime).

Shape<_16,_8> = 16 × 8 = 128 threads, the whole warpgroup. It must match the MMA’s thread count,
which the kernel checks at L46: size(copy_a) == size(mma).

What it all produces

Combining the three pieces gives what you saw in the build log: one copy step covers a 16 × 64
piece of A:
thread t copies row t/8, K elements 8·(t mod 8) … +7     (one 16-byte cp.async)
128 threads × 8 halfs = 1024 elements = 16 rows × 64

A full 128 × 64 tile of A needs 128 / 16 = 8 copy steps, so each thread issues 8 cp.async per 
k-tile for A (and 8 more for B).

Line 394: TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{});
It create an object tiled_mma for math layout. It uses Hopper engine SM90. the m,n,k is
64x64x16. The type is half_t, FP16, half precision. It is shared and shared memory. Matrix A and
B are from shared memory. K-dimension, k-major.

1. It describes; it doesn’t compute yet. Like copyA, tiled_mma is only a description. The actual
MMAs are issued in the kernel, at cute::gemm(mma, …) (L240).
2. “Hopper engine”: more precisely, the wgmma.mma_async instruction on Hopper’s tensor cores.
It’s issued by a warpgroup (4 warps = 128 threads) together, not by a single warp. That’s why
size(tiled_mma) = 128, and why it sets the block size at L414 (dim3 dimBlock(size(tiled_mma))).
3. 64 × 64 × 16 is one instruction, not the whole tile. One wgmma computes a 64 × 64 piece of C
with K = 16. The CTA tile (128 × 128 × 64) needs:
(128/64) × (128/64) × (64/16) = 2 × 2 × 4 = 16 wgmma instructions per k-tile
4. F16F16F16 means three types, including the accumulator. The order is D, A, B:
</pre>
|Position|	Matrix|	Type|
|:-------|:-------|:----|
|1st F16|	D (accumulator/result)|	FP16|
|2nd F16|	A|	FP16|
|3rd F16|	B	FP16|
<pre>
So this GEMM also accumulates in FP16, not FP32. That’s why the inputs are kept to ±1 (I
discussed this with rand()): sums of 4096 products must fit in FP16.
5. The two Major::K arguments: first for A, second for B. Both say “K is the contiguous
dimension in shared memory”. They must match the shared-memory layouts: L383–384 use
Layout_K_SW128_Atom for both A and B. If the layout and the Major disagreed, the hardware would
read the data wrongly. The NT version uses Major::MN at L302, together with Layout_MN_SW128_Atom
at L291–292.
SS: A and B both from shared memory: The other option, RS, takes A from registers.
The build log shows the full type, including two default arguments not written in the source:
MMA_64x64x16_F16F16F16_SS<Major::K (0), Major::K (0), ScaleIn::One (1), ScaleIn::One (1)>
ScaleIn::One means A and B are used as-is; the alternative, Neg, negates them.

Line 413: int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  
1. decltype(sA) gives the layout’s type, not its shape at runtime.
sA (L383) is a fully static layout, so everything about it (shape, stride, swizzle) is encoded
in its type. decltype(sA) takes that type, and it’s passed to SharedStorage as the template
parameter SmemLayoutA (L20). decltype itself doesn’t check or calculate anything.
2. The element count is computed inside the struct, by cosize_v (L24–25):
alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;

cosize_v<Layout> is the number of elements the layout spans in memory: its largest offset + 1.
(size would be the number of coordinates. For this compact layout they’re equal, but cosize is
the one that matters for allocating memory.)

3. The pipeline stages are already in the layout. sA was built with make_shape(bM, bK, bP) =
(128, 64, 3) (L383), so its PIPE mode (1,3):(0,8192) already covers 3 stages:
cosize_v<SmemLayoutA> = 3 stages × 8192 = 24,576 halfs
SharedStorage does no multiplication of its own. It just declares arrays as large as the 
layouts need.

4. sizeof gives the total in bytes, at compile time:
</pre>
|Member|	Elements|	Bytes|
|:-----|:---------|:-----|
|A	|24,576 halfs|	49,152|
|B	|24,576 halfs|	49,152|
|Total||		98,304 = 96 KB| 
<pre>
alignas(128) could add padding between A and B, but 49,152 is a multiple of 128, so none is
needed here.
5. No SharedStorage object is created on the host. Only its type is used, to get its size. 
The struct is used for real inside the kernel, where it’s laid over the dynamic shared memory
(L77–79: reinterpret_cast<SharedStorage*>(shared_memory)).

6. smem_size is an int: 32 bits on your platform. sizeof returns a size_t (64 bits), and
int(...) converts it. Its value is 98,304, which is then passed as the dynamic shared-memory
size at launch (L418) and to cudaFuncSetAttribute (L427–430).

Summary:
decltype(sA) and decltype(sB) pass the static shared-memory layouts as types to SharedStorage.
Inside it, cosize_v gives each array’s element count; since the layouts already include the 3
pipeline stages, A and B each get 24,576 halfs (three 128 × 64 tiles). sizeof gives the struct’s
size at compile time: 98,304 bytes, stored in the int smem_size.

template <class ElementA,
          class ElementB,
          class SmemLayoutA, 
          class SmemLayoutB> 
struct SharedStorage
{
Line 24:  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
};

• ArrayEngine has two template arguments: the element type (ElementA = half_t, FP16) and the
number of elements (cosize_v<SmemLayoutA>).
• cosize_v is evaluated at compile time: The layout is static, so the count is a compile-time
constant (24,576).
• ArrayEngine has no rows, columns or swizzle: It’s plain storage, 
like std::array<half_t, 24576>: one flat block of elements. Rows, columns and swizzle come
from the layout, which is attached to this storage inside the kernel 
(L80: make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{})).
  
The pipeline stages are a separate mode, not part of K
The layout has three modes: M, K and PIPE. The 3 stages are the third mode, (1,3):(0,8192), 
not stacked along K:
one stage  = 128 (M) × 64 (K) = 8,192 elements
3 stages   = 3 × 8,192        = 24,576 elements   ← cosize_v<SmemLayoutA>

Each stage is a complete 128 × 64 tile, 8,192 elements after the previous one.
The swizzle doesn’t change the count. It only permutes 16-byte chunks within each 1 KB atom, 
so the largest offset, and therefore cosize_v, stays the same.
cosize_v is a count, not an index
cosize_v gives one number: how many elements the array must hold (24,576). The index is what
the layout computes later, for each coordinate: (m, k, stage) → a position in 0 … 24,575 inside
this flat array.
_v stands for “value”

It’s the standard C++ naming convention (since C++17) for a variable template that gives a
value: std::is_same_v<T,U> means std::is_same<T,U>::value. In the same way, cosize_v<L> is 
the compile-time value of cosize for layout L.

misalignment usually gives wrong results, not a crash
alignas(128) makes each array start at an address that is a multiple of 128 bytes, relative to
the start of the struct. What happens if alignment were wrong depends on which requirement is
broken:
</pre>
|Requirement|	If broken|
|:----------|:---------|
|16-byte alignment for each<br> 16-byte cp.async|	runtime error (“misaligned address”):<br> the kernel faults.<br> This is the “crash” case|
|1024-byte alignment for the<br> 128-byte swizzle pattern<br> (as discussed for SharedStorage)|	no error at all: CuTe’s swizzled writes<br> and the hardware’s swizzled wgmma<br> reads would disagree, and the GEMM would<br> silently produce wrong numbers|
<pre>
The second case is the dangerous one, because nothing tells you. That’s another reason the
program should check its result against a reference, which, as noted in Chunk 1, it currently
doesn’t.
As I found earlier, alignas(128) alone doesn’t guarantee 1024-byte alignment. It works here
because A starts at the beginning of the dynamic shared memory, and A’s size (49,152 bytes = 48
× 1024) puts B on a 1024-byte boundary too. 
The three conditions"
</pre>
|#	|Condition|	In this kernel|
|:--|:--------|:--------------|
|1|	the dynamic shared memory base<br> address is 1,024-aligned| assumed, not guaranteed by alignas(128)|
|2|	A is at offset 0 of SharedStorage	| A is the first member|
|3|	A’s size is a multiple of 1,024 B,<br> so B starts on a 1,024 boundary| 49,152 = 48 × 1,024|
<pre>
If any one fails, B (or A) may no longer start on a 1,024-byte boundary. Some ways it could
fail:
• Condition 2: a buffer placed before A in the struct, for example barriers. In
wgmma_tma_sm90.cu they come after A and B, at L61–62.
• Condition 3: a different tile size. bM = 96 gives 96 × 64 × 3 × 2 = 36,864 B = 36 × 1,024,
which is still fine. But a size that isn’t a multiple of 1,024 would break B’s alignment.
• Condition 1: static __shared__ variables in the same kernel. I believe dynamic shared memory
is placed after them, which would shift its base.
Is condition 1 guaranteed?
As far as I know, CUDA doesn’t document a 1,024-byte alignment guarantee for the start of
extern __shared__ memory. I’m not certain, so don’t put conclusion in the write-up as fact.

There’s evidence that it holds in practice: CUTLASS’s own code doesn’t ask for 1,024 either. 
I checked on main:
• wgmma_tma_sm90.cu, L59–60: alignas(128) for A and B, the same as your kernel
• blackwell/01_mma_sm100.cu, L123–124: alignas(128)
• the production Hopper mainloop (sm90_mma_tma_gmma_ss_warpspecialized.hpp, L155–157):
aligned_struct<128> with array_aligned, whose default alignment is 16 (array_aligned.hpp, L39)

So CUTLASS relies on the same assumption. That suggests the base is suitably aligned on these
GPUs, but it’s an inference, not a documented guarantee.
How to verify it
Print the shared-memory address of A and B, modulo 1,024:
if (thread0()) {
  printf("A %% 1024 = %u\n", (unsigned)(__cvta_generic_to_shared(smem.A.begin()) % 1024));
  printf("B %% 1024 = %u\n", (unsigned)(__cvta_generic_to_shared(smem.B.begin()) % 1024));
}
I can run this part locally on my SM120: write a small kernel with the same SharedStorage and
dynamic shared-memory size, without the wgmma. That tells me how your GPU places dynamic 
shared memory. The H100 could place it differently, so repeat the check there.
Summary:
</pre>
|Question|	Answer|
|:-------|:-------|
|Does dynamic shared memory alone<br> make it work?|	No: three conditions are<br> needed (base aligned, A first, A’s size<br> a multiple of 1,024)|
|Is the base 1,024-aligned?|	Not documented as far as I know;<br> CUTLASS relies on it in<br> practice|
|Evidence|	CUTLASS tutorials and the production<br> mainloop also use only 128-byte<br> (or 16-byte) alignment|
|How to confirm|	print __cvta_generic_to_shared(...) % 1024<br> for A and B: locally on SM120,<br> then on the H100|

<pre> 
Here is that line piece by piece, without the swizzle details.
alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
1. cute::ArrayEngine<T, N>: a fixed-size array
ArrayEngine<T, N> is CuTe’s wrapper around a plain array of N elements of type T. It behaves
like:
T A[N];
2. ElementA: the element type
It’s the template parameter of SharedStorage, and here it’s TA = half_t (FP16, 2 bytes).
3. cosize_v<SmemLayoutA>: how many elements the array needs
SmemLayoutA is the type of the sA layout built on the host (L383): shape (128, 64, 3).
cosize_v<…> is a compile-time number: how many elements of storage that layout reaches. 
The _v means “value”.
The layout covers 128 rows × 64 columns × 3 stages with no gaps, so:
cosize_v<SmemLayoutA> = 128 × 64 × 3 = 24,576 elements
The print_layouts output showed 24,576.
So the line is equivalent to:
alignas(128) half_t A[24576];      // 24,576 × 2 B = 49,152 B = 48 KB
On the swizzle: it rearranges where elements sit inside this array. It doesn’t need extra
space, so the size is still 24,576. That’s all the previous explanation meant.
4. alignas(128): where the array may start
The array’s start address must be a multiple of 128 bytes.
There’s one thing to remember from earlier: the 128-byte swizzle pattern repeats every 1,024
bytes, and I noted that the buffers need to be 1,024-aligned for CuTe’s swizzle and the
hardware’s to agree. alignas(128) alone only guarantees 128. Whether the A buffer actually
starts at a multiple of 1,024 depends on the base address of dynamic shared memory. That’s the
open hypothesis we listed, still to verify (for example, by printing the address of smem.A on
the H100).
5. A: the member name
It’s a member of the SharedStorage struct (L18–26). The kernel places that struct on the
dynamic shared memory:
extern __shared__ char shared_memory[];
SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
Then L80 wraps it as a tensor: make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{}).
That’s the pointer plus layout view.

Summary
</pre>
|Part|	Meaning|	Value here|
|:---|:--------|:-----------|
|ArrayEngine<T, N>|	fixed array of N elements of T|	like T A[N]|
|ElementA|	element type|	half_t, 2 B|
|cosize_v<SmemLayoutA>|	elements the layout needs	|128 × 64 × 3 = 24,576|
|total size|	N × 2 B|	49,152 B = 48 KB|
|alignas(128)|	start address multiple of 128 B|	the swizzle needs<br> 1,024 (open item)|
|used by|	sA = make_tensor(smem_ptr(smem.A.begin()),<br> layout)|	L80|

<pre>

</pre>



|Form	|Text|	Where it comes from|
|:----|:---|:--------------------|
|hierarchical	|((8,16),(64,1),3):((64,512),(1,0),8192)|	what tile_to_shape built and<br> print showed you|
|flat|	(128,64,3):(64,1,8192)	|the same layout after merging modes (coalesce)|
|read it as|	offset = 64m + k + 8192p	|shape gives ranges, stride gives steps|
<pre>
Line 66: Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); 
Line 67:  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); 
Line 68:  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); 
make_gemem_ptr(A), a pointer point to Matrix A.
  We use select<0,2> select<1,2>, and select<0,1> instead of make_shape(M,N), make_shape<N,K>, and make_shape<M,N>. It quarantee the K-dimenstion in Matrix A is indentically matched to the K-dimenstion in Matrix B. 
  dA, dB, and dC are strides, we use make_stride(..) before.
  mA, mB, and mC tensor object that required by GPU needs to read and write the 
  physical bytes of the Matrices.

make_gmem_ptr(A): a pointer tagged as global memory
A (L37) is already a plain pointer to matrix A in GPU global memory: d_A.data().get() from main
(L555). make_gmem_ptr(A) wraps it and tags it as global memory. The tag matters later: it tells
CuTe which memory space it’s dealing with, so it can pick valid copy operations (here, cp.async
from global to shared).
select<…>: correct idea; here’s exactly why
select<0,2>(shape_MNK) takes modes 0 and 2 of (M, N, K), giving (M, K):
</pre>
|Line|select|	Result|	Matrix|
|:---|:-----|:------|:------|
|L66|	<0,2>|	(M, K)	A|
|L67	|<1,2>	|(N, K)	|B|
|L68|	<0,1>|	(M, N)|	C|
<pre>
The benefit: all three shapes come from one source, shape_MNK, so the K of A and the K of B are
the same value by construction. The same holds for M (shared by A and C) and N (shared by B and
C).
There’s also a practical reason: inside the kernel, M, N and K don’t exist as separate
variables. The kernel received only shape_MNK (L36), so select is the natural way to get the
pieces.
  
dA, dB, dC: from make_stride on the host
They were created on the host (L371–373) and passed as kernel arguments. The kernel checks at
compile time (L57–59, congruent) that each stride has the same structure as its shape: two
modes for a 2-D shape.
mA, mB, mC: views, not data

This is the part to sharpen. A CuTe Tensor is just:
Tensor = pointer (engine) + layout (shape + stride)
• No memory is allocated and no data is copied. mA is a view of matrix A that already exists in
global memory.
• It covers the whole matrix (M × K), not a tile. Tiles come next, with local_tile (L72–74).
• It doesn’t “read or write bytes” by itself. It lets the code compute addresses from 
coordinates: mA(m, k) → A + m·dA[0] + k·dA[1].
• After compilation, the abstraction disappears: what’s left is a pointer and a few integers in
registers, plus the address arithmetic. The GPU only executes the resulting loads and stores.

Who reads and who writes: mA and mB are only read (as sources for the copies); mC is only
written, in the epilogue (L257).

Summary:
make_gmem_ptr tags the raw device pointer as global memory. select derives each matrix’s shape
from the single shape_MNK, so shared dimensions (K for A and B, M for A and C, N for B and C)
stay consistent by construction. Combined with the strides from the host, make_tensor builds 
mA, mB, mC: views (pointer + layout) of the full matrices in global memory, with no allocation
or copying. They’re used to compute addresses; the actual reads and writes happen later, in the
copies and the epilogue.

auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);             
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{}); 

make_coord(blockIdx.x, blockIdx.y, _)
Correction: blockIdx.x and blockIdx.y are tile numbers, not rows and columns. They say which
128 × 128 tile of C this CTA computes: 
</pre>
||Meaning	|Range (defaults)|
|:-|:-----|:---------------|
|blockIdx.x	|tile index along M|	0 … 39 (5120 / 128 = 40)|
|blockIdx.y	|tile index along N	|0 … 39|
<pre>
The grid is 40 × 40 (L416–417). For example, the CTA with blockIdx = (3, 5) computes C rows
3·128 … 3·128+127 = 384 … 511 and columns 5·128 … 5·128+127 = 640 … 767.  
_, keep the whole K mode: correct. _ (cute::Underscore) means “don’t pick one tile along K;
keep all of them”. That’s why gA gets an extra mode holding all the k-tiles of this CTA’s strip
of A.

local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{})
Correct: cta_tiler is 3-D (bM, bN, bK), but A has only M and K, so the Step says which modes
apply: _1 = use this mode, X = skip it. Your reading of each position is right: M used, N
skipped, K used.
One detail to add: the Step applies to both the tiler and the coordinate:
tiler: (bM, bN, bK) = (128, 128, 64)          coord: (blockIdx.x, blockIdx.y, _)
Step:  ( _1,  X,  _1)                          Step:  (    _1,        X,     _1)
  ⇒   (128,       64)                            ⇒   (blockIdx.x,         _)
So blockIdx.y is not used for A. That makes sense: A doesn’t depend on which column-tile of C
is being computed.
What gA is, exactly: a view (no copy) of this CTA’s strip of A, cut into k-tiles:
gA shape:  (BLK_M, BLK_K, k) = (128, 64, 64)       k = K / bK = 4096 / 64 = 64
• modes 0 and 1: one 128 × 64 tile;
• mode 2: which k-tile, 0 … 63. The main loop walks this mode, one k-tile per iteration
(tAgA(_,_,_,k_tile_next), L222).
For blockIdx.x = 3, gA covers rows 384 … 511 of A, all 4096 columns, split into 64 tiles of 64
columns.

gB: Step<X, _1, _1>
Skip M, use N and K. Here, blockIdx.x is the one not used, and gB is built from blockIdx.y:
gB shape: (BLK_N, BLK_K, k) = (128, 64, 64)
For blockIdx.y = 5, gB covers rows 640 … 767 of B (B is N × K), split into 64 k-tiles.
Completing the picture: gC (L74), for comparison
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});   // (BLK_M, BLK_N)
Here K is skipped, both in the tiler and in the coordinate, so the _ is dropped too. gC has no
k mode: it’s a single 128 × 128 tile, the output this CTA writes once, at the end.
Summary
</pre>
|Tensor	|Uses	|Shape	|For CTA (3, 5)|
|:------|:----|:------|:--------------|
|gA|	blockIdx.x, all K|	(128, 64, 64)	|A rows 384–511, 64 k-tiles|
|gB|	blockIdx.y, all K|	(128, 64, 64)	|B rows 640–767, 64 k-tiles|
|gC|	both	|(128, 128)|	C rows 384–511 × columns 640–767|
<pre>  
Each main-loop iteration takes k-tile k of gA and of gB. Over 64 iterations, they cover the 
full K = 4096 needed for this C tile.
Over 64 iterations, they cover the full K = 4096 needed for this C tile:
It means we need to loop 64 times, 4096/64=64.
Each CTA runs the main loop K / bK = 4096 / 64 = 64 times, one k-tile per iteration.
I can see it in the code:
L181: auto K_TILE_MAX = size<3>(tAgA);, the size of the k mode, i.e. 64. (tAgA is gA
partitioned for one thread, so its last mode is still the k-tile count.)
L213: for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile): the loop over those 64 k-tiles.
Two details:
• The count is a runtime value. K comes from the command line (L495–497), so 64 is only the
result for the default K = 4096. That’s also why the main loop isn’t unrolled (CUTE_NO_UNROLL,
L212), while everything inside it, with its static counts, is.
• K should be a multiple of 64. The tutorial has no handling for a partial last tile, so with K
= 4100, for example, the last k-tile would read past the end of each row. I believe nothing in
this file guards against that; the same goes for M and N not being multiples of 128. Production
CUTLASS kernels handle it with predication (masking out-of-bounds elements), which is the topic
of the 0y_predication tutorial in your CuTe reading list.

tAgA naming convention from CuTe’s GEMM tutorials, where each letter has a meaning. Read as four parts:

t   A   g   A
│   │   │   └─ which tensor:        A (the matrix)
│   │   └───── where it lives:      g = global memory
│   └───────── partitioned by:      A = the copy pattern for A (copy_a)
└───────────── "thread": this thread's share of the data


So tAgA reads as: this thread’s partition, according to the A-copy pattern, of the global-
memory tile gA. It’s created at L88: tAgA = thr_copy_a.partition_S(gA).
The same convention covers every partitioned tensor in this file: 
</pre>
|Name|	Partitioned by|	Memory|	Tensor|	Line|
|:---|:---------------|:------|:------|:----|
|tAgA|	A copy|	global|	A|	L88|
|tAsA|	A copy|	shared|	A	|L90|
|tBgB|	B copy|	global|	B	|L93|
|tBsB|	B copy|	shared|	B|	L95|
|tCsA|	MMA|	shared|	A|	L128|
|tCsB|	MMA|	shared|	B|	L129|
|tCgC|	MMA	|global	|C	|L130|
|tCrA|	MMA	|register|	A|	L133|
|tCrC|	MMA	|register	|C	|L136|
<pre>
Two things that commonly confuse readers:
• The MMA partition is marked C, even when it partitions A or B (tCsA). The C refers to the MMA
partitioning pattern (named after the accumulator C it produces), not the matrix being
partitioned. The last letter says which matrix.
• r means register only by naming. As we found earlier, for this _SS MMA, tCrA and tCrB actually
hold shared-memory descriptors, not A and B values. The name follows the convention even where
the contents are unusual.

partitioned by: A = the copy pattern for A (copy_a) which means the way how to copy, a 
plan or recipe.
copy_a is the TiledCopy object built on the host at L387, and it fixes two things:
• Which instruction: cp.async moving 16 bytes (8 halfs) at a time.
• Which thread copies which elements: the thread layout (16,8):(8,1) plus the value layout
(1,8). Together they say that thread t copies row t/8, K elements 8·(t mod 8) … +7, in each 16
× 64 copy step.
“Partitioned by A” then means: take the tile gA and cut out only the elements that this 
thread copies, following that plan. That’s what L87–88 do:
ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);   // the plan, for this one thread
Tensor tAgA = thr_copy_a.partition_S(gA);              // this thread's elements of gA

For example, thread 13 (row 13/8 = 1 of the thread grid, column 13 mod 8 = 5) gets, in every
k-tile:
rows 1, 17, 33, …, 113 (8 copy steps, 16 rows apart) × K elements 40 … 47
That’s 8 × 8 = 64 halfs per k-tile, as 8 cp.async instructions of 16 bytes each.
The same plan is applied to the destination too: tAsA = thr_copy_a.partition_D(sA_) (L90). 
So each thread’s source elements in global memory and destination elements in shared memory
line up one-to-one, and copy(copy_a, tAgA(…), tAsA(…)) just moves them.

What the plan(recipe) decides is how a tile is divided among the 128 threads:
</pre>
|Already fixed before partitioning	|Decided by the copy plan(recipe) (copy_a)|
|:----------------------------------|:----------------------------------------|
|A’s shape and strides (gA: 128 × 64 tiles, K-major)|	which elements of a tile each thread gets
||	which instruction moves them (cp.async, 16 bytes)|
<pre>
A more accurate phrasing:
“Partitioned by A” = divided among the threads according to the copy plan for A: which elements
of each tile every thread copies, and with which instruction.
The result of the partitioning, tAgA, does have its own shape and strides. They’re derived by
combining gA’s layout with the plan, rather than planned directly. For example, tAgA has shape
(CPY, CPY_M, CPY_K, k) = (8, 8, 1, 64) for each thread: 8 halfs per cp.async, 8 copy steps
along M, 1 along K, 64 k-tiles.
A description of tAgA: A thread view in global matrix A, the thread responsible for copying,
and the thread need to following the copy plan(recipe) for A. it’s a view of this CTA’s tiles
gA, not of the whole matrix A, and it contains only the elements this thread copies.
tAgA is one thread’s view of the global-memory tiles gA: exactly the elements of A that this
thread is responsible for copying, chosen according to the copy plan for A (copy_a). 
It’s the source side of the copy; tAsA is the matching destination in shared memory.

Like every CuTe tensor here, it’s only a view: no data is moved until copy(copy_a, tAgA(…),
tAsA(…)) runs (L193, L222).
gA isn’t the entire matrix, and tAgA covers more than one tile.
</pre>
|Tensor	|Line	|What it covers|
|:------|:----|:-------------|
|mA	|L66	|the entire matrix A (M × K = 5120 × 4096)|
|gA|	L72	|this CTA’s strip of A: 128 rows × all of K,<br> cut into 64 k-tiles of<br> 128 × 64 → shape (128, 64, 64)|
|tAgA|	L8	|this thread’s elements of gA,<br> in all 64 k-tiles → shape (8, 8, 1, 64)|
<pre>
So the chain narrows step by step:
mA   (whole matrix)
 └─ gA    (one CTA's 128-row strip, as 64 k-tiles)
     └─ tAgA  (one thread's 64 elements in each of those 64 k-tiles)
  
The last mode of tAgA is still the k-tile index (64), the same as gA’s. The main loop picks 
one k-tile per iteration from it: tAgA(_,_,_,k_tile_next) (L222). That slice is “one thread’s
view of one tile”.
gA holds 64 tiles, all from one row of tiles: the 128-row strip belonging to this CTA.
Matrix A as a whole is divided into many more:
A (5120 × 4096) in 128 × 64 tiles:
   along M: 5120 / 128 = 40 rows of tiles
   along K: 4096 /  64 = 64 tiles per row
   total:   40 × 64 = 2560 tiles
            k-tile 0   1   2  …  63
tile-row  0 [  ][  ][  ] … [  ]
tile-row  1 [  ][  ][  ] … [  ]
   …
tile-row  b [■■][■■][■■] … [■■]   ← gA for the CTA with blockIdx.x = b: 64 tiles
   …
tile-row 39 [  ][  ][  ] … [  ]

So: gA = the 64 tiles in this CTA’s row of tiles (selected by blockIdx.x), out of 2560 tiles 
in total. The other 39 tile-rows belong to CTAs with different blockIdx.x values.

One gA has 64 tiles.
gA shape: (128, 64, 64)
            │    │   └─ 64 tiles   (K / bK = 4096 / 64)
            │    └───── each tile is 64 columns wide (bK)
            └────────── and 128 rows tall (bM)
That’s 64 tiles × (128 × 64) = 524,288 elements: this CTA’s full 128-row strip of A, across all
of K.

Across the whole grid, there are 40 different gAs, one per tile-row of A (blockIdx.x = 0 … 39).
Each is used by 40 CTAs, the ones with the same blockIdx.x but different blockIdx.y, because
every C tile in the same row needs the same strip of A:

40 different gA strips × 64 tiles = 2560 tiles = all of A 
A has 2560 tiles of 128 × 64:
40 tiles along M (5120 / 128) × 64 tiles along K (4096 / 64) = 2560
Each gA is one row of 64 of them.
B is divided the same way: B is N × K = 5120 × 4096, also 40 × 64 = 2560 tiles. C is divided
into 40 × 40 = 1600 tiles of 128 × 128, one per CTA.  
40 tiles along M (5120 / 128). We have 5120 because A's M = 5120.
A is M × K, and M = 5120 is the default set in main (L487: int m = 5120;), so:
tiles along M = M / bM = 5120 / 128 = 40
Since M is a runtime value, this changes if you pass a different M on the command line 
(e.g. ./wgmma_sm90 2560 … gives 2560 / 128 = 20). It’s the same number as the grid’s x
dimension (L416, ceil_div(m, bM)), because there’s one CTA per tile-row along M.
gA is 64 tiles, 2560 is the number of tiles in all of A. gA is one CTA’s row of tiles:
</pre>
||Tiles|
|:--|:----|
|matrix A (mA)	|2560|
|gA (one CTA)	|64|
<pre>
one thread copies a part of each tile, not a whole tile
A thread doesn’t belong to a tile. It belongs to a CTA (128 threads). All 128 threads copy
each tile together: 
one tile = 128 × 64 = 8192 elements
8192 elements / 128 threads = 64 elements per thread per tile
For each of the CTA’s 64 tiles, thread t copies its 64 elements (8 cp.async × 8 halfs), and 
the other 127 threads copy the rest.
The full picture  
A                 2560 tiles
 └─ gA            64 tiles        (one CTA, picked by blockIdx.x)
     └─ one tile  8192 elements   (copied by all 128 threads together)
         └─ tAgA  64 elements     (this thread's share of each tile)
A has 2560 tiles; gA is one CTA’s row of 64 of them. In tAgA, t is one of the CTA’s 128
threads. For every tile in gA, that thread copies its own 64 elements, chosen by the copy 
plan for A; the 128 threads together copy the whole tile.
The g means global memory, but gA isn’t the whole global matrix. It’s this CTA’s tiles of A 
(64 of them), which live in global memory.
The prefixes in this file follow a convention:
</pre>
|Name|	Prefix meaning|	What it is|
|:---|:---------------|:-----------|
|mA|	m = full matrix|	all of A (2560 tiles)|
|gA|	g = global-memory tile(s) for this CTA	|64 tiles of A|
|sA|	s = shared memory|	the 3 pipeline stages of A|
|tCrA|	r = register (by convention)|	per-thread MMA fragment|
<pre>
Both mA and gA are in global memory. The difference is the extent: mA covers the entire matrix,
gA only the part one CTA works on.
gA is one array of tiles, its total tiles is 64. 
gA is a sequence of 64 tiles, each 128 × 64, indexed by its last mode:
gA(_, _, 0)   → tile 0
gA(_, _, 1)   → tile 1
…
gA(_, _, 63)  → tile 63
it’s a view into A in global memory, not a separate array with its own copy of the data.
Indexing it just computes addresses inside matrix A.
1. A thread isn’t “in the tiles”. t is one of the 128 threads of this CTA. The threads belong
to the CTA; the tiles are data.
2. One thread doesn’t copy whole tiles. For each of the 64 tiles, thread t copies only its own
part. All 128 threads together copy each complete tile.
In tAgA, t is one of the CTA’s 128 threads. For each of the 64 tiles in gA, this thread copies
its own small part of the tile, chosen by the copy plan for A. The 128 threads together copy
each whole tile.
t is one of the 128 threads, following the copy plan for A. 
1. A thread doesn’t own a tile; it owns a part of every tile. All 128 threads share each tile.
No thread copies a whole tile by itself.
2. A thread works on all 64 tiles, one per loop iteration. In iteration 0 it copies its part o
tile 0, in iteration 1 its part of tile 1, and so on (tAgA(_,_,_,k_tile_next), L222).

               tile 0     tile 1     tile 2   …   tile 63
thread 0     [part 0]   [part 0]   [part 0]  …  [part 0]
thread 1     [part 1]   [part 1]   [part 1]  …  [part 1]
  …
thread 127   [part 127] [part 127] [part 127] … [part 127]

tAgA for thread t is one row of this picture: its part of every tile. One column is a whole
tile, copied by all 128 threads together in one loop iteration.
In tAgA, t is one of the CTA’s 128 threads. Following the copy plan for A, it is responsible
for its own part of each of the 64 tiles in gA, one tile per loop iteration. All 128 threads
together copy each complete tile.  

• CTA = one thread block. “Array” refers to its threads, indexed by threadIdx.x = 0 … 127. 
Here one CTA = 128 threads (L414, dimBlock(size(tiled_mma))).
“Cooperative”: those threads work together. They share the CTA’s shared memory (the 96 KB 
holding sA and sB) and can synchronize with each other (__syncthreads(), L201).
• There are many CTAs: the grid has 40 × 40 = 1600 of them. Each is one array of 128 threads
computing one 128 × 128 tile of C.

extern __shared__ char shared_memory[];
using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{});
using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  
extern __shared__ char shared_memory[]; (L77)
dynamic shared memory, seen as a flat array of bytes. Three details:
• No size is written. extern with an empty [] means the size is supplied at launch: 
smem_size = 98,304 bytes (L413, L418). That’s what makes it dynamic.
• One copy per CTA. Each of the 1600 CTAs gets its own 98,304 bytes, shared by its 128 threads.
CTAs can’t see each other’s shared memory.
• char is just the unit for addressing. The real types come from the next two lines.
  
using SharedStorage = SharedStorage<…>; (L78)
a shorthand alias.
• The alias reuses the template’s name. From here on, inside this function, SharedStorage means
the concrete struct SharedStorage<half_t, half_t, ASmemLayout, BSmemLayout>. It’s legal C++, 
but easy to misread.
• ASmemLayout and BSmemLayout here are the kernel’s template parameters (L29–30), the same types
the host produced with decltype(sA). So the struct is the same on both sides, and its size 
matches the smem_size passed at launch.

SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory); (L79)
a reference to the struct. What makes this line work:
Nothing is created or copied. reinterpret_cast just tells the compiler: treat these bytes as a
SharedStorage. It’s a typed view onto the raw shared memory.
After this line, smem.A and smem.B name the two regions:
byte      0 … 49,151   smem.A   (24,576 halfs: 3 stages of A)
byte 49,152 … 98,303   smem.B   (24,576 halfs: 3 stages of B)

Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{}); (L80)
a shared-memory tensor view of A. Three details:
smem.A.begin() returns a half_t* pointing to the first element of A’s array: byte 0 of shared
memory.
make_smem_ptr wraps it and tags it as shared memory, like make_gmem_ptr did for global memory. 
CuTe uses the tag to choose valid operations (for example, cp.async needs a shared-memory 
destination).
ASmemLayout{} creates the layout from its type alone. That works because the layout is fully
static: every number is part of the type, so default-constructing it gives the complete
swizzled (128, 64, 3) layout. The kernel also received it as an argument (sA_layout, L37), 
but doesn’t need that value.
The resulting sA covers all 3 pipeline stages. One stage is selected later, for example with
tAsA(_,_,_,k_pipe_write) (L222).
Summary
shared_memory (98,304 raw bytes, size set at launch)
   └─ reinterpret_cast → smem (struct view: A array, B array)
        └─ smem.A.begin() + ASmemLayout → sA (tensor: 128 × 64 × 3, swizzled, in shared memory)

sB (L81) is the same, built from smem.B.

Lien 87: ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
Lien 88: Tensor tAgA = thr_copy_a.partition_S(gA); 
Lien 89: Tensor sA_ = as_position_independent_swizzle_tensor(sA);
Lien 90: Tensor tAsA = thr_copy_a.partition_D(sA_);
L87: copy_a.get_slice(threadIdx.x)

copy_a is the plan for the whole team of 128 threads, and get_slice(threadIdx.x) returns
thr_copy_a, the same plan specialized to this one thread.
get_slice doesn’t cut out any elements yet. It only records which thread this is. The cutting
happens on the next lines, when the plan is applied to a tensor with partition_S / partition_D.
  
L88: thr_copy_a.partition_S(gA)
each tile is 8192 elements, the 128 threads share the work, each thread is responsible for 64
elements per tile, and partition_S means “partition the Source”.
thr_copy_a doesn’t read anything, and neither does partition_S. Both only build views:
</pre>
|Step	|What it does|
|:----|:-----------|
|get_slice (L87)|	records which thread|
|partition_S (L88)|	builds the view tAgA: this thread’s elements of gA|
|copy(copy_a, tAgA(…), tAsA(…))<br> (L193, L222)|	this is where the cp.async <br>instructions run and data actually moves|
<pre>
gA holds 64 tiles, so tAgA covers this thread’s 64 elements in each of the 64 tiles: shape 
(8, 8, 1, 64). One tile is selected later, in the loop.

L89: as_position_independent_swizzle_tensor(sA)
1. The swizzle here is Swizzle<3,4,3>. In this kernel’s layout (and the build log), the swizzle
is written in byte-address bits: Swizzle<3,4,3>.
2. An XOR swizzle doesn’t survive simple shifting. Ordinary layouts are linear: if I move to a
thread’s starting point, I can just add an offset. An XOR swizzle isn’t linear:

swizzle(base + offset)  ≠  base + swizzle(offset)        in general

So when a swizzled tensor is partitioned (split into a per-thread starting point plus a 
remaining layout), CuTe must be careful about which address the swizzle is applied to. 
That’s the problem this function deals with.

3. The consequence would be wrong addresses, not just wrong banks. If the swizzle were applied
to the wrong value, data would land in the wrong place in shared memory. The wgmma would then
read the wrong elements and the GEMM would compute wrong results, silently, as discussed with
alignment. Bank conflicts only affect speed; this would affect correctness.

4. “Locks the swizzle to the physical pointer” is the opposite of what the name says. 
“Position-independent” means the result does not depend on where the buffer sits in memory: 
the swizzle is applied to the offset within the buffer. That’s only valid if the buffer starts
at a multiple of the swizzle period (1024 bytes here), which is why the alignment of 
SharedStorage matters.

What I’m not certain of: CuTe’s exact internal representation. As far as I know, a swizzled
shared-memory tensor can keep the swizzle either in the layout or folded into a special
swizzled pointer, and this function converts between those forms. Why the tutorial applies it
to the copy side only (L89, L94) but not to the MMA side (L128 uses plain sA) is something I
can’t explain with confidence from memory. Rather than guess, let’s read the actual code:

grep -rn "as_position_independent_swizzle_tensor" -A20 ../cutlass/include/cute/ | head -60
  
L90: thr_copy_a.partition_D(sA_)
it returns the view tAsA, this thread’s partition of the shared-memory tile, using the same 
A-copy plan, and D means Destination.
thr_copy_a doesn’t write. Writing happens in copy().
One difference from tAgA worth noticing:
</pre>
||Shape	|Last mode|
|:--|:--|:--------|
|tAgA (source, global)|	(8, 8, 1, 64)|	64 k-tiles|
|tAsA (destination, shared)|	(8, 8, 1, 3)	|3 pipeline stages|
<pre>
The first three modes match exactly (the kernel checks this at L97–98), so each thread’s source
elements pair one-to-one with its destination elements. The loop then pairs one k-tile with one
stage in each copy: copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe_write)) (L222).

Summary
get_slice specializes the A-copy plan to this thread. partition_S and partition_D apply it to
the source gA and the destination sA_, producing two views of this thread’s elements: tAgA (64 
elements in each of 64 k-tiles) and tAsA (the matching 64 elements in each of 3 stages).
Nothing is read or written yet; that happens in copy(). as_position_independent_swizzle_tensor
rewrites the swizzled shared-memory tensor so partitioning still produces correct swizzled
addresses; this relies on the buffer being aligned to the swizzle period.

Swizzle<3,4,3> is right because it’s what the build log shows, which is the most reliable
source for what this kernel actually uses.
these three commands give the full picture. Paste all three outputs:
# 1. The function itself
grep -rn "as_position_independent_swizzle_tensor" -A20 ../cutlass/include/cute/ | head -60

# 2. Where CuTe turns a swizzled layout + smem pointer into a "swizzle pointer"
grep -rn "swizzle_ptr" ../cutlass/include/cute/ | head -20

# 3. The files involved, so we know where to read further
grep -rln "as_position_independent_swizzle" ../cutlass/include/cute/

Optionally, a fourth check makes the difference visible: in the print_layouts.cu, build a small
shared-memory-style tensor from sA_tn and print it before and after
as_position_independent_swizzle_tensor. Comparing the two printed types shows directly where
the swizzle sits in each form. I can write that together once I’ve read the function.

print_layouts.cu output confirms most of the earlier analysis with real data from my machine,
and it adds two new facts. Line by line:
The warning (first line)
nvcc warning : Support for offline compilation for architectures prior to '_75' will be
removed …

I compiled without -arch, so nvcc used its default target, an old architecture that’s being
phased out. It’s harmless for this host-only program. To silence it, compile for my own GPU:

nvcc -std=c++17 --expt-relaxed-constexpr -arch=sm_120 -I … print_layouts.cu -o print_layouts
Shared-memory layouts: 
sA_tn : Sw<3,4,3> o smem_ptr[16b](unset) o ((_8,_16),(_64,_1),(_1,_3)):((_64,_512),(_1,_0),(_0,_8192))
sA_nt : Sw<3,4,3> o smem_ptr[16b](unset) o ((_64,_2),(_8,_8),(_1,_3)):((_1,_512),(_64,_1024),(_0,_8192))
cosize: _24576
Both layouts match the types in your build log exactly, and the TN one matches the hand
derivation.
The three parts appear in order: swizzle Sw<3,4,3> ∘ 16-bit element flag smem_ptr[16b] ∘ layout.
(unset) means no actual pointer is attached yet: this is only a layout, as expected on the host.
cosize = 24,576 halfs = 3 stages × 8192, the size of SharedStorage::A.
The _ prefix on every number marks it as static (_8 = Int<8>).
Tiler_MN:       (_16,_64)
TiledLayout_TV: ((_8,_16),_8):((_128,_1),_16)

Exactly the 16 × 64 copy step and the TV layout derived earlier: thread t copies row t/8, K =
8·(t mod 8) … +7.

Copy_Atom  ThrID: _1:_0   ValLayoutSrc/Dst: (_1,_8):(_0,_1)   ValueType: 16b
ThrID: _1:_0: one copy atom is executed by one thread. cp.async is a per-thread instruction.
(_1,_8):(_0,_1): each atom moves 8 contiguous values, and ValueType: 16b makes them halfs: 
8 × 2 bytes = the 16-byte cp.async.
ThrLayoutVMNK: (_128,_1,_1,_1)     ThrID: _128:_1     Shape_MNK: (_64,_64,_16)

128 threads (one warpgroup), one MMA atom of 64 × 64 × 16. This confirms dimBlock =
size(tiled_mma) = 128.

New finding 1: A and B aren’t distributed among threads
LayoutA_TV: (_128,(_64,_16)):(_0,(_1,_64))
LayoutB_TV: (_128,(_64,_16)):(_0,(_1,_64))

The thread mode has stride 0. All 128 threads map to the same 64 × 16 block of A: no thread
owns a separate piece. That’s how CuTe expresses the _SS case: the operand is read from shared
memory by the hardware, through a descriptor, not split among threads’ registers.

This is direct evidence for the earlier claim that tCrA and tCrB hold descriptors, not A and B
values, now from your own printout rather than my explanation.

New finding 2: who holds which accumulator element
LayoutC_TV: ((_4,_8,_4),(_2,_2,_8)):((_128,_1,_16),(_64,_8,_512))

This maps (thread, value) to a position in the 64 × 64 C tile (column-major: index = m + 64·n).
I decoded it and checked that it covers all 4096 elements exactly once:
</pre>
|Thread	|Its first values (m, n)|
|:------|:-----------------------|
|0|	(0,0) (0,1) (8,0) (8,1) (0,8) (0,9) …|
|1|	(0,2) (0,3) (8,2) (8,3) (0,10) (0,11) …|
|4|	(1,0) (1,1) (9,0) (9,1) (1,8) (1,9) …|
|32	|(16,0) (16,1) (24,0) (24,1) …|
<pre>
Each thread holds 32 values per atom (2 × 2 × 8). The CTA uses 2 × 2 atoms, so 128 values per
thread: the 64 accumulator registers (FP16 packed two per register) from the analysis.
Each warp covers 16 rows (thread 32 = warp 1 starts at row 16).
A thread’s values come in pairs adjacent along N: (0,0) and (0,1).

That last point supports the earlier epilogue hypothesis. C is stored M-contiguous (dC = (1,
ldC)), but each thread’s adjacent values run along N, so the final stores to C probably aren’t 
well coalesced. It’s still a hypothesis until the SASS (store instruction widths) or ncu 
confirms it, but now it’s based on the actual layout.

The 8 × 8 table matches the one computed earlier exactly: physical chunk = logical chunk XOR
(row mod 8).

What this run settled  
sA indicates it is shared memory Matrix A. We use cta_tile object to make shape in shared
memory. bM, bK, and bP is integer, static, evaluate in compile-time, bM, bK, bP dimension 
carve out physical bound of shared memory in compile-time . GMMA::Layout_K_SW128_Atom<TA> is a 
configuration of specification hardware  for Hopper/Blackwell shared memory.
128 byte hardware swizzle, tell the CuTe to apply 128 byte swizzle. We get bank conflict free.
K is a contiguous dimension. The elements alone the K-dimension, sit right next to each other
in physical memory, moving contiguous across column which means matrix A is store a row-major
format in shared memory.    
The Layout_K_SW128_Atom is not 128x64, it is an atom 8x64, a tiny atom.

Sw<3,4,3> from the beginning, one step at a time. I checked every number below by running it.
Step 1: Where it appears and what it acts on
The build log and print_layouts output show:
Sw<3,4,3> o smem_ptr[16b] o ((_8,_16),(_64,_1),(_1,_3)):((_64,_512),(_1,_0),(_0,_8192))
The o means “composed with”, applied right to left:
the layout turns a coordinate (m, k, stage) into an element offset;
smem_ptr[16b] says elements are 16 bits, so element offset × 2 = byte offset;
Sw<3,4,3> rearranges that byte offset.
So the swizzle works on byte addresses within shared memory.
  
Step 2: The problem it solves, shared-memory banks
Shared memory is divided into 32 banks, each 4 bytes wide. A byte address belongs to bank:
bank = (byte_address / 4) mod 32
32 banks × 4 bytes = 128 bytes: one 128-byte row covers each bank exactly once. If several
threads access different addresses in the same bank at the same time, those accesses are 
served one after another: a bank conflict.

Without a swizzle, one atom is 8 rows × 128 bytes, row-major. The 16-byte chunk c of row r is
at byte 128·r + 16·c, so its first bank is:

(128·r + 16·c) / 4 mod 32 = (32·r + 4·c) mod 32 = 4·c        ← independent of r!

Chunk 7 of every row starts at bank 28, as the “before” column below shows. Accessing the same chunk in 8 different rows (a column of the tile) hits the same 4 banks 8 times: an 8-way conflict.

Step 3: The definition of Swizzle<B, M, S>

A swizzle takes some bits of the address and XORs them into other bits:

Take B bits starting at bit M + S, and XOR them into the B bits starting at bit M.

As a formula, for Swizzle<3,4,3>:

source bits: B = 3 bits starting at M + S = 7  → bits 7, 8, 9   (mask 0b1110000000 = 0x380)
target bits: B = 3 bits starting at M = 4      → bits 4, 5, 6

swizzle(x) = x XOR ((x AND 0x380) >> 3)

The >> 3 is the S: it shifts bits 7–9 down by 3 onto bits 4–6.

Step 4: What those bits mean in this atom

Lay out the byte offset within one 1024-byte atom (8 rows × 128 bytes):

bit:      9  8  7 │ 6  5  4 │ 3  2  1  0
          ─ row ─ │ ─chunk─ │ ─ byte within chunk ─
         (0 … 7)  │ (0 … 7) │       (0 … 15)
bits 0–3: byte within a 16-byte chunk. Untouched by the swizzle (they’re below M = 4).
bits 4–6: which 16-byte chunk in the row (0 … 7). These are changed.
bits 7–9: which row in the atom (0 … 7), since each row is 128 = 2⁷ bytes. These are the source, not changed.

So the swizzle is simply:

physical chunk = logical chunk XOR (row mod 8)
Step 5: Each of the three numbers
</pre>
