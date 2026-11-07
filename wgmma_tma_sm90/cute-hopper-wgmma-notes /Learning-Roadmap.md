<pre>
Line 271 - Line 276:
Tensor mA = make_tensor(A, make_shape(M,K), dA);
Tensor mB = make_tensor(B, make_shape(N,K), dB);
Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));
	
These four statements are part of the host-side TMA setup in NVIDIA's wgmma_tma_sm90.cu tutorial. They prepare the information that Hopper's 
Tensor Memory Accelerator (TMA) will use to move matrix tiles from global memory into shared memory.
The most important distinction is:
make_tensor() creates a logical view of an existing matrix, whereas make_tma_atom() creates a TMA copy-operation object using the source 
tensor, shared-memory layout, and tile shape. Neither function performs the actual data transfer.
The default execution path:
transA = 'N';
transB = 'T';
which calls gemm_nt, not gemm_tn. 
1. First, understand the complete context
Here is the relevant setup code:
// Global matrix strides
auto dA = make_stride(Int<1>{}, ldA);
auto dB = make_stride(Int<1>{}, ldB);
// CTA tile dimensions
auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int<64>{};
// Number of pipeline stages
auto bP = Int<3>{};
// Shared-memory layouts

auto sA = tile_to_shape(
    GMMA::Layout_MN_SW128_Atom\<TA\>{},
    make_shape(bM, bK, bP));
auto sB = tile_to_shape(
    GMMA::Layout_MN_SW128_Atom\<TB\>{},
    make_shape(bN, bK, bP));
// The four statements
Tensor mA = make_tensor(A, make_shape(M,K), dA);
Tensor mB = make_tensor(B, make_shape(N,K), dB);
Copy_Atom tmaA = make_tma_atom(
    SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
Copy_Atom tmaB = make_tma_atom(
    SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

There are three different kinds of objects:
</pre>
|Object|Meaning|Memory role|
|:-----|:------|:----------|
|mA, mB|CuTe tensor views of the full<br> global matrices|Describe global memory|
|Host-side<br> sA, sB|CuTe layouts for three<br> shared-memory pipeline stages|Describe shared-memory organization|
|tmaA, tmaB|TMA copy atoms containing the<br>information needed for the transfer|	Configure the TMA operation|
<pre>
2. Statement 1: Tensor mA = make_tensor(A, make_shape(M,K), dA);
Let's break it into three arguments.
Argument 1: A
A is a pointer to matrix A's data in GPU global memory.
It is not a pointer to shared memory.
Argument 2: make_shape(M,K)
This describes the logical dimensions of matrix A:
A belong to R^(M x K)
For the tutorial's default matrix dimensions:
M = 512;
K = 1024;
The shape is:
(512, 1024)
That means matrix A has 512 rows and 1024 columns logically.
Argument 3: dA
Earlier in gemm_nt:
auto dA = make_stride(Int<1>{}, ldA);
For the default configuration:
ldA = M;  // 512
Therefore:
dA = (1, 512)
In CuTe, a tensor's stride specifies how many elements to advance in memory when a logical coordinate increases by one.
For matrix A:
offset_A(m,k) = m + 512k
So:
mA(0,0)  // A[0]
mA(1,0)  // A[1]
mA(0,1)  // A[512]
mA(1,1)  // A[513]

A small column-major example
Suppose M = 4, K = 3, and dA = (1,4).
k = 0           k = 1        k = 2
A(0,0)          A(0,1)       A(0,2)
offset 0        offset 4     offset 8
A(1,0)          A(1,1)       A(1,2)
offset 1        offset 5     offset 9
A(2,0)          A(2,1)       A(2,2)
offset 2        offset 6     offset 10
A(3,0)          A(3,1)       A(3,2)
offset 3        offset 7     offset 11
*****************
This is additional explanation regarding column/row major, they are not specific in this tutorial.
M = 4, K = 3, and dA = (1,4) describe column-major storage, not row-major storage.
The important distinction is that the stride determines the memory layout, not the shape alone.
1). Column-major: dA = (1,4)
auto dA = make_stride(Int<1>{}, Int<4>{});
Tensor mA = make_tensor(A, make_shape(4,3), dA);
The linear memory offset is:
offset(m,k)=m x 1 + k x 4
Logical matrix A (4 rows × 3 columns)
          k=0       k=1       k=2
m=0       0          4         8
m=1       1          5         9
m=2       2          6        10
m=3       3          7        11
The numbers indicate linear memory offsets, not matrix values.Physical memory order
A(0,0), A(1,0), A(2,0), A(3,0),
A(0,1), A(1,1), A(2,1), A(3,1),
A(0,2), A(1,2), A(2,2), A(3,2)
Elements in the same column are contiguous.
  
2). Row-major: dA = (3,1)
For the same logical shape (4,3), the row-major stride would be:
auto dA = make_stride(Int<3>{}, Int<1>{});
The offset becomes:
offset(m,k)=m x 3 + k

	      k=0	      k=1	        k=2
m=0	    0	        1	           2
m=1	    3	        4	           5
m=2	    6	        7	           8
m=3	    9	       10	           11
Elements in the same row are contiguous.
  
3). Remember this rule
For a 2D CuTe tensor with shape (M,K):
Memory layout	Stride	Contiguous dimension
Column-major	(1,M)	First dimension, M
Row-major	(K,1)	Second dimension, K

Therefore, in the gemm_nt tutorial:
auto dA = make_stride(Int<1>{}, ldA);
Since ldA = M for tightly packed column-major A, this describes column-major global-memory storage.
CuTe’s name for the “column-major.” In the NT path both A and B are MN-major: M for A, N for B is the stride-1 mode.
One detail: ldA can be larger than M when the columns have padding. The layout is still column-major because the first dimension has 
stride 1.  

****************
  
Elements in the same column are adjacent in global memory because the first dimension has stride 1.
The result of make_tensor() is a tensor view. It combines an engine referencing the existing memory with a layout mapping logical coordinates 
to memory offsets.
It does not allocate a new matrix and does not copy its elements.
So I can summarize:
Tensor mA = make_tensor(A, make_shape(M,K), dA);
as:
Create a CuTe view of the complete global-memory matrix A, logically shaped (M,K) with column-major strides (1,ldA).

3. Statement 2: Tensor mB = make_tensor(B, make_shape(N,K), dB);
This line works the same way, except the matrix shape is (N,K) rather than (M,K).
Tensor mB = make_tensor(B, make_shape(N,K), dB);
The default values are:
N = 256;
K = 1024;
ldB = N;  // 256
dB = (1, 256);
Thus:
offset_B(n,k) = n + 256k
And:
mB shape  = (256, 1024)
mB stride = (1, 256)
Why is B shaped (N,K), not (K,N)?
B is described as (N,K) in both gemm_nt (L272) and gemm_tn (L356). The kernel always computes C(m,n) = Σ_k A(m,k)·B(n,k). The transB flag 
does not change the shape, only the strides: dB = (1, ldB) in NT (L252), (ldB, 1) in TN (L336). Nothing is transposed anywhere; the kernel 
reads B with whichever strides it is given.
	
4. Statement 3: Copy_Atom tmaA = make_tma_atom(...)
Now we reach the most important statement:
Copy_Atom tmaA = make_tma_atom(
    SM90_TMA_LOAD{},
    mA,
    sA(_,_,0),
    make_shape(bM,bK)
);
There are four arguments, each with a different purpose.
Argument	              Role
SM90_TMA_LOAD{}         Specifies a Hopper TMA global-to-shared-memory load operation
mA	                    Describes the global-memory source tensor
sA(_,_,0)               Provides the shared-memory layout of one pipeline stage
make_shape(bM,bK)	      Specifies the CTA's logical A tile shape, (128,64)

Argument 1: SM90_TMA_LOAD{}
A CuTe type that wraps the PTX instruction cp.async.bulk.tensor and selects how the host builds the atom.
It has static copy functions containing the inline PTX [CuTe cute/arch/copy_sm90_tma.hpp]
It selects Hopper's TMA load mechanism, which transfers data asynchronously from global memory into shared memory.
It does not itself execute a TMA instruction. That happens later when the kernel issues a corresponding copy() operation.
This tutorial uses SM90_TMA_LOAD, not the multicast variant SM90_TMA_LOAD_MULTICAST. 
*******************
This is additional explanation regarding SM90_TMA_LOAD and SM90_TMA_LOAD_MULTICAST:
The main difference between SM90_TMA_LOAD and SM90_TMA_LOAD_MULTICAST is the number of CTAs that can receive data from one TMA operation.
- SM90_TMA_LOAD transfers a tensor tile from global memory into the shared memory of one CTA.
- SM90_TMA_LOAD_MULTICAST allows one TMA operation to distribute the same tensor tile into the shared memory of multiple selected CTAs within 
the same thread-block cluster.
Both are CuTe operation tags for Hopper's Tensor Memory Accelerator, but they select different forms of the same PTX instruction.
1). Side-by-side comparison

Feature	                  SM90_TMA_LOAD	        SM90_TMA_LOAD_MULTICAST
Source	                  Global memory	        Global memory
Destination	One           CTA's shared memory   Selected CTAs' shared memories
Transfer execution	      Asynchronous         	Asynchronous
Uses TMA hardware 	        Yes	                  Yes
Uses a multicast        	No	                  Yes
CTA mask
Can distribute one      	No	                  Yes
tile to several CTAs
Barrier-based completion	Yes	                  Yes, 
                                                  for the selected recipients
Main advantage          	Efficient tensor	  Avoids redundant loads of 
                            loads                 shared input tiles

These distinctions follow NVIDIA's CuTe implementation and PTX instruction definitions.
[PTX ISA, cp.async.bulk.tensor; CuTe copy_sm90_tma.hpp]

2). Example: A cluster containing two CTAs
Suppose:
dim3 dimCluster(2, 1, 1);
The cluster contains two CTAs, numbered 0 and 1.
Both CTAs need exactly the same A tile.
Case A — SM90_TMA_LOAD: Two independent TMA loads
                  Global Memory: A tile
                         |
___________________________________________________
|TMA#1                                             | TMA #2 
CTA 0, shared A tile                               CTA 1, shared A tile
Each CTA independently issues a TMA load of the same data.
  
Case B — SM90_TMA_LOAD_MULTICAST: One multicast operation
              Global Memory: A tile
                         | 
                  One TMA multicast
___________________________________________________
|                                                 | 
CTA 0, shared A tile                               CAT 1, shared A tile

  
A single issued TMA multicast operation delivers the tile to both CTAs. Each CTA still has its own physical shared-memory copy.
Multicast is particularly useful for GEMM when neighboring CTAs need identical operand tiles. It can reduce redundant data movement and 
improve effective memory-system bandwidth. 

3). The multicast mask: Which CTAs receive the data?
This is an important additional argument in SM90_TMA_LOAD_MULTICAST.
The hardware instruction accepts a 16-bit ctaMask. Each bit represents a CTA's rank within its cluster.

For a two-CTA cluster:
Multicast destination selection
0b01 — CTA 0 only   (CTA 0 receives tile, CTA 1 not selected)
0b10 — CTA 1 only   (CTA 0 not selected, CTA 1 receives tile)
0b11 — Both CTAs    (CTA 0 receives tile, CTA 1 receives tile)
	
This illustrates the destination mask of one multicast instruction, not an entire GEMM kernel's loading schedule.
The multicast mask determines which CTAs receive data from one TMA instruction. It does not determine how all the A and B tiles are loaded
throughout the GEMM kernel.
We use the Hopper tutorial with a cluster containing two CTAs.
1). One TMA multicast instruction
Suppose:
dim3 dimCluster(2, 1, 1);
The cluster contains CTA 0 and CTA 1.
Now imagine CTA 0 issues one TMA multicast instruction (in a real kernel, every group member issues its own slice) with:
uint16_t multicast_mask = 0b11;
One TMA multicast operation
This instruction transfers one particular B tile to the shared memory of both CTAs.
The mask 0b11 means:
- Bit 0 = 1: CTA 0 receives the tile.
- Bit 1 = 1: CTA 1 receives the tile.
That is all the mask specifies for this operation.
It does not tell the GPU which B tile to load next or which K tile WGMMA should compute.
2). But GEMM requires many tile loads
Suppose the GEMM has nine K tiles:
K tile 0
K tile 1
K tile 2
...
K tile 8
Assume both CTAs need the same B tile for each K iteration.
The kernel must issue multiple TMA operations:
TMA operation	Source	     Multicast mask	  Recipients
1	            B tile 0	 0b11	          CTA 0 and CTA 1
2	            B tile 1	 0b11	          CTA 0 and CTA 1
3	            B tile 2	 0b11	          CTA 0 and CTA 1
...	...	...	...
9	            B tile 8	 0b11             CTA 0 and CTA 1

The kernel's pipeline controls when each of these transfers is issued and which shared-memory pipeline stage receives the data.
For example, with three pipeline stages:
K tile	 Destination pipeline stage  	Multicast mask
0	     0	                            0b11
1	     1	                            0b11
2	     2	                            0b11
3	     0 (reused)                 	0b11
4	     1 (reused)	                    0b11
5	     2 (reused)	                    0b11

Notice the difference:
- Multicast mask 0b11: Which CTAs receive the tile.
- Pipeline stage 0, 1, or 2: Which shared-memory buffer within each recipient CTA is used.
- K tile 0, 1, 2, etc.: Which portion of the input matrix is being transferred.
These are three independent concepts.
3). Connect this to your code
In the tutorial, I studied:
copy(tma_b.with(producer_mbar[pipe]),
     tBgB(_,k_tile),
     tBsB(_,pipe));
Here:
k_tile selects the global-memory B tile, and pipe selects the destination pipeline stage.
This particular tutorial uses SM90_TMA_LOAD, not multicast, so no multicast destination mask appears in the call.
In a multicast-enabled implementation, a suitably configured operation would additionally identify the recipient CTAs using a multicast mask.
4). One important clarification
The table above is a hypothetical multicast-enabled GEMM schedule, not the actual loading schedule of the wgmma_tma_sm90.cu tutorial.
Also, just because the mask selects both CTAs does not mean both CTAs should independently issue the identical multicast operation.
In CuTe’s multicast pattern each CTA in the multicast group issues the load for its own slice of the tile, and the hardware delivers that slice 
to every CTA in the mask. The rank passed to tma_partition decides which slice a CTA issues.
In one sentence: The multicast mask answers “Who receives this tile?”, while the GEMM pipeline answers “Which tile is loaded, where is it stored,
and when can WGMMA use it?”
	
For example:
uint16_t mcast_mask = 0b0011;
Bits 0 and 1 are set, so both CTAs receive the data.
The TMA hardware writes the tile to the same CTA-relative shared-memory offset in every selected recipient, and propagates the associated
transaction-completion notification to the corresponding barriers. 

4. Difference at the PTX instruction level
The CuTe operations ultimately select different PTX instruction forms.
For a two-dimensional TMA transfer:
Normal TMA load:
cp.async.bulk.tensor.2d.shared::cluster.global
    .mbarrier::complete_tx::bytes

Multicast TMA load:
cp.async.bulk.tensor.2d.shared::cluster.global
    .mbarrier::complete_tx::bytes.multicast::cluster

The second instruction includes the multicast modifier and accepts the additional CTA mask operand.
The difference is visible in CUTLASS's implementation:
// Ordinary TMA load: no multicast_mask
SM90_TMA_LOAD_2D::copy(desc_ptr, mbar_ptr, cache_hint,
     smem_ptr, coord0, coord1);
// Multicast TMA load: includes multicast_mask
SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, mbar_ptr, multicast_mask,
     cache_hint, smem_ptr, coord0, coord1);
These are simplified signatures of the actual CuTe architecture-layer functions. 

5). How this relates to the wgmma_tma_sm90.cu tutorial
The tutorial creates:
Copy_Atom tmaA = make_tma_atom(
    SM90_TMA_LOAD{},
    mA,
    sA(_,_,0),
    make_shape(bM,bK));

This selects an ordinary, non-multicast TMA load.
Later, the kernel calls:
auto [tAgA, tAsA] =
    tma_partition(
        tma_a,
        Int<0>{},
        Layout<_1>{},
        group_modes<0,2>(sA),
        group_modes<0,2>(gA));

As I previously corrected, the meaning is:
- Int<0>{}: Rank 0 in the multicast group.
- Layout<_1>{}: Multicast-group layout with one CTA.
Even though the launch uses dimCluster(2,1,1), the TMA operation is not multicasted.

To use SM90_TMA_LOAD_MULTICAST instead, the kernel would need a coordinated multicast configuration: correct recipient mapping, multicast mask,
appropriate TMA partitioning, and correctly initialized barriers in the destination CTAs. Simply replacing SM90_TMA_LOAD{} with 
SM90_TMA_LOAD_MULTICAST{} is not sufficient. 

One additional detail: Per NVIDIA/cutlass Discussion #1315 (answer by @thakkarV, Jan 22, 2024): the SMEM layout passed to the TMA creator is 
that of a single CTA, not of the entire cluster, <a href="https://github.com/NVIDIA/cutlass/discussions/1315?utm_source=chatgpt.com">Source:
NVIDIA/cutlass Discussion #1315</a>. The SMEM layout passed to the TMA creator is that of a single CTA, not of the entire cluster, even for
multicast (NVIDIA/cutlass Discussion #1315, accepted answer by @thakkarV, Jan 2024, about make_tma_copy). I expect the same for make_tma_atom
	
6). When is multicast useful for GEMM?
Imagine two CTAs computing different output tiles:
C_0=A_0B, C_1=A_1B
Both need the same B tile, but different A tiles.
Without multicast, both CTAs independently load B. With multicast, a single TMA operation can distribute B into both CTAs' shared-memory 
buffers.
For this pattern, an efficient design might use:
- SM90_TMA_LOAD to load each CTA's unique A tile.
- SM90_TMA_LOAD_MULTICAST to distribute the common B tile to both CTAs.
This can reduce redundant traffic, especially through L2, although actual performance depends on tile sharing, cluster arrangement,
synchronization overhead, and the hardware.

The essential distinction: A CTA cluster determines which CTAs can cooperate. The multicast mask passed to a SM90_TMA_LOAD_MULTICAST copy 
decides which of those CTAs receive a particular transfer..
*******************
	
Argument 2: mA  (Line 275)
We have already established that mA describes the entire A matrix:
Global tensor mA
Shape  = (512,1024)
Stride = (1,512)
The TMA builder inspects this tensor to obtain information about the global-memory data organization.
This includes the base address, element type, dimensions, and strides.
Importantly, mA represents the entire A matrix, not the specific 128×64 tile owned by one CTA.
The CTA-specific region will be selected later, inside the kernel.
	
Argument 3: sA(_,_,0) (Line 275)
This is the part that deserves particular attention.
Recall that the host-side sA was constructed as:
auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom\<TA\>{}, make_shape(bM,bK,bP));
Its shape is:
sA layout: (128,64,3)  
Here:
- Mode 0 corresponds to M, with extent 128.
- Mode 1 corresponds to K, with extent 64.
- Mode 2 corresponds to the pipeline stage, with extent 3.
Now examine:
sA(_,_,0)
The underscore _ means retain that mode, while 0 selects pipeline stage 0.
Conceptually:
Host-side sA layout: (128,64,3)
Stage 0        Stage 1         Stage 2
128 × 64       128 × 64        128 × 64
Selected       Not selected    Not selected
          |
		  V  sA(_,_,0)
2D shared-memory layout
Logical shape (128,64), with the stage's swizzled address mapping
Important: Because this is inside the host-side gemm_nt function, sA is a layout object, not the kernel's shared-memory tensor.
Consequently, sA(_,_,0) is a sliced layout, not a data copy and not an already allocated shared-memory buffer.
Why select only stage 0?
Because all three pipeline stages use the same within-stage organization.
make_tma_atom() needs the layout of one destination tile to construct the transfer configuration. It does not need a separate TMA atom for 
every pipeline stage.
The actual destination stage is selected later inside the kernel:
tAsA(_,pipe)
where pipe can be 0, 1, or 2.
Argument 4: make_shape(bM,bK)
This creates the logical shape:
make_shape(Int<128>{}, Int<64>{})
or:
(128,64)
This tells CuTe the CTA tile to be transferred for A.
The distinction between the global matrix and the tile is:
Object	             Shape	       Meaning
mA	                 (512,1024)	   Entire A matrix
sA(_,_,0)	         (128,64)	   Layout of one shared-memory A tile
make_shape(bM,bK)	 (128,64)      Logical CTA A tile shape for TMA construction

The tile shape and the shared-memory layout both describe a (128,64) logical region, but they are not interchangeable: the shape 
describes the tile's extent, while the layout describes how its elements are arranged in shared memory.
For the default cute::half_t type:
128 x 64 x 2=16384 bytes
Thus, each A tile contains 16 KiB of FP16 data.
make_tma_atom plans the transfer and asks the CUDA driver to fill in a small hardware-format record. That record is stored inside tmaA, 
an ordinary host variable. Nothing is allocated on the GPU at this point.
1). CuTe (host C++) reads mA’s pointer, shape and strides and the layout sA(_,_,0). It works out the box shape, the dimension order and 
the swizzle mode. (CuTe copy_traits_sm90_tma.hpp)
2). CUDA driver: CuTe calls cuTensorMapEncodeTiled with plain numbers. The driver fills a 128-byte CUtensorMap. (CUDA driver API docs)
3). Where it lives: that struct sits by value inside tmaA in host memory.

Why gemm_nt uses Layout_MN_SW128_Atom.
TMA cannot transpose, so the smem tile must be contiguous in the same direction as global memory. In NT, A’s stride-1 mode is M, so the smem 
atom is MN-major (Line 263; GMMA::Major::MN at Line 267). gemm_tn uses K-major atoms (Line 347, Line 351). The driver also requires every 
global stride except the first to be a multiple of 16 bytes, so for half_t, ldA must be a multiple of 8.

5. Statement 4: Copy_Atom tmaB = make_tma_atom(...)
Copy_Atom tmaB = make_tma_atom(
    SM90_TMA_LOAD{},
    mB,
    sB(_,_,0),
    make_shape(bN,bK)
);
The mechanism is identical, except that it operates on B.
Argument	          Meaning for B
SM90_TMA_LOAD{}	      TMA global-to-shared load
mB	                  Complete B tensor, shape (256,1024)
sB(_,_,0)	          One shared-memory B-stage layout, shape (128,64)
make_shape(bN,bK)	  CTA's B tile shape, (128,64)

For the default example, A and B tiles have the same number of elements:
128 x 64 = 8192
Each operand’s per-stage copy moves 16 KiB (16,384 B), so the two transfers together account for 32 KiB per pipeline stage.
	
6. How do these four statements connect to the actual TMA transfer?
This is where host-side setup meets kernel execution.
TMA setup and execution
Host: Describe and configure
	
mA global tensor                sA smem layout
______________________________________________
	                |
	                V
            make_tma_atom(...)
            Constructs tmaA
      Kernel launch passes the TMA atom
	  - Launch: tmaA is passed by value as a kernel argument. In the kernel it is the __grid_constant__ parameter tma_a, 
	  so it stays in kernel parameter space.
	 // Kernel Launch
     Line 303: cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             A, tmaA,
                                                             B, tmaB,
                                                             C, dC, tiled_mma,
                                                             alpha, beta);
	  - Hardware: when the elected thread issues copy(tma_a.with(...), ...), the TMA unit reads the descriptor from that parameter space 
	  and fetches the box.

This function cutlass::Status status = cutlass::launch_kernel_on_cluster is the call that launches the kernel. Everything it list after params 
and kernel_ptr becomes a kernel argument, and tmaA is one of them.
How it matches the kernel. The arguments after params and kernel_ptr are in the same order as the parameters of gemm_device (Line 43–47):

Launch argument  	Kernel parameter
prob_shape	        shape_MNK
cta_tiler	        cta_tiler
A	                TA const* A
tmaA	            CUTLASS_GRID_CONSTANT TmaA const tma_a (Line 44)
tmaB	            CUTLASS_GRID_CONSTANT TmaB const tma_b (Line 45)

Why “by value”. CUDA kernel arguments are copied into the kernel’s parameter space at launch. So the whole tmaA object, including its 
128-byte descriptor, is copied there, and the kernel’s tma_a is that copy.

What the function itself does. It is declared in cutlass/cluster_launch.hpp, which the source includes at L10. I recall that it launches 
the kernel with the cluster dimension from params (the dimCluster(2,1,1)) through the CUDA extended launch API, cudaLaunchKernelEx. 

GPU kernel: Partition and execute:
tma_a.get_tma_tensor(...) : get_tma_tensor builds a tensor of coordinates, not of memory.
(TMA coordinate tensor: counting iterator + basis strides, no pointer)
local_tile(...) and tma_partition(...)
(CTA tile and TMA source/destination views)
copy(tma_a.with(...), ..., ...)
(Initiates the actual asynchronous TMA load)
	
Inside gemm_device, the tutorial builds a coordinate tensor of shape (M,K) from the TMA atom:
Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));
It is not the same object as the host-side mA. mA(m,k) is a TMA coordinate, and A’s address stays inside the descriptor. It is a 
descriptor-backed tensor suitable for the kernel's TMA operations.
Next:
Tensor gA = local_tile(
    mA,
    cta_tiler,
    cta_coord,
    Step<_1,X,_1>{}
);
gA is this CTA’s tiles of A as TMA coordinates, shape (128,64,k). It is not a view of memory.
Later:
auto [tAgA, tAsA] =
    tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                  group_modes<0,2>(sA),
                  group_modes<0,2>(gA));

Here, sA is now the kernel's shared-memory tensor, not the host-side layout. The Int<0>{} and Layout<_1>{} describe the single-CTA 
multicast-group configuration, not the two-CTA cluster layout.
Finally:
copy(tma_a.with(producer_mbar[pipe]),
     tAgA(_,k_tile),
     tAsA(_,pipe));

This initiates a TMA load for the selected K tile into the selected pipeline stage. 

7. A concrete example: Which elements does TMA load?
Consider the default matrices:
A: 512 × 1024
B: 256 × 1024

bM = 128
bN = 128
bK = 64

Suppose the CTA is responsible for:
blockIdx.x = 1;
blockIdx.y = 0;

and the kernel is processing:
k_tile = 3;
pipe   = 1;

For A, the CTA's M tile starts at:
m_start = 1 x 128 = 128

The selected K tile starts at:
k_start=3 x 64=192

Therefore, the A source region is:
A[128:256, 192:256]

******************
Explaining A[128:256, 192:256]
A[128:256, 192:256] means select 128 rows and 64 columns from the original matrix A.
Let's derive those numbers step by step using your wgmma_tma_sm90.cu tutorial.
1. Start with the original matrix A
In the default gemm_nt example:
M  = 512;
K  = 1024;
	
bM = 128;
bK = 64;

Therefore:
- Original matrix A has 512 rows and 1024 columns.
- Each CTA handles 128 rows of A.
- Each K tile contains 64 columns of A.
The original matrix has logical shape:
A:(512,1024)
2. How do we get rows 128 through 255?
Suppose the current CTA has:
blockIdx.x = 1;
The CTA's starting row is:
row start = blockIdx.x x bM
=1 x 128=128
Since bM = 128, the CTA handles 128 rows:
128,129,130,....,255
Thus:
A[128:256, ...]
Notice that 256 is excluded. The last included row is 255.
A matrix divided into 128-row CTA tiles
CTA M tile 0       CTA M tile 1      
Rows 0–127         Rows 128–255
CTA M tile 2       CTA M tile 3
Rows 256–383       Rows 384–511

3. How do we get columns 192 through 255?
Suppose the kernel is processing:
k_tile = 3;
Each K tile has width:
bK = 64;
Therefore:
column start = k_tile x bK
=3 x 64=192
The 64 columns are:
192,193,194,....,255
Thus:
A[..., 192:256]
Again, 256 is excluded.
4. Combine the two dimensions
Selected source region of A
Row selection
128–255
128 rows

Column selection
192–255
64 columns

A[128:256, 192:256]
128 × 64 elements

The first element is:
mA(128,192);

The last element is:
mA(255,255);

The selected region is therefore:
A_tile belong to R^128 x 64

5. How does this relate to column-major storage?
Recall the global-memory tensor:
Tensor mA = make_tensor(
    A,
    make_shape(M,K),
    make_stride(Int<1>{}, ldA)
);
For the default tightly packed matrix, ldA = 512.
Thus:
offset_A(m,k)=m+512k

The first selected element mA(128,192) is at offset:
128+512 x 192=98432
The last selected element mA(255,255) is at offset:
255+512 x 255=130815

These offsets are measured in elements, not bytes.
Important: Even though the selected A tile contains 8,192 elements, those elements are not stored as one contiguous 8,192-element region in 
global memory. Each selected column contains 128 contiguous elements, but successive columns are 512 elements apart at their starting 
addresses.
The TMA tensor descriptor uses the global-memory strides to locate the correct elements.
		
6. General formula
For the tutorial, the A tile selected by a CTA and a K-tile index is:
A[blockIdx.x x bM: (blockIdx.x+1) x bM, k_tile x bK:(k_tile+1) x bK]
For:
blockIdx.x = 1;
k_tile     = 3;
bM         = 128;
bK         = 64;
we get:
A[128:256,192:256]

So the two numbers 128 and 192 are starting coordinates, while the two numbers 256 and 256 are exclusive ending coordinates. 
The selected tile is 128 rows by 64 columns.
One more distinction: this source region is part of the full global-memory matrix mA. The kernel's gA is a CTA-specific view of that global
memory, and tAgA(_,k_tile) selects the TMA source portion for the relevant K tile.
*******************

For B, the CTA's N tile starts at:
n_start=0 x 128=0
So the B source region is:
B[0:128, 192:256]
The transfer can be represented as:
Global A    ------>         Shared A      
Rows 128–255                Pipeline stage 1
K 192–255                   MN-major swizzled
128 × 64                    128 × 64

Global B  -------->         Shared B
Rows 0–127                  Pipeline stage 1
K 192–255                   MN-major swizzled
128 × 64                    128 × 64

Illustrative tile selection for CTA coordinate (1,0), K tile 3, and pipeline slot 1. The source is global memory; the destinations are the
CTA's own shared-memory buffers.
Notice that the same tmaA and tmaB objects work for different CTA coordinates, K tiles, and pipeline stages. The global coordinate and
the destination buffer are selected at execution time.
8. Final comparison
Statement	                         What it creates	       Does it move data?
make_tensor(A, ...)	                 Global A tensor view	   No
make_tensor(B, ...)	                 Global B tensor view	   No
make_tma_atom(...mA...)              TMA copy atom for A       No
make_tma_atom(...mB...)              TMA copy atom for B	   No
Kernel copy(tma_a.with(...), ...)	 Initiates A transfer	   Yes, asynchronously
Kernel copy(tma_b.with(...), ...)	 Initiates B transfer	   Yes, asynchronously

The essential concept is that TMA setup separates the description of the memory layout from the execution of the transfer.
make_tma_atom() prepares the machinery. The kernel's copy() uses it.
One subtle point worth remembering: sA(_,_,0) does not mean TMA will always write to stage 0. It provides the layout information for a 
single stage; the actual destination stage is determined by tAsA(_,pipe).
That is the connection between these four host-side statements and the three-stage pipeline you have been studying.


start from here Nov 7
Line 42: void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
            TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
            TC      * C, CStride dC, TiledMma mma,
            Alpha alpha, Beta beta)

TA and TB are template type parameters of gemm_device (L37–38), so the kernel works with any
element type. In this program they become cute::half_t (FP16) because main sets using TA =
cute::half_t;. FP16 is this instantiation, not a fixed property of the kernel.
1. They are virtual, not physical, addresses, the same point as with L271.
2. In this kernel they tell the kernel nothing, because they are never used. Search the body of
gemm_device (L46–219): A and B never appear. The kernel reaches A and B only through tma_a and
tma_b, which already hold the base addresses inside their descriptors. Compare C (L47), which is
used, at L71 to build mC. So A and B are dead parameters, kept for signature symmetry.
  
CUTLASS_GRID_CONSTANT TmaA const tma_a:
tma_a is the whole Copy_Atom built at L275, and TmaA is its type, deduced as decltype(tmaA) at
L289. The atom contains the descriptor (CUtensorMap, 128 bytes) plus CuTe’s aux params.
• TmaA is a type; the object tma_a holds the values.
• The descriptor holds encoded numbers, not CuTe layouts: global dims, global strides in bytes, 
box dims, and a swizzle mode (an enum such as 128B), not “instructions.”
• The aux params hold CuTe’s mapping from gmem modes to TMA coordinates, which get_tma_tensor
uses at L68.

CUTLASS_GRID_CONSTANT / __grid_constant__
“Maps to __grid_constant__”: (CUTLASS, cutlass/detail/helper_macros.hpp).
“Read-only”: as a requirement. __grid_constant__ may only be applied to const-qualified kernel
parameters, which is why the signature says TmaA const.

Every kernel parameter is identical for the whole grid. All parameters are passed once per
launch and shared by all threads.
  
So the purpose isn’t saving registers. It’s making the descriptor’s address usable by the TMA
hardware. This connects back to the hardware/software distinction: the TMA unit reads the 
descriptor directly from memory, so the descriptor must sit somewhere the TMA unit can address.
Summary:
</pre>
|claim|	Verdict|
|:----|:-------|
|TA/TB are FP16|They are template type parameters;<br> FP16 (half_t) is this instantiation,<br> set in main.|
|A/B tell the kernel where<br> data starts in physical memory|Base pointers, yes, but virtual<br> addresses, and unused in this kernel.<br> The addresses travel inside the descriptors.|
|tma_a is the TMA descriptor|	It’s the whole Copy_Atom:<br> the 128-byte descriptor plus<br> CuTe aux params.|
|TmaA|The object holds encoded dims,<br> byte strides, box dims, a swizzle mode,<br> and CuTe’s coordinate mapping.|
|__grid_constant__ means<br> read-only|	 requires const.|
|… identical for the whole grid|True of all kernel parameters;<br> not specific to __grid_constant__.|
|Params are always<br> in param/constant space.| __grid_constant__ guarantees that taking<br> its address gives the param-space<br> address instead of a local-memory copy,<br> which the TMA instruction requires.|

</pre>
<pre>
Explaining where kernel parameters live and why they get copied.
Step 1: Where kernel parameters live
When you launch a kernel, its arguments (shape_MNK, tma_a, C, alpha, …) are copied once per
launch into a special memory region called parameter space. On the hardware, this is a constant
bank; in SASS you see reads like c[0x0][0x210].

All threads in the grid read the same copy, and this is true of every kernel parameter, with or 
without __grid_constant__. That’s what “params are always in param/constant space” means.
__grid_constant__ does not move the parameter anywhere.
Step 2: Two ways to use a parameter
(a) Use its value. For example, int M = get<0>(shape_MNK);. The thread reads the bytes from
param space into a register. No problem, no copy.
(b) Take its address. For example, &tma_a, or passing tma_a by reference to a function that
needs a pointer to it. Now the code needs a pointer to the parameter, and this is where the
issue arises.
Step 3: Why taking the address normally produces a copy
In C++, a function parameter is an ordinary object. Code holding a pointer to it could, in
principle, write through that pointer. But param space is read-only and shared by the whole
grid, so it can’t be handed out as an ordinary writable per-thread object.
So without __grid_constant__, when the code takes the address, the compiler plays it safe:
it allocates space in local memory, the thread’s private stack;
it copies the parameter’s bytes from param space into that local copy;
it gives us the address of the local copy.
Each thread now has its own private copy, and the pointer points there, not to param space.
With __grid_constant__, we promise the parameter is const and will never be modified. That’s why
the qualifier requires const. Given that promise, no private copy is needed, so taking the
address gives us the address of the original in param space.
Step 4: Why TMA cares
The TMA instruction does not receive the descriptor’s contents. It receives the address of the
descriptor. The TMA hardware unit then reads the 128 bytes itself, from that address.
The PTX documentation for cp.async.bulk.tensor requires that address to be in .param (a kernel
parameter), .const, or .global memory. Local memory is not allowed. Local memory is a thread’s
private stack, and the TMA unit is a separate engine, not that thread.
Putting it together:
Without __grid_constant__: &tma_a points to a local-memory copy, and the TMA instruction gets 
an address it cannot use.
With __grid_constant__: &tma_a points into param space, and the TMA instruction gets a valid
address.

An analogy: the descriptor is a notice posted on a shared board (param space). Without
__grid_constant__, each thread photocopies the notice into its own desk drawer (local memory)
and points to the drawer. The TMA unit can’t open anyone’s drawer; it can only read the board.
__grid_constant__ tells each thread to point at the board directly.

How you can verify this yourself on your 5080
This doesn’t need TMA, so it works on SM120. Write two tiny kernels that take the address of 
a large const struct parameter:

struct alignas(64) Blob { int v[32]; };            // 128 bytes, like a CUtensorMap
__device__ int sink(Blob const* p) { return p->v[threadIdx.x % 32]; }
__global__ void no_gc(Blob const b, int* out)                 { out[threadIdx.x] = sink(&b); }
__global__ void with_gc(__grid_constant__ Blob const b, int* out) { out[threadIdx.x] = sink(&b); }

Mark sink with __noinline__ so the compiler can’t optimize the address away.
Compile with nvcc -arch=sm_120 -Xptxas -v, and inspect with cuobjdump -sass. 
The two outputs to compare:
ptxas -v reports bytes stack frame per kernel. I expect no_gc to show a nonzero stack frame 
(the local copy) and with_gc to show 0.
In the SASS, no_gc should contain local stores (STL) copying the struct onto the stack, and
with_gc should not.

That’s my expectation from how the feature is specified. The compiler output is the real answer,
and it’s the kind of profiling-anchored check that makes this concrete.

Summary:
Step 1: all kernel parameters live in param space (a constant bank). __grid_constant__ doesn’t
change that.
Step 2: reading a parameter’s value is fine; the issue only arises when its address is taken.
Step 3: without __grid_constant__, taking the address makes the compiler copy the parameter
into per-thread local memory and return that copy’s address. With it (and const), you get the
original’s param-space address.
Step 4: the TMA instruction takes the descriptor’s address, which must be in param, const, or
global memory, never local. So __grid_constant__ is what makes tma_a usable by TMA.
Verify: compare the -Xptxas -v stack frame size and the STL instructions in SASS for the two
test kernels.

Line 46:   TC * C, CStride dC, TiledMma mma,
Checking the stride types for the output matrix.
Reviewing how the epilogue and the TiledMma argument are handled.
TC * C
“Raw base pointer for C”: As with A/B:
• TC is a template type parameter. It is half_t in this program only because main sets using 
TC = cute::half_t;.
• The address is virtual, not physical.
Unlike A and B, C is used: at L71 it becomes mC = make_tensor(make_gmem_ptr(C), make_shape(M,N),
dC). It is non-const because the kernel writes C.
CStride dC
• dC is the whole stride tuple, not the leading dimension. In gemm_nt it’s make_stride(Int<1>{},
ldC) (L248). The leading dimension is ldC, one component of it. main sets ldC = m.
• So mC is M-major: (M,N):(_1,ldC), counted in elements, not bytes.
• CStride is the type of that tuple: a template parameter deduced as decltype(dC), with a static
_1 and a dynamic int.
“C is written with ordinary stores, not TMA”
The source is the last line of gemm_device:
axpby(alpha, tCrC, beta, tCgC);
Each thread computes C = alpha·acc + beta·C on its own partition tCgC (from
thr_mma.partition_C(gC)) and writes it with ordinary per-thread global stores.

1. It may also read C. The beta·C term means C may be loaded as well. I’m not sure whether
CuTe’s axpby skips the load when beta == 0 (it is 0 here). Checking the SASS for LDG alongside
STG would settle it.
2. “Often” applies to tutorials, not production. Production CUTLASS SM90 epilogues commonly
stage C/D through shared memory and write it with TMA stores (SM90_TMA_STORE). This tutorial 
uses direct stores for simplicity, which is why C needs only a pointer and stride. That’s a
design choice of this example, not a general Hopper rule.
tma_a is the hardware descriptor for memory movement, mma is the hardware descriptor for 
math.These two parameters are fundamentally different kinds of objects.
</pre>
||tma_a|	mma (TiledMma)|
|:---|:----|:------------|
|Runtime data?|	Yes: a 128-byte CUtensorMap<br> built on the host with real values<br> (address, dims, strides)|	No: an empty object. All its<br> information is in its type.|
|Read by hardware?|	Yes: the TMA unit reads the<br> descriptor from param space|	No: it never reaches hardware.<br> It exists only at compile time.|
|Why pass it?|	To deliver the descriptor’s bytes|	Only so the kernel gets<br> the type (template deduction).<br> The value carries nothing.|
<pre>
Evidence it’s compile-time: __launch_bounds__(decltype(size(TiledMma{}))::value) (L40)
constructs a fresh TiledMma{} from the type alone and gets the thread count as a compile-time
constant.

WGMMA’s actual hardware descriptors are something else. They are the 64-bit shared-memory matrix
descriptors that encode the smem start address, layout type, and swizzle for each operand.
They’re built inside the kernel, by thr_mma.make_fragment_A(tCsA) / make_fragment_B(tCsB). The
tutorial comment above those lines says it: tCrA and tCrB are tensors of MMA descriptors, not
register data. So the closest analogue to tma_a’s descriptor is the contents of tCrA/tCrB, not
mma.
What TiledMma does encode:
Which instruction: SM90_64x64x16_F16F16F16_SS<MN, MN> (L264):
a 64×64×16 tile per instruction;
FP16 inputs and FP16 accumulator;
SS means both A and B are read from shared memory;
both operands are MN-major.
Thread arrangement: one warpgroup of 128 threads. With no extra tiling in make_tiled_mma,
size(tiled_mma) == 128, which sets dimBlock (L281) and __launch_bounds__.
Data mapping: how partition_A/B/C split smem and the accumulators among the 128 threads.
Matrix shapes are passed in explicitly. shape_MNK carries the dynamic (M,N,K) (L238 prob_shape),
and the kernel uses it at L67 (auto [M, N, K] = shape_MNK). It’s an input, not something the
kernel avoids.

What the template parameters actually decouple is the mechanics: memory layout, major-ness,
swizzle, and instruction choice. Evidence: gemm_nt and gemm_tn launch the same gemm_device body 
with different dA/dB, smem layouts, TMA atoms, and MMA atoms. The kernel code doesn’t change 
between NT and TN.

It is also not fully decoupled from tile shapes. The static asserts at L54–57 tie the smem 
layouts to cta_tiler. And the kernel still hard-codes assumptions:

ConsumerBarType::init(&consumer_mbar[pipe], 128) hard-codes 128 consumer threads.
The epilogue is labeled unpredicated: there’s no bounds check, so it assumes M and N are
multiples of the 128×128 CTA tile.
Summary:
</pre>
|Claim|	Verdict|
|:----|:-------|
|C is the raw base pointer<br> of the output|template type TC = half_t here;<br> virtual address;<br> non-const; used at L71).|
|dC | the same issue as dA. dC is the<br> full tuple (_1, ldC); ldC <br>is the leading dimension;<br> units are elements.|
|C is written by ordinary<br> stores, so no TMA descriptor|	for this kernel<br> (axpby, which may also read C for beta).<br> Production SM90 epilogues often use TMA stores.|
|mma, it never reaches hardware.<br> It exists only at compile time |TiledMma is an empty compile-time<br> type that never reaches hardware.<br> WGMMA’s hardware descriptors are<br> the smem descriptors in tCrA/tCrB.|
|TiledMma encodes the instruction,<br> the 128-thread arrangement,<br> the data mapping|	SM90_64x64x16_F16F16F16_SS,<br> one warpgroup, partition_*.|
|The kernel assumes 128 threads| Shapes come in via shape_MNK.<br> Templates decouple the mechanics <br>(layout, major-ness, instructions),<br> and the kernel still assumes<br> 128 threads and tile-multiple M and N.|
<pre>
Line 68: auto [M, N, K] = shape_MNK;
Line 69: Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));               
Line 70: Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));   
Line 71: Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC); 
  
L68: auto [M, N, K] = shape_MNK;
It is a C++17 structured binding that unpacks the CuTe tuple shape_MNK (built at L238 as
make_shape(M, N, K)). M, N, K are plain runtime ints (L241–243), so they are dynamic,
not Int<…> static constants.

mA / mB: Represents the full extent of A and B, ready to be sliced: local_tile at L76–77 tiles
them. But they are not views of memory:
mC (a pointer tensor): mC(m,n) is a reference to an actual half_t in global memory.
mA (a coordinate tensor): mA(m,k) is a TMA coordinate tuple, just integers. There is no pointer
inside mA, and indexing it never touches memory.
The TMA instruction takes descriptor + coordinates, not addresses, so the kernel-side
representation of A only needs coordinates.

tma_a.get_tma_tensor(...):
tma_a is the Copy_Atom. It contains the 128-byte descriptor plus CuTe’s aux params. It’s kernel
param space, implemented as a constant bank.
The descriptor contains the base address, the strides in bytes (not dA itself, which is in 
elements, and only for dims ≥1), the box dims, and the swizzle mode.

get_tma_tensor(make_shape(M,K)) builds a coordinate tensor of shape (M,K) from the atom’s aux 
params. Each element is the TMA coordinate of that position. Sliced tiles of it (tAgA(_,k_tile)) 
provide the box coordinates the TMA instruction needs. The address of A lives only in the 
descriptor.

make_gmem_ptr(C)
make_gmem_ptr(C) wraps the raw half_t* in CuTe’s gmem_ptr<…> type. It is a compile-time tag 
with no runtime cost. print(mC) would show the iterator as a gmem pointer.




make_tensor(..., make_shape(M,N), dC)
dC is the stride tuple (_1, ldC), counted in elements, not a “physical stride.” The epilogue
writes with ordinary per-thread stores, not TMA, so a real address-producing tensor is needed.

</pre>
||	mA / mB|	mC|
|:--|:----|:----|
|Built from|	Atom aux params plus the shape|	Pointer C plus the shape<br> plus stride dC|
|Iterator|	Counting iterator (ArithTuple)|	gmem_ptr<half_t>|
|tensor(i,j) yields|	A TMA coordinate tuple (integers)|	A reference to a half_t in<br> global memory|
|Consumer|	TMA instruction (via tAgA), plus the descriptor|	Threads’ ordinary loads<br> and stores (axpby)|
|Where the address lives|	Only inside the descriptor|	In the tensor itself|
<pre>
Summary
</pre>
|claim	|Verdict|
|:-----|:-------|
|mA/mB represent the full<br> A and B, ready to slice|yes, extent and slicing. they’re<br> coordinate tensors with no pointer.|
|tma_a is the 128-byte descriptor| It’s the atom containing the descriptor<br> plus aux params, in param space.|
|The descriptor contains A, dA, and the swizzle|	Address, byte strides (dims ≥1), <br>box dims, and swizzle mode.|
get_tma_tensor doesn't read the descriptor|It doesn’t read the opaque descriptor;<br> it uses the aux params plus the shape<br> to build coordinates. The address<br? stays in the descriptor.
|make_gmem_ptr attaches a<br> memory-space tag| yes.|
|mC is built from pointer,<br> shape, and stride because C<br> isn’t written with TMA.| dC is the stride tuple<br> in elements, not a “physical stride.”|
<pre>
Line 98:  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
          group_modes<0,2>(sA), group_modes<0,2>(gA)); 
Line 101:  auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
          group_modes<0,2>(sB), group_modes<0,2>(gB));  
These two statements are very important in NVIDIA's Hopper programming model. They connect TMA
(Tensor Memory Accelerator), CuTe tensor layouts, global memory, and shared memory.
The purpose of tma_partition() is to partition the global-memory and shared-memory tensors into
the portions that the TMA operation will transfer.
The first statement in detail. The second statement does exactly the same thing for matrix B.
auto [tAgA, tAsA] = tma_partition(
    tma_a,
    Int<0>{},
    Layout<_1>{},
    group_modes<0,2>(sA),
    group_modes<0,2>(gA)
);

The result contains two different tensor views:
- tAgA: the global-memory side of the TMA transfer.
- tAsA: the shared-memory side of the TMA transfer.
They describe corresponding portions of the source and destination tensors.
1. The data movement
Suppose matrix A is stored in GPU global memory, and I want to transfer a tile into Hopper
shared memory.
Global memory:
mA: represents the original, full matrix A in global memory.
gA: represents the portion of matrix A selected for this CTA, not the entire original matrix.
gA is constructed using local_tile():
Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
Tensor gA = local_tile(
    mA,
    cta_tiler,
    cta_coord,
    Step<_1, X, _1>{}
);
For example, if the CTA tile has dimensions 128 x 64, gA represents this CTA's assigned 
128-row block, with its K tiles available for iteration.
More precisely, its logical shape is: gA=(128,64,k)
where k represents the number of K tiles assigned to this CTA.
Tensor	      description
mA	          Full matrix A in global memory
gA	          This CTA's selected portion of matrix A in global memory
sA	          Shared-memory layout for the CTA's A operand tiles
tAgA	        TMA-partitioned global-memory source view
tAsA	        TMA-partitioned shared-memory destination view

Additional distinction: gA is a tensor view of the selected region, not a separate physical copy
of the matrix data.
      
Large matrix stored in device memory
TMA transfer
Uses the TMA tensor descriptor
Shared memory:
sA — Swizzled shared-memory layout
Tile storage organized for WGMMA
The TMA operation moves the data. tma_partition() prepares the corresponding tensor views 
that describe the transfer.
It is important to distinguish preparation from execution: calling tma_partition() does not
itself transfer matrix data.
2. The C++ structured binding
First, look at this part:
auto [tAgA, tAsA] = ...
This is C++17 structured binding syntax.
The function tma_partition() returns a pair of tensor objects, and C++ assigns them to two
variables.
For this CUTLASS implementation, the order is:
tAgA = global-memory partition;
tAsA = shared-memory partition;
The naming convention is useful:
Variable	Meaning
t	Tiled or partitioned tensor
A	Matrix operand A
g	Global memory
s	Shared memory

So:
- tAgA: partitioned global-memory tensor for A.
- tAsA: partitioned shared-memory tensor for A.
I should remember this because some newer CuTe DSL examples use a different return ordering.
In this C++ tutorial, follow the actual source code and its comments. 
3. Understand the first argument: tma_a
tma_a"
This is the TMA copy atom created earlier, using:
Copy_Atom tmaA = make_tma_atom(
    SM90_TMA_LOAD{},
    mA,
    sA(_,_,0),
    make_shape(bM,bK)
);
Understand what this does.
- SM90_TMA_LOAD{} specifies a Hopper TMA global-to-shared-memory load.
- mA describes matrix A in global memory.
- sA(_,_,0) supplies the shared-memory layout for one pipeline stage.
- make_shape(bM,bK) specifies the tile transferred by the TMA operation.
The resulting tma_a contains the information needed to perform the TMA operation, including its
tensor-map descriptor and copy configuration.
In other words, tma_a tells CuTe and the hardware how to transfer a tile of A from global memory
into shared memory.
4. Understand the second argument: Int<0>{}
Int<0>{}:
This is a compile-time integer value of zero.
In this particular TMA partitioning operation.
In this tma_partition() call:
auto [tAgA, tAsA] =
    tma_partition(tma_a, Int<0>{}, Layout<_1>{},
    group_modes<0,2>(sA), group_modes<0,2>(gA));
Argument	     meaning
Int<0>{}	     The CTA's rank within the multicast group
Layout<_1>{}	 The layout describing the multicast group
The tutorial uses: 
dim3 dimCluster(2,1,1); 
That creates a cluster containing two CTAs, while the multicast group described by Layout<_1>{}
contains one CTA. Therefore, the TMA transfer is not multicast to the other CTA in the cluster.
In other words:
- CTA cluster: The two-CTA execution group.
- Multicast group: The group of CTAs that receive the same TMA transfer.
- Int<0>{}: The calling CTA's rank in that multicast group.
- Layout<_1>{}: The multicast-group layout, not the cluster layout.

  
  

5. Understand Layout<_1>{}
Layout<_1>{} is the layout of the multicast group: the CTAs that receive the same TMA load. 
It has size 1, so each CTA loads its own tile and nothing is shared. Together with Int<0>{},
this CTA’s rank within that group, it says “no multicast.” This must match the host side, where
make_tma_atom was built without a multicast cluster size. The 2-CTA cluster launched at L282 is
unrelated to this layout.
Where this comes from
The distinction between the two concepts rests on two things in the file: the dimCluster(2,1,1)
launch at L282, and the tutorial comment at L92–93. The statement that these arguments describe 
the multicast group, not the cluster, is my recollection of CuTe’s tma_partition 
(cute/atom/copy_traits_sm90_tma.hpp) and of how CUTLASS’s SM90 mainloops use it. Open the
function’s definition in the CUTLASS checkout and look at what it does with cta_layout and
cta_coord. I expect only the multicast offset computation, and nothing about cluster launch
geometry.

Summary
Layout<_1>{} is a multicast-group layout of size 1, not a cluster layout.
Int<0>{} is this CTA’s rank in that group, not its CTA index. Every CTA passes 0.
Together they mean no multicast.
Rule: cluster = how the grid is launched; multicast group = which CTAs share one TMA load. 
Both tma_partition arguments describe the second.
  
Layout<_1>{} describes the multicast-group layout … not the layout of the entire CTA cluster.
Size = 1, only one CTA in the group, rank = 0.
Each cluster contains two CTAs (L282). A size-1 multicast group doesn’t change the cluster size.
The TMA transfer is not shared with another CTA through multicast, and this is the real
consequence.
Int<0>{} is the CTA’s rank within the multicast group.
The remember-this sentence:
Cluster size and multicast-group size are two different quantities. A cluster can contain two
CTAs while a particular TMA transfer uses a multicast group of size one.
  
A multicast group containing one CTA vs. the cluster: to make the relationship precise, a
multicast group is a set of CTAs within a cluster. Here the group is size 1 in a cluster of 2.
In a multicast kernel, the group would be some subset of the cluster, or all of it.
The 0 is the same in every CTA. Both CTAs in a cluster pass Int<0>{}, because each is alone in
its own group. Rank 0 does not mean the first CTA of the cluster.
The multicast setting has to match the host side. make_tma_atom was called without a multicast
cluster size (default 1), and the kernel’s Layout<_1> has to agree with that.

Summary
The cluster is a launch property (2 CTAs here); the multicast group is a TMA property (1 CTA here).
Int<0> is the rank in the multicast group, and every CTA passes 0.
The multicast group sits inside the cluster, and the kernel’s Layout<_1> must match the host
atom.

multicast group and  cluster:
In NVIDIA Hopper (SM90), a CTA cluster and a multicast group are related, but they describe two
different things.

The simplest way to distinguish them is:
- CTA cluster: Defines which CTAs are grouped together for cooperative execution.
- Multicast group: Defines which CTAs receive the same data through a single multicast TMA
transfer.
Let's examine them step by step.
1. CTA cluster
A CTA (Cooperative Thread Array) is essentially a CUDA thread block.

A CTA cluster groups multiple CTAs together so they can cooperate. Hopper introduced thread-
block clusters as a hardware-supported programming model.

For example, the tutorial uses:
dim3 dimCluster(2, 1, 1);
This means: 2 x 1 \times 1 = 2  CTAs per cluster

One CTA cluster
CTA 0                         CTA 1
Thread block 0               Thread block 1
  
Both CTAs belong to the same cluster.
CTAs in a cluster can use cluster-level synchronization and access each other's shared memory
through distributed shared memory mechanisms.

Importantly, belonging to the same cluster does not automatically mean they receive the same TMA
data.
2. Multicast group
A multicast group is associated with a particular TMA transfer.
It determines which CTAs in the cluster participate as recipients of the same transferred data.
For example, suppose the cluster contains two CTAs:
Case A: No multicast
Multicast group 0
One recipient
CTA 0

CTA 1
Not included in this transfer's multicast group

The TMA transfer targets CTA 0 only.

Case B: Multicast
One multicast group
CTA 0

CTA 1
Both CTAs receive the same TMA transfer.

Multicast can reduce redundant data movement when multiple CTAs need the same data.
3. How they relate in your tutorial
Your tutorial has:
dim3 dimCluster(2, 1, 1);
Therefore: Cluster size=2
But the TMA partition uses:
Int<0>{}, Layout<_1>{}

The multicast-group configuration has size one:
Multicast group size=1

So:
Concept	Value	Meaning
CTA cluster size	2	Two CTAs belong to the cluster
Multicast-group size	1	Only one CTA participates in this TMA multicast group
Multicast	Disabled for this transfer	No data distribution to multiple CTAs
One important clarification: The multicast group is not another independent hardware execution
group like a CTA cluster. It is a recipient configuration for a particular multicast operation, mapped within the cluster.
4. The key takeaway
Think of it this way:
- A cluster answers: Which CTAs are grouped together and can cooperate?
- A multicast group answers: Which of those CTAs should receive this particular TMA transfer?
A cluster of two CTAs can have a multicast group of one or two CTAs, depending on the TMA
operation.
That distinction is essential when studying Hopper TMA, especially when developer later examine
multicast barriers and shared-memory synchronization.








  
  
6. Understand group_modes<0,2>(sA)
group_modes<0,2>(sA)
That line is in gemm_tn. The default run (transA='N', transB='T') goes through gemm_nt, which
uses Layout_MN_SW128_Atom.
The sA that group_modes receives is the kernel’s smem tensor (L83), built from that layout. It
is not the host-side layout.
The shape stays ((128,64),3), which is hierarchical. What equals 8192 is size<0>, not the shape.
Grouping does not rearrange or copy data.
gA is ((128,64),k) and sA is ((128,64),3). Using k = number of k-tiles (16 here).

Layout<_1>{} is the layout of the multicast group: the CTAs that receive the same TMA load. 
It has size 1, so each CTA loads its own tile and nothing is shared. Together with Int<0>{},
this CTA’s rank within that group, it says “no multicast.” This must match the host side, where
make_tma_atom was built without a multicast cluster size. The 2-CTA cluster launched at L282 is
unrelated to this layout.

Where this comes from
The distinction between the two concepts rests on two things in the file: the dimCluster(2,1,1) 
launch at L282, and the tutorial comment at L92–93. The statement that these arguments describe the multicast group, not the cluster, is my recollection of CuTe’s tma_partition
(cute/atom/copy_traits_sm90_tma.hpp) and of how CUTLASS’s SM90 mainloops use it. I haven’t re-
read the source for this answer. Ppen the function’s definition in the CUTLASS checkout and look
at what it does with cta_layout and cta_coord. I expect only the multicast offset computation,
and nothing about cluster launch geometry.

Summary
Layout<_1>{} is a multicast-group layout of size 1, not a cluster layout.
Int<0>{} is this CTA’s rank in that group, not its CTA index. Every CTA passes 0.
Together they mean no multicast.
Rule: cluster = how the grid is launched; multicast group = which CTAs share one TMA load. 
Both tma_partition arguments describe the second.






  
  
    GMMA::Layout_K_SW128_Atom<TA>{},
    make_shape(bM,bK,bP)
);
Its logical shape is:
sA=(bM,bK,bP)
For the tutorial example:
sA=(128,64,3)
These three dimensions represent:
- M: rows of the matrix tile.
- K: columns of the matrix tile.
- P: pipeline stages.
What does group_modes<0,2> do?
It groups modes starting at index 0 and ending before index 2.
In other words, it combines the first two dimensions into one dimension.
Before:
(128,64,3)
After:
((128,64),3)
Conceptually:
Before grouping
M        K         p
128      64        3
  
After grouping
M × K                                    p
8192                                     3
One complete matrix tile                 Pipeline stages

Mathematically: 128 x 64=8192
The grouped tensor now has two modes: ((8192),3)

Why do we group M and K?
Because TMA is responsible for transferring the entire matrix tile in mode 0. We want the TMA
operation to handle the complete 128 × 64 tile as one transfer, while the pipeline stage remains
a separate dimension.
Notice that grouping modes does not physically rearrange or copy the data. It changes the
logical organization of the tensor view.
The same operation is performed on the global-memory tensor:
group_modes<0,2>(gA)
7. Understand group_modes<0,2>(gA)
Earlier in the tutorial, the global-memory tensor is constructed as follows
auto [M, N, K] = shape_MNK;
Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
Tensor gA = local_tile(
  mA, 
  cta_tiler, 
  cta_coord, 
  Step<_1, X,_1>{}
);  // (BLK_M,BLK_K,k)
- mA represents the full matrix A in global memory.
- local_tile() selects the matrix tiles assigned to this CTA.
- gA represents the selected tiles across the K iteration dimension.
Its shape is: gA=(bM,bK,k)
lowercase k represents the number of K tiles, not the K dimension within an individual tile.
After grouping:
group_modes<0,2>(gA)
the shape becomes: ((bM,bK),k)
For example: ((128,64),k)

Compare the two grouped tensors:
Tensor	Before grouping	       After grouping
sA	    (128,64,3)	           ((128,64),3)
gA	    (128,64,k)	           ((128,64),k)
The distinction:
- The shared-memory tensor has 3 pipeline stages.
- The global-memory tensor has however many K tiles must be processed.
The first grouped mode represents the matrix tile on both sides. The second mode represents the 
iteration or storage stage.
8. tma_partition() actually does
Reorders and offsets mode 0 per the atom and multicast configuration. 
The shapes (TMA,k) and (TMA,PIPE) and “TMA is not another matrix dimension” are also correct.
  
tma_partition(
    tma_a,
    Int<0>{},
    Layout<_1>{},
    group_modes<0,2>(sA),
    group_modes<0,2>(gA)
);
Argument	             Purpose
tma_a	                 Defines the TMA transfer operation
Int<0>{}	             CTA cluster coordinate
Layout<_1>{}           CTA cluster layout; one CTA, no multicast
group_modes<0,2>(sA)   Grouped shared-memory destination view
group_modes<0,2>(gA)	 Grouped global-memory source view

tma_partition() then performs a specialized partitioning for the TMA operation.
It reorders and offsets the first mode according to the TMA atom's requirements and the
multicast configuration.

The result is:
auto [tAgA, tAsA] = ...;
The resulting shapes are documented in the tutorial: 
tAgA=(TMA,k)
tAsA=(TMA,PIPE)
TMA represents the TMA-managed tile portion. It is not another matrix dimension.
TMA handles the entire first mode, and the program selects the appropriate global tile and
shared-memory pipeline stage using the second mode. 
Reorders and offsets mode 0 per the atom and multicast configuration. The shapes (TMA,k) and (TMA,PIPE) and TMA is not another matrix dimension.

9. See how tAgA and tAsA are actually used
In the tutorial, I find:
copy(
    tma_a.with(producer_mbar[pipe]),
    tAgA(_, k_tile),
    tAsA(_, pipe)
);
A. tAgA(_, k_tile)
This selects a particular K tile from global memory.
Suppose: k_tile = 0;
Then: tAgA(_, 0)
means:
- Select global-memory tile number 0.
- Keep the entire TMA-managed portion, represented by _.
If:
k_tile = 1;
it selects the next tile along K.
B. tAsA(_, pipe)
This selects a particular shared-memory pipeline stage.
For example: pipe = 0;
selects shared-memory stage 0.
If: pipe = 1;
it selects stage 1.
C. The complete copy
copy(
    tma_a.with(producer_mbar[pipe]),
    tAgA(_, k_tile),
    tAsA(_, pipe)
);
This tells TMA:
Transfer the selected matrix tile from global memory into the selected shared-memory pipeline 
stage, and use the specified producer barrier to signal completion.
The TMA transfer is asynchronous. The CPU-like CUDA thread issuing the operation does not need
to copy each element individually.
  
10. Visualize the three pipeline stages
Suppose we have three shared-memory stages:
Global-memory K tiles
0            1            2          3         4         5
K tile       K tile       K tile    K tile     K tile    K tile

Shared-memory pipeline
Stage 0  Shared memory       K tile 0
sA(:,:,0)

Stage 1  Shared memory       K tile 1
sA(:,:,1)



Stage 2  Shared memory  K tile 2
sA(:,:,2)

Illustration of the initial loading sequence. During the main loop, the three stages are reused
in a circular pipeline.
The program can load a subsequent tile while WGMMA consumes a previously loaded tile.
This is one of the important performance advantages of Hopper's asynchronous TMA and WGMMA
operations.
11. What about matrix B?
The second statement:
auto [tBgB, tBsB] = tma_partition(
    tma_b,
    Int<0>{},
    Layout<_1>{},
    group_modes<0,2>(sB),
    group_modes<0,2>(gB)
);
Does exactly the same thing for matrix B.
The difference is the logical dimensions:
Matrix A	         Matrix B
sA                 sB
gA	               gB
tma_a	             tma_b
tAgA	             tBgB
tAsA	             tBsB

For a conventional GEMM:
C_MxN = (A_MxK) x B_KxN
This tutorial uses a transposed representation for B:
B_NxK
That is why its B tile is organized using N and K rather than K and N.
12. One more connection: TMA transaction bytes
Immediately after the partitioning statements, the tutorial calculates:
Line 105:  // The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
constexpr int tma_transaction_bytes =
    sizeof(make_tensor_like(tensor<0>(tAsA)))
  + sizeof(make_tensor_like(tensor<0>(tBsB 

Why?  
Because TMA is asynchronous, and the program needs to know how many bytes are expected to 
arrive before the producer barrier can signal completion.
The expression: tensor<0>(tAsA)
selects the first mode of the shared-memory partition, which represents the complete 
TMA-managed tile.
make_tensor_like() creates a corresponding tensor representation, and sizeof() calculates its 
storage size.
The sizes for A and B are added together because the producer barrier tracks both transfers.
Later:
Line 144: // Set expected Tx Bytes after each reset / init
ProducerBarType::arrive_and_expect_tx(
    &producer_mbar[pipe],
    tma_transaction_bytes // sets the expected transaction size.
);
This connects tma_partition() directly to Hopper's asynchronous synchronization mechanism.

13. Summary: the complete execution flow
TMA data preparation and transfer
1. Create global-memory tensors
gA, gB are this CTA’s tiles of A, B, expressed as global TMA coordinates. Each element is the
global position in A, B that the tile element corresponds to. It does not contain the tile’s
data, and it is not a pointer.
Here use gA as explanation.
Here is the shape of it. gA has shape (128, 64, 16) in the default run:
Mode 0 (128): rows within a tile.
Mode 1 (64): columns within a tile.
Mode 2 (16): the k-tiles, one tile per step along K.
In the simple case where the TMA dimensions are just A’s (M,K) modes, the element for CTA
blockIdx.x = b is:
gA(i, j, k) = (b·128 + i,  k·64 + j)
That is a coordinate in the whole matrix A, not a position within the tile. “Global” means the 
coordinate is measured from the origin of A, so the b·128 and k·64 offsets are already included. 
(For this NT configuration the real TMA dimension order may differ from plain (M,K). Printing gA
would show it.)
Two points to keep straight:
Coordinates, not addresses. The TMA hardware turns a coordinate into a memory address using the
descriptor’s base address and strides.
Why “tile coordinates” matters. tAgA(_, k_tile) later hands the TMA instruction the first 
coordinate of that tile, the box origin.

2. Create shared-memory tensors
   sA, sB describe the shared-memory buffers and their swizzled layouts.
3. Group tensor modes
   group_modes<0,2>() combines the two matrix dimensions into one tile mode, leaving the K 
  iteration or pipeline stage separate.
4. Partition for TMA
   tma_partition() creates the corresponding global and shared-memory views.
5. Calculate expected transaction bytes
   The program determines how many bytes the TMA operations will transfer.
6. Issue asynchronous TMA copies
   copy() transfers global-memory tiles into shared-memory pipeline stages.
7. Synchronize and execute WGMMA
   The producer barrier signals when TMA has finished, allowing WGMMA to consume the 
   shared-memory tiles.

Summary:
- tma_partition prepares, doesn’t transfer
- Return order (gmem, smem)	Correct
- Int<0> is Rank in the multicast group
- Layout<_1> is Multicast-group layout; the cluster is 2 CTAs
- sA built with NT uses Layout_MN_SW128; it’s also the kernel’s smem tensor, not the host layout
((8192),3) after grouping	Wrong. Shape is ((128,64),3); 8192 is size<0>
- tAgA/tAsA shapes (TMA,k), (TMA,PIPE)	
- tAgA(_,k_tile) selects a global tile	Selects that tile’s coordinates
Pipeline overlap;  refills wait for the MMA on that stage
- B is N×K
- Transaction-byte logic

  

  
The relationship between these three statements:
auto [tAgA, tAsA] = tma_partition(...);

copy(tma_a.with(producer_mbar[pipe]),
     tAgA(_, k_tile),
     tAsA(_, pipe));

ProducerBarType::wait(
    &producer_mbar[read_pipe],
    read_state.phase()
);
     
Together, they demonstrate how Hopper moves data asynchronously from global memory to shared
memory and synchronizes that transfer with WGMMA.

The key concept I should remember is:
tma_partition() describes the transfer, copy() launches the transfer, and the producer barrier
tells the consumer when the transfer is complete.

Line 105:
constexpr int tma_transaction_bytes =
    sizeof(make_tensor_like(tensor<0>(tAsA)))
  + sizeof(make_tensor_like(tensor<0>(tBsB)));
 
This statement calculates the combined byte count for mode 0 of tAsA and mode 0 of tBsB. It calculates the number of bytes expected from the 
two TMA loads for one pipeline stage. It does not calculate the total shared memory allocation for tAsA and tBsB across all pipeline stages.
Earlier in the kernel:
auto [tAgA, tAsA] = tma_partition(...); // (TMA, k) and (TMA, PIPE)
auto [tBgB, tBsB] = tma_partition(...); // (TMA,k) and (TMA,PIPE) 
The relevant destination tensor shapes are:
Tensor	Mode 0	                            Mode 1
tAsA	Elements transferred by TMA for A	Pipeline stage
tBsB	Elements transferred by TMA for B	Pipeline stage

Therefore:
- tensor<0>(tAsA) selects the mode-0 tensor for A.
- tensor<0>(tBsB) selects the mode-0 tensor for B.
- The pipeline dimension is excluded from the byte calculation.
TMA copies all elements in mode 0.
The count is in bytes because the mbarrier’s transaction count counts bytes [PTX docs, mbarrier].
It can be constexpr because the smem layouts are fully static, which the static_asserts at L53–54 enforce [your file].
  
2. tensor<0>, make_tensor_like, and sizeof
Consider:
sizeof(make_tensor_like(tensor<0>(tAsA)))

It works in three steps:
1. tensor<0>(tAsA) constructs a tensor view using the layout of mode 0 of tAsA.
2. make_tensor_like(...) creates an Owning, array-backed tensor with the same element type and shape, and a compact layout, strides ordered 
like the source, no swizzle copied. 
Owning, array-backed: the new tensor stores its own elements in a fixed-size array.
  
3. sizeof(...) obtains the C++ storage size of that newly constructed tensor type, in bytes.
Important: sizeof is not measuring the original shared-memory tensor view, which contains pointer and layout information. Instead, 
make_tensor_like provides an owning tensor whose storage size can be used to calculate the transfer byte count. This behavior follows the CuTe
implementation. 

3. Calculate the actual byte count for the default tutorial
The default gemm_nt configuration specifies:
auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int<64>{};
auto bP = Int<3>{};

The elements of both A and B have type cute::half_t, which occupies 2 bytes. 

For one pipeline stage:
Tensor	        Elements        	Bytes
tensor<0>(tAsA)	128 × 64 = 8,192	16,384
tensor<0>(tBsB)	128 × 64 = 8,192	16,384
Total	16,384	32,768

Therefore:
tma_transaction_bytes=32768 bytes=32 KiB
Because bP = 3, the A and B pipeline buffers together hold 96 KiB of tile data across three stages, excluding barriers and other storage overhead.
  
4. Why does the kernel need this value?
Look at the following code:
if ((warp_idx == 0) && lane_predicate) {   // L141: one elected thread
  ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);  // L144
  copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));                 // L145
  copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));                 // L146
}
Only one thread, warp 0 and the lane picked by elect_one_sync(), runs these three calls. The barrier was initialized with arrival count 1
(L130), so exactly one arrive_and_expect_tx per phase is correct.  

The kernel uses one transaction barrier for both TMA copies in a pipeline stage.
Before issuing the transfers, arrive_and_expect_tx() registers the expected number of transferred bytes:
The barrier is initialized only once (L130, arrival count 1). The expected bytes are set again each time the stage is reused.
16384(A)+16384(B)=32768 bytes

The barrier's completion condition accounts for both the software arrival and the completion of the expected TMA transactions. 
The consumer can then safely access the data after the appropriate barrier wait. 
So the most precise description of the statement is:
  
tma_transaction_bytes is the combined byte count of the mode-0 TMA destination tiles of tAsA and tBsB, for one pipeline stage. It is passed to
arrive_and_expect_tx each time a stage is filled, in the prefetch (L144) and in the mainloop (L216). That tells the barrier how many bytes to 
wait for in that phase.
It is not the total size of all pipeline stages, nor the number of bytes transferred over the entire GEMM execution.

tma_transaction_bytes is the number of bytes one stage’s two TMA loads will deliver. The barrier waits for that many bytes to arrive before 
it lets the consumer proceed.

Two things happen in one call. arrive_and_expect_tx counts the barrier’s one arrival and adds the expected bytes. I believe it maps to
mbarrier.arrive.expect_tx [CUTLASS barrier.h]. The barrier was initialized with arrival count 1 (L130). A phase completes only when pending
arrivals and pending transaction bytes both reach zero [PTX docs]. software arrival and the completion of the expected TMA transactions.
It is re-armed every phase, not configured once. L130 sets only the arrival count. The byte count is set again each time a stage is filled: 
L144 in the prefetch and L216 in the mainloop. That’s what the comment “after each reset / init” means (L143, L215).
Only the elected thread does it. The calls sit inside if ((warp_idx == 0) && lane_predicate) (L141, L210). That matches the arrival count of 1. 
If all 128 threads called arrive_and_expect_tx, the arrivals and the expected bytes would be wrong.

Consequences of a wrong value: too large, and the phase never completes, so the kernel hangs. Too small, and the phase can complete before all
bytes land [follows from the PTX completion rule].

Summary
tensor<0> selects mode 0, excluding pipe (it stays a view on the same smem pointer)
make_tensor_like makes an owning, array-backed; never instantiated, since sizeof doesn’t evaluate it
sizeof is not measuring the original view	
16,384 B + 16,384 B = 32,768 B = 32 KiB; 96 KiB over 3 stages (actual smem_size is about 98,432 B)
One barrier covers both copies
arrive_and_expect_tx registers the expected bytes, and it also counts the one arrival

As a check on the last note, the whole-GEMM total per CTA in the default run would be 16 k-tiles × 32,768 B = 524,288 B.

L185–223:
auto write_state = cutlass::PipelineState<K_PIPE_MAX>();          
auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();           
  CUTE_NO_UNROLL
  while (k_tile_count > -K_PIPE_MAX)
  {
    int read_pipe = read_state.index();
    ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());
    warpgroup_arrive();
    gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);   
    warpgroup_commit_batch();

    warpgroup_wait<0>();
    ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
    ++read_state;

    if ((warp_idx == 0) && lane_predicate && (k_tile_count > 0))
    {
      int pipe = write_state.index();
      ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
      ++write_state;
    }
    --k_tile_count;
    ++k_tile;
  }


TMA must not overwrite shared-memory data while WGMMA is still using it, and WGMMA must not read data before TMA finishes loading it.
1. Understand the pipeline's shared-memory organization
For the default gemm_nt path, the host-side configuration includes:
auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int<64>{};
auto bP = Int<3>{};
The shared-memory layouts use GMMA::Layout_MN_SW128_Atom, and the kernel constructs the shared-memory tensors sA and sB from those layouts.
The destination tensors produced by tma_partition() have two logical modes:
tAsA  // (TMA, PIPE)
tBsB  // (TMA, PIPE)
PIPE has size three. Each pipeline stage contains space for one A tile and one B tile.
Pipe 0           Pipe 1         Pipe 2
A                A              A
128 × 64         128 × 64       128 × 64

B                B              B
128 × 64         128 × 64       128 × 64
Three reusable pipeline stages. Each stage has a different shared-memory region for A and B.
That's 32 KiB of operand data per stage, or 96 KiB across the three stages, excluding other shared-memory objects.
The pipeline does not allocate more shared memory every time it processes another K tile. It reuses these same three stages.
  
2. PipelineState<K_PIPE_MAX>
auto write_state = cutlass::PipelineState<K_PIPE_MAX>();
auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();
These create two independent pipeline-state trackers.
- write_state: Tracks which pipeline stage the next TMA copy should fill.
- read_state: Tracks which pipeline stage WGMMA should consume next.
A PipelineState contains two important values:
Member	     Purpose
.index()	   Selects a physical pipeline slot: 0, 1, or 2
.phase()	   Identifies the barrier generation associated with the current use of that slot

Initially, both states are:
index = 0;
phase = 0;
Number of advances	index()	phase()
0                 	0	      0
1	                  1	      0
2	                  2	      0
3	                  0	      1
4	                  1	      1
5	                  2	      1
6                 	0	      0
7                 	1	      0

Notice that the index wraps every three advances and the phase toggles at each wrap.
phase() provides the expected barrier-phase parity. Initially it is 0. When the hardware barrier completes phase 0, its internal phase advances 
to 1, signaling that phase 0 has completed. The software PipelineState advances its own phase when the circular buffer index wraps around.

There are three producer barriers, one per stage. Each has its own phase, starting at 0. When producer_mbar[0] completes phase 0 it moves to 
phase 1. That says nothing about producer_mbar[1] or [2].
1. Example: Three stages, all initially at phase 0
Suppose the three TMA transfers are started:
Pipeline stage	TMA operation	Barrier    phase
Stage 0	        Loading K tile 0	       0
Stage 1	        Loading K tile 1	       0
Stage 2	        Loading K tile 2	       0
Now suppose TMA completes the transfer for stage 0 first.
The barrier phases become:
Pipeline stage	TMA status	        Current hardware barrier phase
Stage 0	        Completed	          1
Stage 1       	Still loading	      0
Stage 2	        Still loading     	0
Only producer_mbar[0] advances from phase 0 to phase 1.
The other two barriers remain at phase 0 until their own completion conditions are satisfied.
Thus:
ProducerBarType::wait(&producer_mbar[0], 0);
can finish even if the TMA transfers for stages 1 and 2 are still running.
2. The reason read_state.phase() remain 0 for stages 0, 1, and 2:
Consider:
ProducerBarType::wait(
    &producer_mbar[read_state.index()],
    read_state.phase()
);

The read_state advances in this order:
K tile	read_state.index()	read_state.phase()	Barrier being checked
0	0	0	producer_mbar[0], phase 0
1	1	0	producer_mbar[1], phase 0
2	2	0	producer_mbar[2], phase 0
3	0	1	producer_mbar[0], phase 1
4	1	1	producer_mbar[1], phase 1
5	2	1	producer_mbar[2], phase 1
Notice that read_state.phase() changes from 0 to 1 after the software pipeline index wraps from 2 to 0.
It does not change simply because a hardware barrier completed.
3. One more important distinction
Even when producer_mbar[0] completes, that only means the TMA loads associated with that barrier have completed. It does not mean WGMMA has
finished consuming stage 0.

That is why the tutorial has another barrier:
ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
The producer barrier tracks when data is ready to consume; the consumer barrier tracks when the stage is safe to reuse.
The key point: Each of the three pipeline stages has its own producer barrier and consumer barrier, with independently advancing hardware phases.
read_state.phase() is a separate software value that selects the expected phase for whichever stage read_state.index() currently identifies.
The software pipeline phase and the hardware barrier phase are two separate things, although they work together.
TMA: a hardware copy engine in the SM
TMA: the hardware unit.
cp.async.bulk.tensor: the PTX instruction that commands it.
SM90_TMA_LOAD: the CuTe C++ type that wraps that instruction

  
TMA is a hardware feature introduced with NVIDIA Hopper (SM90) that performs asynchronous tensor data transfers.
There is an important distinction:
- TMA: The hardware mechanism that performs the transfers.
- TMA instruction: An instruction that tells the hardware to initiate a transfer.
For example, Hopper supports the PTX instruction:
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
This instruction initiates an asynchronous 2D tensor copy from global memory into shared memory, with completion tracked by an mbarrier. 
CUDA Core Compute
2. How TMA works in your CUTLASS tutorial
copy(tma_a.with(producer_mbar[pipe]),
     tAgA(_,k_tile),
     tAsA(_,pipe));

Conceptually:
copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe))  
  -> CuTe SM90_TMA_LOAD_<N>D::copy (inline PTX)                  
  -> PTX cp.async.bulk.tensor ... mbarrier::complete_tx::bytes
  -> ptxas lowers it to SASS
  -> TMA unit reads the descriptor (param space), fetches the box from gmem,
     writes it swizzled into smem
  -> as bytes land, the TMA unit performs complete_tx on producer_mbar[pipe]
    

TMA can perform the transfer asynchronously while GPU threads continue other work
3. How this relates to the two phases
In the code:
auto read_state = cutlass::PipelineState<K_PIPE_MAX>();
This is a software state object.
Its phase() value changes when ++read_state wraps the stage index back to zero.
However:
ProducerBarType::wait(
    &producer_mbar[read_pipe],
    read_state.phase());

uses that software phase as the expected parity for the hardware barrier.
The hardware barrier advances its own phase only when its synchronization requirements are satisfied, including the expected TMA transaction
completion.
Concept	Controlled by	When phase changes
Software read_state.phase()	CUTLASS PipelineState	When the stage index wraps around
Hardware producer_mbar[pipe] phase	GPU barrier mechanism	When that barrier's current phase completes

I should remember: TMA performs the data transfer. The hardware barrier tracks when that transfer is complete. The CUTLASS software phase tells
the waiting thread which barrier phase to wait for.
These are three related mechanisms, but they are not the same thing.

Before the while loop(Line 191): the prefetch prologue.
Line 191: while (k_tile_count > -K_PIPE_MAX)
This is essential to understanding the loop.
The source code already executed the following operations before the two pipeline states were created:
auto K_PIPE_MAX = size<1>(tAsA);
int k_tile_count = size<1>(tAgA);
int k_tile = 0;
  
Each CTA’s elected thread initializes the barriers (L127–133): arrival count 1 for each producer barrier, 128 for each consumer barrier. 
Then cluster_sync() (L135) makes every thread wait until all of them have finished initializing. Only after that does the elected thread issue 
the first TMA copies (L137–150):
for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
{
    if ((warp_idx == 0) && lane_predicate)
    {
        ProducerBarType::arrive_and_expect_tx(
            &producer_mbar[pipe], tma_transaction_bytes);

        copy(tma_a.with(producer_mbar[pipe]),
             tAgA(_,k_tile), tAsA(_,pipe));

        copy(tma_b.with(producer_mbar[pipe]),
             tBgB(_,k_tile), tBsB(_,pipe));
    }

    --k_tile_count;
    ++k_tile;
}
- The barriers are initialized by one elected thread per CTA (L129–131), but used by all threads.
- cluster_sync() (L135) is a stop-until-everyone-arrives point placed between init and first use.
Without it, a thread could use a barrier that isn’t initialized yet.
- The comment’s “all CTAs” reflects the cluster-wide pattern. In this tutorial the within-CTA ordering is what actually matters.
                          
For three pipeline stages, this initiates:
Pipeline stage    	K tile loaded	    Producer barrier
0                 	0	                producer_mbar[0]
1	                  1               	producer_mbar[1]
2	                  2               	producer_mbar[2]

The copies are asynchronous: the code initiates the transfers but does not immediately wait for all three to finish.
This is the pipeline's prologue. The while loop can immediately begin consuming the first prefetched stage while subsequent transfers are
progressing.
For the tutorial's default problem size, K=1024 and bK=64, so there are 16 K tiles.
After the prologue:
k_tile_count = 13;  // 16 - 3
k_tile       = 3;   // next global-memory tile to load

These values matter later, I can understand the unusual loop condition.
4. First half of the loop: Consume a completed TMA tile
Let's examine the consumer portion line by line.
int read_pipe = read_state.index();
ProducerBarType::wait(
    &producer_mbar[read_pipe],
    read_state.phase());

At the beginning of the first iteration:
read_state.index() = 0
read_state.phase() = 0

Therefore, the operation is equivalent to:
ProducerBarType::wait(&producer_mbar[0], 0);

This waits for the TMA transaction associated with stage 0, barrier phase 0 to complete.
Recall that the prologue initiated two TMA copies into stage 0:
copy(tma_a.with(producer_mbar[0]), ..., tAsA(_,0));
copy(tma_b.with(producer_mbar[0]), ..., tBsB(_,0));
Both copies are associated with the same producer barrier.
The ProducerBarType is a ClusterTransactionBarrier, which tracks both arrivals and expected transaction bytes. The barrier completes its 
phase once the required arrival and transaction-completion conditions have been satisfied. 

Why wait here? WGMMA cannot safely read an A or B tile until TMA has finished writing that tile into shared memory.
Execute WGMMA
warpgroup_arrive();

gemm(mma,
     tCrA(_,_,_,read_pipe),
     tCrB(_,_,_,read_pipe),
     tCrC);
warpgroup_commit_batch();
  
I separate these three operations.
warpgroup_arrive():
Despite its name, this is not an arrival on producer_mbar or consumer_mbar.
It issues the WGMMA fence operation, corresponding to wgmma.fence.sync.aligned, which establishes the required register-access ordering 
before WGMMA execution. 

gemm(...):
For the first iteration, it becomes:
gemm(mma,
     tCrA(_,_,_,0),
     tCrB(_,_,_,0),
     tCrC);
The three arguments have different roles:
Operand	Meaning
tCrA(_,_,_,0)	WGMMA shared-memory operand descriptors for A at pipeline stage 0
tCrB(_,_,_,0)	WGMMA shared-memory operand descriptors for B at pipeline stage 0
tCrC	Register-based accumulator fragment
The A and B operands are not ordinary register-resident matrix fragments. In this tutorial's shared/shared WGMMA configuration, 
they are descriptor-based views of shared-memory operands.
Conceptually, the operation updates:
C_acc <-- C_acc+(A_0)(B_0)
where A_0 and B_0 are the first K tiles.
For the default gemm_nt configuration, the physical matrix multiplication involves a 128 x 64 A tile and the corresponding 64 x 128 logical 
B operand.

warpgroup_commit_batch():
WGMMA execution is asynchronous. This function commits the WGMMA instructions issued since the preceding commit into a WGMMA group.
It does not mean the matrix multiplication has finished.
Wait for WGMMA to finish
warpgroup_wait<0>();
This is a critical synchronization point.
The 0 means that the executing warpgroup must wait until no previously committed WGMMA groups remain outstanding.
Only after this wait can the code safely treat WGMMA consumption of the shared-memory operands as complete. 
Next:
ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
This signals that the consumer thread has finished using this pipeline stage.
In the tutorial, each consumer barrier was initialized with an expected arrival count of 128:
ConsumerBarType::init(&consumer_mbar[pipe], 128);
All 128 participating threads execute the consumer arrival. Once all expected arrivals complete, the barrier phase completes, permitting the
producer to reuse that stage.
Finally:
++read_state;
The consumer advances to the next pipeline stage.
For the first iteration:
// Before
read_state = {index: 0, phase: 0};
// After
read_state = {index: 1, phase: 0};
Notice that read_state advances only after the current tile has been consumed.
5. Second half of the loop: Refill an available pipeline stage
Now examine the producer portion:
if ((warp_idx == 0) && 
  lane_predicate && 
  (k_tile_count > 0))

This condition has three parts.
Condition	Purpose
warp_idx == 0	Select warp 0
lane_predicate	Elect one thread within that warp
k_tile_count > 0	Ensure another K tile remains to be loaded
Thus, one elected thread from warp 0 issues the TMA transfers.
The rest of the CTA does not execute the producer block.
Determine the destination pipeline stage
int pipe = write_state.index();
Initially:
write_state.index() = 0
write_state.phase() = 0
Therefore:
pipe = 0;
Here is an important point: stage 0 currently holds the first prefetched tile, which WGMMA just consumed.
The next TMA transfer wants to overwrite that same stage with K tile 3.
Wait until the stage is available
ConsumerBarType::wait(
    &consumer_mbar[pipe],
    write_state.phase());

During the first iteration, this is equivalent to:
ConsumerBarType::wait(&consumer_mbar[0], 0);
It waits for the completion of the consumer barrier's phase 0.
This ensures that all 128 participating consumer threads have released stage 0 before its storage is reused.
The distinction between the two waits is essential:
- ProducerBarType::wait(...) prevents WGMMA from reading an incompletely loaded tile.
- ConsumerBarType::wait(...) prevents TMA from overwriting a tile before WGMMA has finished using it.
Register the next TMA transaction
ProducerBarType::arrive_and_expect_tx(
    &producer_mbar[pipe],
    tma_transaction_bytes);

arrive_and_expect_tx does two things. It adds 32,768 to the barrier’s expected transaction count, and it counts the one arrival 
(the barrier was initialized with arrival count 1 at L130). A phase completes only when pending arrivals and pending transaction bytes are 
both zero. That arrive_and_expect_tx maps to mbarrier.arrive.expect_tx is.
This registers the one arrival and the 32,768 expected bytes for the barrier’s currently pending phase. That phase completes only after 
the arrival has happened and both TMA copies have delivered all 32,768 bytes.

As established previously, for the default half-precision tile shapes:
tma_transaction_bytes=32768
The producer barrier will track the completion of the two new TMA transfers totaling 32 KiB (128x64x2x2).
Issue both TMA copies
copy(tma_a.with(producer_mbar[pipe]),
     tAgA(_,k_tile),
     tAsA(_,pipe));
copy(tma_b.with(producer_mbar[pipe]),
     tBgB(_,k_tile),
     tBsB(_,pipe));
For the first loop iteration:
pipe   = 0;
k_tile = 3;

Therefore, the code initiates:
copy(tma_a.with(producer_mbar[0]),
     tAgA(_,3),
     tAsA(_,0));
copy(tma_b.with(producer_mbar[0]),
     tBgB(_,3),
     tBsB(_,0));

This initiates the transfer of the fourth K tile into the shared-memory buffers previously occupied by K tile 0.
The copy() calls initiate asynchronous TMA operations. They do not block until the transfers complete.
Finally:
++write_state;
Advances the producer's stage and phase tracker:
// Before
write_state = {index: 0, phase: 0};
// After
write_state = {index: 1, phase: 0};
  
6. Visualize the pipeline over several iterations
To make the schedule easier to follow, suppose there are only five K tiles, numbered 0 through 4, with three pipeline stages.
This is a tracing example; the tutorial's default problem has 16 K tiles.
After the prologue, tiles 0, 1 and 2 are in flight in stages 0, 1 and 2. k_tile_count = 5 − 3 = 2, k_tile = 3, and both read_state and write_state
are {index 0, phase 0}.
</pre>
Iter	k_tile_count at start	WGMMA consumes (stage / phase / tile)	Producer block (needs k_tile_count > 0)	k_tile_count after
1	2	0 / 0 / tile 0	runs: tile 3 → stage 0	1
2	1	1 / 0 / tile 1	runs: tile 4 → stage 1	0
3	0	2 / 0 / tile 2	skipped	−1
4	−1	0 / 1 / tile 3	skipped	−2
5	−2	1 / 1 / tile 4	skipped	−3, loop exits
<pre>
The table follows the code’s logical order. The asynchronous TMA transfers may complete at different times; the barriers enforce the 
necessary dependencies.
Next 
The table follows the code's logical ordering. Actual asynchronous TMA transfers may complete at different times; barriers enforce the 
necessary dependencies.
The important pattern is:
Stage 0: Tile 0 → Tile 3 → Tile 6 → ...
Stage 1: Tile 1 → Tile 4 → Tile 7 → ...
Stage 2: Tile 2 → Tile 5 → Tile 8 → ...
The three stages form a circular buffer.

The producer wait (L195) and the consumer arrive (L206) are executed by all 128 threads, which is why the consumer barrier has arrival 
count 128. The refill block (L210–220) is executed by one elected thread. That thread is therefore both a consumer and the producer. 
This tutorial does not use warp specialization (comment at L179–182).
  
7. Why is the loop condition k_tile_count > -K_PIPE_MAX (Line 191)?
groups A and B by stage. That is a logical grouping. Physically, SharedStorage (L28–29) holds all three A stages in one array and all three 
B stages in another. Add “logical view” to the caption. I believe each stage is a contiguous block inside its array
  
This is probably the most unusual statement:
Line 191: while (k_tile_count > -K_PIPE_MAX)
You might expect:
while (k_tile_count > 0)
But that would be wrong for this particular pipeline.
The reason is that k_tile_count has already been reduced by the prologue.
If there are N total K tiles and P pipeline stages:
Initial k_tile_count=N-P
The main loop performs exactly one WGMMA consumption per iteration and decrements k_tile_count every time.
After N iterations:
k_tile_count=(N-P)-N=-P
Thus the stopping condition is:
k_tile_count=-K_PIPE_MAX

For the default configuration:
N=16, P=3
The counter starts at 13 and must count down to -3, giving 16 WGMMA-consumption iterations.
Why allow it to become negative?
Because k_tile_count has two roles:
- While positive, it indicates there are more global-memory tiles to issue.
- Once zero or negative, the loop continues draining prefetched or in-flight tiles through WGMMA.
This is why the producer block has a separate condition:
if (... && (k_tile_count > 0))

When no more tiles remain to be loaded, TMA issuing stops, but WGMMA consumption continues until every prefetched tile has been processed.
That final portion is called the pipeline drain or mainloop tail.
The counter update:
--k_tile_count;
++k_tile;
happens at the end of every iteration, even if no new TMA copy was issued. The resulting excess k_tile values are harmless because no source
tensor is accessed once the producer condition is false.
8. Understand the two kinds of barriers
The tutorial deliberately uses two separate barrier arrays.
</pre>
Barrier	               Who completes/signals it?             Who waits?          What it protects

producer_mbar[pipe]    TMA transaction completion,           WGMMA consumers     Prevents reading 
                       together with the registered arrival                      incomplete tiles
consumer_mbar[pipe]    All 128 consumer-thread arrivals      Elected TMA         Prevents overwriting tiles 
                                                             producer thread     still in use
<pre>
TMA producer
Global memory → shared memory
|
V Producer barrier: tile ready

WGMMA consumer
Read shared memory and accumulate
Consumer barrier: slot reusable

The arrows represent synchronization dependencies, not additional data copies.
Although both CUTLASS types have Cluster in their names, this particular loop uses barriers in each CTA's own shared memory. 
The 2-CTA launch cluster does not imply that the two CTAs share these particular pipeline stages or that TMA multicasts data between them.
  
9. Where does the performance improvement come from?
The advantage is that TMA and WGMMA are separate asynchronous mechanisms.
For example, after the first WGMMA consumes stage 0, the code can issue the next TMA transfer into stage 0.
While WGMMA later computes using stage 1, that TMA transfer can be progressing concurrently.
However, this tutorial is intentionally simple.
Because it executes:
warpgroup_wait<0>();
during every iteration, it waits for the current WGMMA group to complete before advancing to the next consumer stage. Consequently, it does 
not exploit the maximum possible overlap between multiple outstanding WGMMA groups.
The code demonstrates overlap between TMA loads and WGMMA computation, but more advanced CUTLASS kernels can pipeline WGMMA itself more 
aggressively.
  
10. Final execution sequence
Here is the whole loop expressed in plain language:
1. Find the next shared-memory stage to consume using read_state.index().
2. Wait until both TMA loads for that stage complete, using the producer barrier and the correct phase.
3. Issue the WGMMA operation using the A and B descriptors for that stage.
4. Commit the WGMMA group and wait for its completion.
5. Signal that the stage has been consumed, then advance read_state.
6. If additional K tiles remain, the elected producer thread waits for the next write stage to be released.
7. Register the expected transaction bytes, issue the two new TMA loads, and advance write_state.
8. Update the K counters and repeat until all prefetched and subsequently loaded tiles have been consumed.

The single most important concept to retain is that the pipeline has two independent synchronization directions:
TMA finished writing  ----> WGMMA may read

WGMMA finished reading ----> TMA may overwrite

read_state and write_state identify the slot and barrier generation for each direction. The barriers, rather than the state objects themselves,
enforce the safety of the overlapping execution.


</pre>

