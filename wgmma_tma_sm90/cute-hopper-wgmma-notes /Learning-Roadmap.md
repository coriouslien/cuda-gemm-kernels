<pre>
Line 271:  Tensor mA = make_tensor(A, make_shape(M,K), dA);
Line 272:   Tensor mB = make_tensor(B, make_shape(N,K), dB);
Line 271: Tensor mA = make_tensor(A, make_shape(M,K), dA);
mA is a CuTe Tensor object. It does not allocate or store data.
It’s a non-owning view: an iterator (here the raw pointer) plus a Layout (shape plus stride). 
Contrast this with make_tensor_like(...) at L104, which does create an owning, array-backed 
tensor.
The label “global tensor” is only a convention here. Because A is passed as a plain TA const*,
not wrapped in make_gmem_ptr(A), the tensor carries no gmem tag. CuTe doesn’t know it points to 
global memory, and doesn’t need to, since make_tma_atom only extracts the raw address. 
Compare L71, where mC uses make_gmem_ptr(C).
A is not the matrix object itself, A is the physical starting address.
A is the starting address of the matrix data, but it is a GPU virtual address, not a physical 
one. 
It comes from d_A.data().get() in main, which is device memory allocated by 
thrust::device_vector.
Two more facts from the pointer:
Its type TA const* (half_t const*) also gives the tensor its element type and makes it read-only.
It’s a device address being used inside a host function. That’s safe only because the host never 
dereferences it; it just reads the pointer value.
make_shape(M, K) defines the 2D logical bounds (Rows, Columns) ... domain of valid coordinates 
(i, j).
A is M×K, so mode 0 is M (rows) and mode 1 is K (columns), and valid coordinates are 0 ≤ i < M, 
0 ≤ j < K. M and K are plain int (L241–243), so the shape is dynamic, known only at runtime.
dA is not the leading dimension. It is already the full stride tuple, defined at L246 in gemm_nt:
auto dA = make_stride(Int<1>{}, ldA);   // (dM, dK)
The leading dimension is ldA, the function parameter, which is one component of dA.
So make_tensor doesn’t build (1, dA) from dA. It uses dA as-is, giving the layout (M,K):
(_1,ldA). 
Element (i,j) is at A + i*1 + j*ldA, and strides are counted in elements, not bytes.
If the matrix is column-major ...?
CuTe makes no such decision. The layout is whatever stride you pass, and the “column-major” 
choice was made by the programmer when writing L246 for the NT case. In CuTe terms this is 
M-major (M is the stride-1 mode).
The TN path (L314) passes dA = (ldA, _1) instead, which is K-major, and the same make_tensor
call  then describes row-major A. Same line of code, different stride, different layout.
Where the value comes from: main sets ldA = m when transA == 'N', so with the defaults, mA is 
(512,1024):(_1,512).
  
Line 272: Tensor mB = make_tensor(B, make_shape(N,K), dB);
The shape is (N,K), not (K,N). This code stores B as N×K and computes C(m,n) = 
Σ_k A(m,k)·B(n,k). That convention is why the kernel tiles B with Step<X,_1,_1> at L77
(selecting the N and K tile modes).
dB = (_1, ldB) (L247), so B is N-major in the NT path. main sets ldB = n when transB == 'T', so
with the defaults mB is (256,1024):(_1,256).
Summary:
Tensor as a view: correct. Note A is a raw pointer, so there’s no gmem tag, unlike mC.
A: the starting address, but a GPU virtual address, not physical. Its type also sets the element
type and const-ness.
Shape (M,K): correct. Dynamic ints defining the coordinate domain.
dA: not the leading dimension. It is the complete stride tuple (_1, ldA) built at L246, used 
unchanged, in units of elements. ldA is the leading dimension.
Major-ness: set by the stride the programmer passes (M-major here, K-major in TN), not inferred
by CuTe.
L272: same pattern, but B is shaped (N,K) with dB = (_1, ldB), so it is N-major.
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

  
Line 275:  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK))
tmaA is a Copy_Atom<Copy_Traits<SM90_TMA_LOAD, …>, …>, and its Copy_Traits (in CuTe’s
cute/atom/copy_traits_sm90_tma.hpp) holds two things:
The TmaDescriptor, a 128-byte CUtensorMap, which is the hardware descriptor;
aux params, mainly the mapping from gmem modes to TMA coordinate dimensions. These are CuTe-side
metadata, not hardware. get_tma_tensor uses them later.
So tmaA contains a hardware descriptor plus CuTe metadata. It is not purely a hardware
descriptor.

SM90_TMA_LOAD{}
It isn’t an empty tag. It has no data members, but it has static copy(...) member functions.
These dispatch to SM90_TMA_LOAD_1D … _5D (in cute/arch/copy_sm90_tma.hpp) depending on the
descriptor’s rank. Each of those contains the inline PTX cp.async.bulk.tensor.
<N>d.shared::cluster.global.mbarrier::complete_tx::bytes.
It produces PTX, not SASS. ptxas then lowers that PTX to SASS. “It selects the PTX instruction”
is the accurate level.

It also matters on the host: the type selects which Copy_Traits specialization make_tma_atom
builds. So it determines how the descriptor is constructed, not just which device instruction
runs later.

mA (the source)
It supplies the global bounds and strides, and the base address and the element type. All of it
is consumed on the host to encode the descriptor. At runtime the TMA unit never sees mA; it sees
only the descriptor.

sA(_,_,0): sA is a Layout, not a tensor. It was defined at L257:
auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
That is a swizzled ComposedLayout of shape (128,64,3), with no pointer. The smem tensor sA in
the kernel (L83) is a different variable that happens to share the name. So sA(_,_,0) is a 
layout slice: the (128,64) layout of one stage.
make_tma_atom( SM90_TMA_LOAD{},   // argument 1
               mA,                // argument 2
               sA(_,_,0),         // argument 3  ← this whole thing
               make_shape(bM,bK)) // argument 4

So the sentence means: argument 3, sA(_,_,0), is a layout template for the destination, not the
destination itself.
sA(_,_,0) describes the destination’s layout (shape, stride, and swizzle), not its address. 
The 0 drops the pipe mode, the descriptor describes one stage’s shape and swizzle, and stage 0
is just a representative.
The _ meaning “keep this whole mode”.

The swizzle matters because TMA must write smem in exactly the swizzled pattern that WGMMA 
expects to read (Layout_MN_SW128_Atom is a GMMA-defined atom), and that pattern also avoids 
bank conflicts. It’s just CuTe, not the driver, that translates the layout.

make_shape(bM, bK): It is the CTA tiler: it says which (bM,bK) piece of mA corresponds to the
smem layout, i.e. one stage’s tile. Its shape must match sA(_,_,0)’s shape (128,64).

Summary:
tmaA is not data; it holds a hardware descriptor and metadata. It holds the CUtensorMap plus
CuTe aux params.
SM90_TMA_LOAD selects the TMA instruction, it’s a struct with static copy functions dispatching
to 1D–5D variants. It emits PTX (cp.async.bulk.tensor), not SASS, and also selects the host-side
Copy_Traits.
mA gives bounds and strides. It also gives address and element type, all consumed on the host.
sA(_,_,0) On the host it’s a layout slice. The 0 drops the pipe mode; the real stage is chosen at copy time.
CuTe analyzes the layout and passes plain numbers (including the swizzle enum) to cuTensorMapEncodeTiled.
(bM,bK) is the single-transaction tile	Imprecise. It’s the CTA tiler; CuTe derives the box from it and the smem layout, possibly needing more than one TMA instruction (NT case, unverified).
TMA is a hardware copy engine in the SM.
Software layers program it: CuTe (layout → parameters), the driver API (parameters → descriptor), PTX (instruction) → SASS.
The descriptor is software-built data in a hardware-defined format, read directly by the TMA unit.
Hardware constraints are fixed; CuTe’s choices are library policy.

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
</pre>

