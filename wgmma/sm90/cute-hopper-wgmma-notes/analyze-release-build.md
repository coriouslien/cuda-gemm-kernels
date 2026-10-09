
<pre>
To decode the three kernel names, I ran the mangled names from the log through c++filt. I can 
reproduce it in one line:

grep -oE "_Z[A-Za-z0-9_]+" build-release/build-release.log | uniq | c++filt | sed 's/cute:://g; s/cutlass:://g'
  
void cub::CUB_200802_SM_900::EmptyKernel<void>()
void gemm_device<tuple<int, int, int>, tuple<C<128>, C<128>, C<64> >, half_t, tuple<int, C<1> >, 
                ComposedLayout<Swizzle<3, 4, 3>, 
                smem_ptr_flag_bits<16>, Layout<tuple<tuple<C<8>, C<16> >, tuple<C<64>, C<1> >, 
                tuple<C<1>, C<3> > >, tuple<tuple<C<64>, C<512> >, tuple<C<1>, C<0> >, 
                tuple<C<0>, C<8192> > > > >, 
                TiledCopy<Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t, uint128_t>, half_t>, 
                Layout<tuple<tuple<C<8>, C<16> >, C<8> >, tuple<tuple<C<128>, C<1> >, C<16> > >, 
                tuple<C<16>, C<64> > >, half_t, tuple<int, C<1> >, 
                ComposedLayout<Swizzle<3, 4, 3>, smem_ptr_flag_bits<16>, 
                Layout<tuple<tuple<C<8>, C<16> >, tuple<C<64>, C<1> >, tuple<C<1>, C<3> > >, 
                tuple<tuple<C<64>, C<512> >, tuple<C<1>, C<0> >, tuple<C<0>, C<8192> > > > >, 
                TiledCopy<Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t, uint128_t>, half_t>, 
                Layout<tuple<tuple<C<8>, C<16> >, C<8> >, tuple<tuple<C<128>, C<1> >, C<16> > >, 
                tuple<C<16>, C<64> > >, half_t, tuple<C<1>, int>, 
                TiledMMA<MMA_Atom<SM90::GMMA::MMA_64x64x16_F16F16F16_SS<(SM90::GMMA::Major)0, 
                          (SM90::GMMA::Major)0, (SM90::GMMA::ScaleIn)1,
                          (SM90::GMMA::ScaleIn)1> >, 
                Layout<tuple<C<1>, C<1>, C<1> >, tuple<C<0>, C<0>, C<0> > >, 
                tuple<Underscore, Underscore, Underscore> >, half_t, half_t>
                (tuple<int, int, int>, tuple<C<128>, C<128>, C<64> >, half_t const*, tuple<int, C<1> >, 
                ComposedLayout<Swizzle<3, 4, 3>, smem_ptr_flag_bits<16>, Layout<tuple<tuple<C<8>, C<16> >, 
                tuple<C<64>, C<1> >, tuple<C<1>, C<3> > >, tuple<tuple<C<64>, C<512> >, tuple<C<1>, C<0> >, 
                tuple<C<0>, C<8192> > > > >, 
                TiledCopy<Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t, uint128_t>, half_t>, 
                Layout<tuple<tuple<C<8>, C<16> >, C<8> >, tuple<tuple<C<128>, C<1> >, C<16> > >, 
                tuple<C<16>, C<64> > >, half_t const*, tuple<int, C<1> >, 
                ComposedLayout<Swizzle<3, 4, 3>, smem_ptr_flag_bits<16>, Layout<tuple<tuple<C<8>, C<16> >, 
                tuple<C<64>, C<1> >, tuple<C<1>, C<3> > >, tuple<tuple<C<64>, C<512> >, tuple<C<1>, C<0> >, 
                tuple<C<0>, C<8192> > > > >, 
                TiledCopy<Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t, uint128_t>, half_t>, 
                Layout<tuple<tuple<C<8>, C<16> >, C<8> >, tuple<tuple<C<128>, C<1> >, C<16> > >, 
                tuple<C<16>, C<64> > >, half_t*, tuple<C<1>, int>, 
                TiledMMA<MMA_Atom<SM90::GMMA::MMA_64x64x16_F16F16F16_SS<(SM90::GMMA::Major)0, 
                          (SM90::GMMA::Major)0, (SM90::GMMA::ScaleIn)1, (SM90::GMMA::ScaleIn)1> >, 
                Layout<tuple<C<1>, C<1>, C<1> >, tuple<C<0>, C<0>, C<0> > >, 
                tuple<Underscore, Underscore, Underscore> >, half_t, half_t)
                  
void gemm_device<tuple<int, int, int>, tuple<C<128>, C<128>, C<64> >, half_t, tuple<C<1>, int>, ComposedLayout<Swizzle<3, 4, 3>, smem_ptr_flag_bits<16>, Layout<tuple<tuple<C<64>, C<2> >, tuple<C<8>, C<8> >, tuple<C<1>, C<3> > >, tuple<tuple<C<1>, C<512> >, tuple<C<64>, C<1024> >, tuple<C<0>, C<8192> > > > >, TiledCopy<Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t, uint128_t>, half_t>, Layout<tuple<C<128>, C<8> >, tuple<C<8>, C<1> > >, tuple<C<128>, C<8> > >, half_t, tuple<C<1>, int>, ComposedLayout<Swizzle<3, 4, 3>, smem_ptr_flag_bits<16>, Layout<tuple<tuple<C<64>, C<2> >, tuple<C<8>, C<8> >, tuple<C<1>, C<3> > >, tuple<tuple<C<1>, C<512> >, tuple<C<64>, C<1024> >, tuple<C<0>, C<8192> > > > >, TiledCopy<Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t, uint128_t>, half_t>, Layout<tuple<C<128>, C<8> >, tuple<C<8>, C<1> > >, tuple<C<128>, C<8> > >, half_t, tuple<C<1>, int>, TiledMMA<MMA_Atom<SM90::GMMA::MMA_64x64x16_F16F16F16_SS<(SM90::GMMA::Major)1, (SM90::GMMA::Major)1, (SM90::GMMA::ScaleIn)1, (SM90::GMMA::ScaleIn)1> >, Layout<tuple<C<1>, C<1>, C<1> >, tuple<C<0>, C<0>, C<0> > >, tuple<Underscore, Underscore, Underscore> >, half_t, half_t>(tuple<int, int, int>, tuple<C<128>, C<128>, C<64> >, half_t const*, tuple<C<1>, int>, ComposedLayout<Swizzle<3, 4, 3>, smem_ptr_flag_bits<16>, Layout<tuple<tuple<C<64>, C<2> >, tuple<C<8>, C<8> >, tuple<C<1>, C<3> > >, tuple<tuple<C<1>, C<512> >, tuple<C<64>, C<1024> >, tuple<C<0>, C<8192> > > > >, TiledCopy<Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t, uint128_t>, half_t>, Layout<tuple<C<128>, C<8> >, tuple<C<8>, C<1> > >, tuple<C<128>, C<8> > >, half_t const*, tuple<C<1>, int>, ComposedLayout<Swizzle<3, 4, 3>, smem_ptr_flag_bits<16>, Layout<tuple<tuple<C<64>, C<2> >, tuple<C<8>, C<8> >, tuple<C<1>, C<3> > >, tuple<tuple<C<1>, C<512> >, tuple<C<64>, C<1024> >, tuple<C<0>, C<8192> > > > >, TiledCopy<Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t, uint128_t>, half_t>, Layout<tuple<C<128>, C<8> >, tuple<C<8>, C<1> > >, tuple<C<128>, C<8> > >, half_t*, tuple<C<1>, int>, TiledMMA<MMA_Atom<SM90::GMMA::MMA_64x64x16_F16F16F16_SS<(SM90::GMMA::Major)1, (SM90::GMMA::Major)1, (SM90::GMMA::ScaleIn)1, (SM90::GMMA::ScaleIn)1> >, Layout<tuple<C<1>, C<1>, C<1> >, tuple<C<0>, C<0>, C<0> > >, tuple<Underscore, Underscore, Underscore> >, half_t, half_t)

  

(sed only strips the namespaces so the names fit on screen.)
Each conclusion is labeled with its source:

• [log]: read directly from the build output;
• [name]: from the demangled kernel name, which encodes every template argument;
• [CuTe]: from how CuTe/CUTLASS defines these types (header knowledge, not in your log);
• [source]: from the tutorial’s wgmma_sm90.cu. Eeach [source] claim comes with a grep.

Part 1: The shape of the ptxas report [log]
contain one module-level line, then one block per kernel:
ptxas info : 47 bytes gmem        ← module level (whole .cu file)
ptxas info : Compiling entry function '<name>' for 'sm_90a'    ┐
ptxas info : Function properties for <name>                    │
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads   │ one block
ptxas info : Used N registers, used M barriers                  │ per kernel
ptxas info : Compile time = X ms       ┘

The first of the three kernels in your log, one line at a time. Here’s that line, piece by piece.
void cub::CUB_200802_SM_900::EmptyKernel<void>()
</pre>
|Piece	|Meaning|
|:------|:------|
|void	|return type. Every \__global\__ kernel returns void|
|cub::	|the CUB library’s namespace. The sed strips only cute:: and cutlass::, so cub:: stays|
|CUB_200802_SM_900::|	an inline namespace inside cub (explained below)|
|EmptyKernel<void>|	a function template named EmptyKernel, instantiated with T = void|
|()	|no parameters|
<pre>
Where it comes from
The source file never mentions it. The chain is.
wgmma_sm90.cu L5–6 include thrust/host_vector.h and thrust/device_vector.h;
Thrust’s device code is built on CUB, so CUB’s headers get included;
CUB declares EmptyKernel as a template kernel with an empty body, and it uses EmptyKernel<void> in its own
code. That use is enough to instantiate the kernel, so nvcc compiles it and ptxas reports it.
As far as I know, CUB uses it to ask the driver which PTX version the binary was compiled for 
(by querying the kernel’s attributes), not to do any work. The tutorial itself never launches it.

The developer can confirm the definition in the toolkit’s headers:
  grep -n "EmptyKernel" /usr/local/cuda/include/cub/util_device.cuh
  
The inline namespace CUB_200802_SM_900
The name is built from two parts:
• 200802 = the CUB version, encoded as major × 100000 + minor × 100 + patch, so 2.8.2. That’s the CUB 
that ships with the CUDA 12.9 toolkit. Check:
  grep -n "define CUB_VERSION" /usr/local/cuda/include/cub/version.cuh
• SM_900 = the architecture list this file was compiled for. It comes from nvcc’s __CUDA_ARCH_LIST__ macro,
which is 900 for your compute_90a target. The a doesn’t appear in it.

Why CUB does this: an inline namespace is transparent in source code (a developer still write
cub::EmptyKernel), but it becomes part of the mangled symbol name. If two libraries in one program were 
built with different CUB versions or different architecture lists, their copies of CUB kernels get 
different symbol names and can’t silently replace each other at link time, which would violate C++’s 
One Definition Rule. This versioning scheme is a design choice inside CCCL, the library family on your 
roadmap.

Its ptxas numbers
The command keeps only names, so the numbers aren’t in its output. They’re in the log, lines 26–29:

0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 4 registers, used 0 barriers
Compile time = 0.885 ms

An empty kernel needs no barriers, no stack and no spills. The 4 registers is a small fixed minimum ptxas 
reports even for an empty body; it carries no meaning.

For the study: this kernel is a side effect of including Thrust, so recognize it and move on. The next line 
of your output is the first gemm_device (the TN kernel), which is where the real study starts.
</pre>
<pre>
Step 4: gemm_tn, the host setup that becomes kernel 1
Source matched to the template argument in the ptxas name:
</pre>

|Source/host|Value	|Becomes in the name|
|:------|:------|:------------------|
|make_shape(M, N, K) with int|runtime|tuple<int,int,int>|
|dA = make_stride(ldA, Int<1>{})|	(runtime, static 1) |tuple<int, C<1>>: K-major|
|dB = make_stride(ldB, Int<1>{})	|same	|tuple<int, C<1>>|
|dC = make_stride(Int<1>{}, ldC)|	(static 1, runtime)	|tuple<C<1>, int>|
|bM, bN, bK = Int<128>, Int<128>, Int<64>|static|tuple<C<128>, C<128>, C<64>>|
|bP = Int<3>{}	| static	| the (1,3):(0,8192) PIPE mode|
|L383<br>tile_to_shape(GMMA::Layout_K_SW128_Atom<TA><br>{}, ...)| - | ComposedLayout<Swizzle<3,4,3>, …,<br>((8,16),(64,1),(1,3)):…>|

<pre>
In the source line L383 tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, ...). In build output interpret 
to ComposedLayout<Swizzle<3,4,3>, …, ((8,16),(64,1),(1,3)):…>
This is a three-stage process, and only the first and last stages are visible: L383 (the call) 
and the build log (the result). In between are two things defined inside CuTe’s headers that is not visible. 
These are stage by stage, and say where each piece comes from.
L383:      tile_to_shape( GMMA::Layout_K_SW128_Atom<TA>{},  make_shape(bM, bK, bP) )
                          └──────── the "atom" ────────┘     └─ the target shape ─┘
                                                              (128, 64, 3)

build log: ComposedLayout< Swizzle<3,4,3>, smem_ptr_flag_bits<16>,
                           Layout< ((8,16),(64,1),(1,3)) : ((64,512),(1,0),(0,8192)) > >
Stage 1: The two inputs at L383 [source]

The target shape is make_shape(bM, bK, bP). From L376–380: bM = Int<128>, bK = Int<64>, bP = Int<3>. 
So the target is (128, 64, 3): M × K × pipeline stages. All three are compile-time constants.
The atom is GMMA::Layout_K_SW128_Atom<TA> with TA = half_t. L383 only names it. Its definition is in 
CuTe’s headers, not in the file. That’s Stage 2.

tile_to_shape means: take the small atom and repeat it until it covers the target shape.   

Stage 2: What the atom is [CuTe header]

CuTe defines the GMMA atoms first in units of bits, then converts them to the element type. Roughly 
(this is the shape of the definition; check the exact text with the grep below):
// in bits
using Layout_K_SW128_Atom_Bits =
    ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<1>,
                   Layout<Shape<_8,_1024>, Stride<_1024,_1>>>;

// in units of Type: divide the bit layout by sizeof_bits<Type>
template <class Type>
using Layout_K_SW128_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW128_Atom_Bits{}));
Check in your CUTLASS checkout:
  grep -rn "Layout_K_SW128_Atom" ../cutlass/include/cute/atom/

Converting bits to half_t (16 bits): upcast<16> divides the K dimension and the strides by 16:
</pre>
||in bits|	in half_t (÷16)|
|:---|:------|:---------------|
|shape	|(8, 1024)|	(8, 64)|
|stride	|(1024, 1)|	(64, 1)|
|flag	|smem_ptr_flag_bits<1>|	smem_ptr_flag_bits<16>|
|swizzle|	Swizzle<3,4,3>|	Swizzle<3,4,3>, unchanged|
<pre>
So for half_t the atom is:
ComposedLayout< Swizzle<3,4,3>, smem_ptr_flag_bits<16>, Layout<(8,64):(64,1)> >
What that atom describes: 8 rows (M) × 64 columns (K) of halfs. Moving one step in K moves 1 element; 
one step in M moves 64 elements. Each row is 64 halfs = 128 bytes, contiguous along K, and the whole 
atom is 8 × 128 = 1024 bytes:
        k = 0 ............................ 63
m = 0   [ 128 bytes, contiguous along K ]
m = 1   [ 128 bytes ]
  ...
m = 7   [ 128 bytes ]          ← one atom = 8 rows = 1024 bytes = 512 halfs
The swizzle and the flag are why the result is a ComposedLayout and not a plain Layout (Stage 4).
Stage 3: tile_to_shape repeats the atom [CuTe]

The atom covers (8, 64). The target is (128, 64, 3). Mode by mode, how many atoms are needed:
</pre>
|Mode|	Target|	Atom covers|	Atoms needed|
|:---|:-------|:-----------|:-------------|
|M	|128|	8|	16|
|K	|64	|64	|1|
|PIPE|	3|	— (the atom has only 2 modes)	|3|
<pre>
The atom has no third mode. CuTe extends it with a size-1 mode, shape 1, stride 0, so it has three modes 
like the target. That size-1 mode is the 1 in (1,3).

Each mode of the result is a pair: (inside one atom, which atom). Atoms are placed one after another 
in memory, M first, then K, then PIPE. Each atom occupies 512 elements:

M mode → (8,16):(64,512)
• 8:64: the 8 rows inside an atom, 64 elements apart (from the atom);
• 16:512: 16 atoms stacked along M, each 512 elements (one atom) after the previous.

K mode → (64,1):(1,0)
• 64:1: the 64 columns inside an atom, contiguous (from the atom);
• 1:0: one atom along K. A mode of size 1 only ever has coordinate 0, so its stride never matters. 
CuTe writes 0.

PIPE mode → (1,3):(0,8192)
• 1:0: the size-1 mode added to the atom (it has no third dimension);
• 3:8192: 3 copies of everything so far. Everything so far is 16 atoms × 512 elements = 8192 elements, 
so each stage starts 8192 elements after the previous.

Putting the three modes together gives exactly the layout in your build log:
((8,16), (64,1), (1,3)) : ((64,512), (1,0), (0,8192))
   M        K      PIPE      M          K      PIPE
Check by evaluating it. Write m = m0 + 8·m1 (row inside an atom, atom index):
offset = 64·m0 + 512·m1  +  1·k  +  8192·p
       = 64·(m0 + 8·m1)   +  k   +  8192·p
       = 64·m             +  k   +  8192·p
Before the swizzle, each stage is simply a row-major 128 × 64 tile, with K contiguous. The nested (8,16) 
only records that it was built from 16 atoms of 8 rows.

It has described what tile_to_shape produces, not its internal implementation. If developers want to read it:
  grep -rn "tile_to_shape" ../cutlass/include/cute/layout.hpp ../cutlass/include/cute/layout_composed.hpp

Stage 4: The ComposedLayout wrapper [CuTe]
tile_to_shape only grows the inner layout. The swizzle and the flag are copied unchanged from the atom:
atom:    ComposedLayout< Swizzle<3,4,3>, smem_ptr_flag_bits<16>, (8,64):(64,1) >
result:  ComposedLayout< Swizzle<3,4,3>, smem_ptr_flag_bits<16>, ((8,16),(64,1),(1,3)):((64,512),(1,0),(0,8192)) >
                         └── same ──┘     └────── same ──────┘    └───────── grown by tile_to_shape ─────────┘

ComposedLayout<A, O, B> means “apply B, then A”. In order:
1. B, the inner layout: coordinate (m, k, p) → element offset;
2. smem_ptr_flag_bits<16>: tells CuTe the elements are 16-bit, so element offset × 2 = byte offset;
3. A, Swizzle<3,4,3>: permutes the byte offset (XORs bits 7–9 into bits 4–6).

Why one swizzle can cover the whole tile: Swizzle<3,4,3> only reads and changes address bits 4–9, a pattern
that repeats every 1024 bytes. Every atom is exactly 1024 bytes and starts at a multiple of 1024 bytes 
(512 elements × 2 bytes). So each atom gets the identical swizzle pattern, and repeating the atom never 
breaks it.

L383 [source]                    tile_to_shape( atom ,  (128, 64, 3) )
                                                │
Stage 2 [CuTe]  atom in bits:   Sw<3,4,3> ∘ flag<1>  ∘ (8,1024):(1024,1)
                upcast<16>:     Sw<3,4,3> ∘ flag<16> ∘ (8,64):(64,1)        ← 8×128 B = 1 KB
                                                │
Stage 3 [CuTe]  repeat atom:    M ×16, K ×1, PIPE ×3   (M first; 512 elements per atom)
                                                │
Stage 4         wrap:           Sw<3,4,3> ∘ flag<16> ∘ ((8,16),(64,1),(1,3)):((64,512),(1,0),(0,8192))
                                                │
build log [name]                 ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<16>, Layout<…>>

The program print_layouts.cu. Verify the layouts on your own machine (no Hopper needed)
prints this same layout on your own machine. That lets you check Stage 4 directly. Printing
GMMA::Layout_K_SW128_Atom<half_t>{} on its own would also show the Stage 2 atom before it’s repeated. 
  
swizzle isn’t fixed for SM90.
Swizzle<3,4,3> is fixed for this one atom, Layout_K_SW128_Atom, the one the tutorial chose. Hopper’s 
wgmma supports several shared-memory layout modes, and CuTe defines one atom for each, in the same file. 
Their names follow the hardware modes:
</pre>
|Atom	|Swizzle	|16-byte chunks permuted	|Swizzle width|
|:----|:-------|:-------------------------|:------------|
|Layout_K_INTER_Atom	|Swizzle<0,4,3>|	none (B = 0, no swizzle)|	—|
|Layout_K_SW32_Atom	|Swizzle<1,4,3>	|2	|32 bytes|
|Layout_K_SW64_Atom	|Swizzle<2,4,3>	|4	|64 bytes|
|Layout_K_SW128_Atom|	Swizzle<3,4,3>	|8	|128 bytes|
<pre>
The pattern is in the three numbers:
• M = 4 is always the same: the unit being moved is a 16-byte chunk (address bits 0–3 stay inside the chunk);
• S = 3 is always the same;
• B varies, 0 to 3: it permutes 2^B chunks, so it sets the swizzle width (2^B × 16 bytes).
There is also a matching MN-major family (Layout_MN_INTER_Atom … Layout_MN_SW128_Atom), which kernel
2 (NT) uses at L291.

The truly fixed part on SM90 is that wgmma accepts only these few modes. A shared-memory layout for an _SS 
MMA must be built from one of these atoms, not from an arbitrary layout.

(8,1024):(1024,1) is in bits

Line 84 is the bit layout (the _Bits suffix).
../cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:84:using Layout_K_SW128_Atom_Bits  = 
ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_1024>,Stride<_1024,_1>>>;

The layout the kernel uses is what line 104 produces for half_t, which is (8,64):(64,1). The bit 
version is written once so it can be converted to any element type:
for an 8-bit type it would become (8,128):(128,1), and for a 32-bit type (8,32):(32,1). In every case each 
row is 1024 bits = 128 bytes, which is why the atom matches the 128-byte swizzle.

See all the atoms in your header
sed -n '70,125p' ../cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp
This shows the full families. Line 122 of your grep output sits inside a selector in that range: as far as 
I know it picks an atom automatically from the tile size. That’s how CUTLASS’s production kernels choose 
among INTER/SW32/SW64/SW128. The tutorial instead names Layout_K_SW128_Atom directly. Reading those lines 
will confirm the exact selection rule.

For the shared-memory operands of wgmma, there are four swizzle modes. But your table shows only half of 
the atoms, and a few constraints come with them.
  
The four modes come from the hardware
A wgmma reads A and B from shared memory through a matrix descriptor, a 64-bit value. Its layout-type field 
is 2 bits, so the hardware supports exactly four modes: no swizzle (“interleave”), 32-byte, 64-byte and 
128-byte swizzle. The four Swizzle<B,4,3> atoms are CuTe’s description of those four modes, so there’s 
no fifth one to choose.

Developer can see the encoding in your CUTLASS checkout: 
  grep -rn "enum class LayoutType" -A8 ../cutlass/include/cute/arch/mma_sm90_desc.hpp
  
The conversion: upcast<16>

upcast<16> means: 16 consecutive bits now form one element. It rewrites the layout so it counts in elements
instead of bits. Each mode is treated by its stride:
As far as I know it lists INTERLEAVE, B128, B64 and B32. The numeric order in that enum isn’t the order of 
my table.
</pre>
|Swizzle	|K-major atom (TN, L383)|	MN-major atom (NT, L291)|
|:--------|:----------------------|:------------------------|
|Swizzle<0,4,3>|	Layout_K_INTER_Atom|	Layout_MN_INTER_Atom|
|Swizzle<1,4,3>	|Layout_K_SW32_Atom	|Layout_MN_SW32_Atom|
|Swizzle<2,4,3>	|Layout_K_SW64_Atom	|Layout_MN_SW64_Atom|
|Swizzle<3,4,3>	|Layout_K_SW128_Atom|	Layout_MN_SW128_Atom|
<pre>
Three constraints that go with them
MN-major only works for 16-bit types. wgmma can read an MN-major (transposed) operand only for f16/bf16. 
For tf32, fp8 and integer types, A and B in shared memory must be K-major. That’s why the tutorial can 
offer an NT version: its type is half_t.
Only the atom is fixed. How atoms are arranged to fill the whole tile is set by two stride fields in the 
same descriptor (leading-byte offset and stride-byte offset). That’s why tile_to_shape can stack 16 
atoms along M at stride 512 and the hardware still follows.
A doesn’t have to be in shared memory. With an _RS atom (instead of the tutorial’s _SS), A comes from
registers, and the shared-memory layout rules apply only to B.
_RS means A from Registers, B from Shared memory. The two letters follow operand order: first A, then B. 
And the reason the shared-memory rules then apply only to B is that B can never come from registers. 
That’s a rule of the wgmma instruction, not a CuTe choice. 
What the instruction allows
In PTX, wgmma.mma_async takes its operands like this:
</pre>
|Operand|	Allowed sources|
|:------|:---------------|
|A|	a shared-memory descriptor, or registers|
|B|	a shared-memory descriptor only|
|C/D (accumulator)|	registers only|
<pre>
So there are exactly two combinations, and CuTe has an atom for each:
• _SS: A and B from shared memory (the tutorial, L394 and L302);
• _RS: A from registers, B from shared memory.  
There is no _SR and no _RR. B always goes through a descriptor, so B always needs one of the 8 GMMA 
layout atoms. With _RS, A doesn’t use a descriptor, so the shared-memory layout rules simply don’t apply to it.
Why the hardware is asymmetric
The PTX ISA states the rule; the explanation below is my understanding of the design, not something the
documentation spells out.

A wgmma is executed by a warpgroup = 4 warps. For an M = 64 instruction, each warp computes 16 rows of 
the result:
          B (all N columns, shared by every warp)
          ┌───────────────┐
A rows 0–15   ← warp 0  →  │  C rows 0–15
A rows 16–31  ← warp 1  →  │  C rows 16–31
A rows 32–47  ← warp 2  →  │  C rows 32–47
A rows 48–63  ← warp 3  →  │  C rows 48–63
• A splits naturally between the warps. Each warp needs only its own 16 rows of A, so A can be spread 
  across the warps’ registers, like the A fragment of mma.sync.
• B is needed in full by all four warps. If B were in registers, every warp would need its own copy of 
  the whole B tile. Reading B from shared memory lets the hardware fetch it once and serve all four warps. 
  
A has rules too, just different ones
With _RS, A doesn’t escape layout rules; it trades them for a register fragment layout: which thread holds 
which elements of A, fixed by the instruction (as with the mma.sync fragments). CuTe encodes that in the
atom’s traits, and partition_A / make_fragment_A follow it.

When _RS is used
_RS makes sense when A is already in registers, or has to pass through registers anyway:
• Flash Attention’s second GEMM. P = softmax(S) is computed in registers, then P · V uses P directly as 
the register A operand. That avoids writing P to shared memory and reading it back. This is the pattern in 
the Hopper Flash Attention kernels I'm studying(The reference material can be found at the end of this note.)
• Mixed-input GEMM, e.g. A stored as int8 or fp8 and converted to fp16 in registers before the MMA.

  
Your table is the K-major half

Each mode exists in two orientations, so CuTe defines eight atoms:
Mode 1 (columns, along K): 1024 : 1
• stride 1: the bits are contiguous along this mode;
• 16 contiguous bits = 1 half_t, so 1024 bits = 64 elements;
• consecutive elements are still adjacent, so the stride stays 1;
• → 64 : 1. The shape is divided.
  
Mode 0 (rows, along M): 8 : 1024
• stride 1024: each row starts 1024 bits after the previous one;
• 1024 bits = 64 elements, so the stride becomes 64;
• there are still 8 rows. The number of rows doesn’t depend on the element size;
• → 8 : 64. The stride is divided; the shape is unchanged.

Result: (8,1024):(1024,1) → (8,64):(64,1).

The rule: a contiguous mode (stride 1) has its shape divided. A strided mode (stride a multiple of 16) 
has its stride divided. That’s why the 8 and the 1 stay the same: 8 is the shape of the strided mode, 
and 1 is the stride of the contiguous mode.

Check: the same bits, counted two ways
Take element (m, k), row m, the k-th half_t in that row:
in bits: 1024·m + 16·k
in halfs (÷16): 64·m + k, which is the (8,64):(64,1) layout
Each row is 1024 bits = 128 bytes, whatever the type. Only how many elements fit in it changes:
</pre>
|Type	|Bits per element	|Atom	|Elements per 128-byte row|
|:----|:----------------|:----|:------------------------|
|8-bit (e.g. fp8)|	8	|(8,128):(128,1)|	128|
|16-bit (half_t)|	16	|(8,64):(64,1)	|64|
|32-bit (tf32/float)|	32	|(8,32):(32,1)|	32|
<pre>
This is why CuTe writes the atom once in bits: one definition (line 84) produces the correct element 
layout for every type (line 104), and the row is always 128 bytes, which is what the 128-byte swizzle requires.
../cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:84:using Layout_K_SW128_Atom_Bits  =
ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_1024>,Stride<_1024,_1>>>;

../cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:104:using Layout_K_SW128_Atom =
decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW128_Atom_Bits{}));

To read upcast itself in your checkout:
  grep -rn "upcast" ../cutlass/include/cute/layout.hpp | head
  
In CuTe’s notation Shape : Stride, the left side is the shape and the right side the stride:
(8, 1024) : (1024, 1)
 └ shape ┘   └ stride ┘
in C++:
Layout<Shape<_8,_1024>, Stride<_1024,_1>>
Shape and stride pair up by position, one pair per mode:
</pre>
|Mode	|Shape	|Stride	|Meaning|
|:----|:------|:------|:------|
|0 (rows, M)|	8	|1024|	8 rows; the next row starts 1024 bits later|
|1 (columns, K)|	1024|	1|	1024 bits per row; the next bit is 1 bit later|
<pre>
The offset of a coordinate is each coordinate times its mode’s stride, summed:
offset(m, k) = m·1024 + k·1      (m in [0,8), k in [0,1024), in bits)
  
See all the atoms in your header
  sed -n '70,125p' ../cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp

</pre>
<pre>
=================================================================================
Here’s where to study it, ordered from the most direct source to the code. None of these Hopper kernels can
run on my SM120 GPUs, so the plan is the same as for wgmma_sm90.cu: read and compile locally, run and profile
on a rented H100.

1. The FlashAttention-3 paper (primary source)
</pre>
[FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
<pre>
 FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision https://arxiv.org/abs/2407.08608 (arXiv 2407.08608; also published at NeurIPS 2024). This is the main reference for Hopper FlashAttention. The sections that matter for your question:

Section 3.1, Algorithm 1: the forward pass with warp specialization. The P·V product is explicitly labeled “RS-GEMM”, and the paper defines the SS/RS prefix as whether the first operand comes from shared memory or from the register file. So the second GEMM is exactly the _RS case we discussed. Section 3.1 also covers pingpong scheduling: two warpgroups alternate, one doing GEMMs while the other does softmax.
Section 3.2: intra-warpgroup overlap of softmax with the GEMMs, which is where the second GEMM consuming P directly from registers matters most. Appendix B.3 has a 3-stage variant.
Section 2.2: the FP8 constraint from my previous answer, in the paper’s own words: FP8 wgmma supports only K-major operands, and the FP32 accumulator layout doesn’t match the FP8 operand layout, which blocks feeding one GEMM’s output into the next.
Section 3.3: how they solve that for FP8 (layout transformations inside the kernel). The page I fetched was cut off before this section’s text, so read it in the PDF.
2. Overviews written by the authors
Tri Dao’s blog post on FlashAttention-3: a shorter tour of WGMMA, TMA, pingpong scheduling and intra-warpgroup overlap, with throughput numbers for each step (FP16 forward rising from about 570 to 620 TFLOPS with pingpong, then to 640–660 with intra-warpgroup overlap). It doesn’t describe the second GEMM’s operand sources; the paper does.
PyTorch blog post on FlashAttention-3: similar content.
3. FlashAttention-2 on Hopper with CUTLASS (closest in style to your tutorial)

A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library (Colfax Research, December 2023; the arXiv version appears to be 2312.11918). A fused FA2 forward kernel for Hopper built with CUTLASS layouts and tensors, WGMMA and TMA, measured on an H100 PCIe, the same class of GPU you plan to rent. It predates FA3, so it’s a simpler starting point. The article page doesn’t say whether P stays in registers; check the PDF. Colfax’s research page also lists their other CUTLASS/CuTe tutorials.

4. The instruction-level reference
The PTX ISA, section “Asynchronous Warpgroup Level Matrix Multiply-Accumulate” (wgmma.mma_async). The authoritative source for the operand rule (A from a descriptor or registers, B only from a descriptor), the descriptor format and the register fragment layouts. Use the current version at docs.nvidia.com/cuda/parallel-thread-execution/ matching your CUDA 12.9; search results also turned up an older archived version (PTX 8.2).
CUTLASS Warpgroup MMA Programming Guide: a note first: it’s written for the CuTe DSL (Python). The concepts (warpgroups, descriptors, SS vs RS) carry over to C++ CuTe, but the code examples are Python, so use it for concepts only.
5. The code

Dao-AILab/flash-attention: FlashAttention-3 is in the hopper/ directory. It requires an H100/H800 and CUDA ≥ 12.3 (12.8 recommended); your 12.9 is fine for compiling.

Two things to know about this repository:

hopper/ is C++ (CUTLASS/CuTe), the one for you.
flash_attn/cute is FlashAttention-4, aimed at Hopper and Blackwell (H100, B200). It’s written in the CuTe DSL, which is Python. Since you work in C++, skip it for now. It also targets datacenter Blackwell (SM100), not your SM120.

To find the RS second GEMM in hopper/, search for these terms. They’re my best guesses at the names used, not confirmed:

bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
grep -rn "rs_op_selector\|_RS\b\|PV_is_RS\|Mma_PV\|MmaPV" --include=*.h --include=*.hpp --include=*.cu . | head -30

CUTLASS’s own attention examples: your CUTLASS checkout may also contain Hopper and Blackwell FMHA examples. I couldn’t confirm the exact directory names, so list them yourself:

bash
ls ../cutlass/examples | grep -i -E "fmha|attention"
A suggested reading order
FA3 paper, Sections 2.2 and 3.1 (Algorithm 1 and the RS-GEMM label).
The PTX ISA wgmma section on register A operands, to see the register fragment layout the RS atom must follow.
FA3 paper, Section 3.2 (why keeping P in registers enables the overlap).
The hopper/ mainloop in the FlashAttention repository, using the search above to locate the second GEMM.
Colfax’s FA2-on-Hopper paper, if you want a simpler kernel to compare against.

Sources:

FlashAttention-3 paper (arXiv 2407.08608), HTML version
FlashAttention-3 at NeurIPS 2024
Tri Dao: FlashAttention-3 blog post
PyTorch blog: FlashAttention-3
Colfax Research: FlashAttention-2 on Hopper with CUTLASS
arXiv 2312.11918
Colfax Research publications
PTX ISA 8.2 (archived)
CUTLASS Warpgroup MMA Programming Guide
Dao-AILab/flash-attention on GitHub
</pre>

