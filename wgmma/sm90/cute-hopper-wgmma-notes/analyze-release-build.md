
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
|L383| - | ComposedLayout<Swizzle<3,4,3>, …,|
