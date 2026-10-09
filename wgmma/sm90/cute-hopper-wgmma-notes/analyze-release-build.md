
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
