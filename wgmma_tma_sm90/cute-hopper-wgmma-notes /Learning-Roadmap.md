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
A is the starting address of the matrix data, but it is a GPU virtual address, not a physical one. 
It comes from d_A.data().get() in main, which is device memory allocated by thrust::device_vector.
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
So make_tensor doesn’t build (1, dA) from dA. It uses dA as-is, giving the layout (M,K):(_1,ldA). 
Element (i,j) is at A + i*1 + j*ldA, and strides are counted in elements, not bytes.
If the matrix is column-major ...?
CuTe makes no such decision. The layout is whatever stride you pass, and the “column-major” 
choice was made by the programmer when writing L246 for the NT case. In CuTe terms this is 
M-major (M is the stride-1 mode).
The TN path (L314) passes dA = (ldA, _1) instead, which is K-major, and the same make_tensor call 
then describes row-major A. Same line of code, different stride, different layout.
Where the value comes from: main sets ldA = m when transA == 'N', so with the defaults, mA is 
(512,1024):(_1,512).
Line 272: Tensor mB = make_tensor(B, make_shape(N,K), dB);
The shape is (N,K), not (K,N). This code stores B as N×K and computes C(m,n) = Σ_k A(m,k)·B(n,k). That convention is why the kernel tiles B with Step<X,_1,_1> at L77 (selecting the N and K tile modes).
dB = (_1, ldB) (L247), so B is N-major in the NT path. main sets ldB = n when transB == 'T', so with the defaults mB is (256,1024):(_1,256).
Summary
Tensor as a view: correct. Note A is a raw pointer, so there’s no gmem tag, unlike mC.
A: the starting address, but a GPU virtual address, not physical. Its type also sets the element type and const-ness.
Shape (M,K): correct. Dynamic ints defining the coordinate domain.
dA: not the leading dimension. It is the complete stride tuple (_1, ldA) built at L246, used unchanged, in units of elements. ldA is the leading dimension.
Major-ness: set by the stride the programmer passes (M-major here, K-major in TN), not inferred by CuTe.
L272: same pattern, but B is shaped (N,K) with dB = (_1, ldB), so it is N-major.
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));
