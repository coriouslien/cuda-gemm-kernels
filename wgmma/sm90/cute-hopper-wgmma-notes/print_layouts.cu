// print_layouts.cu — host only; runs on any machine
//nvcc -std=c++17 --expt-relaxed-constexpr \
//     -I ../cutlass/include -I ../cutlass/tools/util/include \
//     print_layouts.cu -o print_layouts && ./print_layouts

#include <cute/tensor.hpp>
using namespace cute;

int main() {
  using TA = cute::half_t;
  auto bM = Int<128>{}; auto bK = Int<64>{}; auto bP = Int<3>{};

  auto sA_tn = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{},  make_shape(bM, bK, bP));
  auto sA_nt = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
  print("sA_tn : "); print(sA_tn); print("\n");
  print("sA_nt : "); print(sA_nt); print("\n");
  print("cosize: "); print(cosize(sA_tn.layout_b())); print("\n");

  auto copyA_tn = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
                                  Layout<Shape<_16,_8>, Stride<_8,_1>>{},
                                  Layout<Shape< _1,_8>>{});
  print(copyA_tn); print("\n");

  auto mma_tn = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
  print(mma_tn); print("\n");

  // TN stage 0: physical 16-byte chunk for rows 0..7, logical chunks 0..7
  for (int m = 0; m < 8; ++m) {
    for (int c = 0; c < 8; ++c) {
      int byte = 2 * sA_tn.layout_b()(m, 8 * c, 0);   // unswizzled byte offset
      int phys = Swizzle<3,4,3>{}(byte);             // byte-address swizzle
      printf("%d ", (phys >> 4) & 7);
    }
    printf("\n");
  }
}
