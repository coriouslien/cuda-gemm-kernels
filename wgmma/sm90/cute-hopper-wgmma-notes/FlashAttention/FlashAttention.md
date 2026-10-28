Paused (do not mix in): FlashAttention study, finished Lessons 1-5 (softmax, safe softmax,
attention, cost of S/P, online softmax); resume later at Lesson 6.


Direction check: this is for the FlashAttention document, which is a separate track 
fromfinishing wgmma_sm90.cu. It’s a quick lookup, so I’ll get it now. Then the kernel’s 
lastlines are still next.

FA3 supports both _SS and _RS for P·V and chooses per configuration. The first GEMM (Q·Kᵀ) is
always _SS.

Everything below is from FA3’s Hopper forward kernel, fetched today from the main branch of 
Dao-AILab/flash-attention, directory hopper/. Line numbers may shift in later versions, so
record the date (or a commit) in your document.

1. Where SS vs RS is decided

hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp

L80–81, Q·Kᵀ is always SS:

  // Register bandwidth is actually a bottleneck so we don't want Q to be in registers.
  static constexpr bool MmaQK_is_RS = false;
L33: MmaPV_is_RS is a template parameter, so P·V can be either.
L77, L82–84: the constraints:

  static_assert(!LargeHeadDimV || !MmaPV_is_RS, "MmaPV must be SS for large Headdim_V");
  // We can have MmaPV with P in smem in rmem to reduce register pressure at the cost of more smem.
  static_assert(!(!MmaPV_is_RS && Is_FP8), "MmaPV must be RS if FP8");
  static_assert(!(!MmaPV_is_RS && Transpose_V), "MmaPV must be RS if Transpose_V");

The L82 comment states the trade-off directly: SS puts P in shared memory to reduce register pressure, at the cost of more shared memory. RS keeps P in registers.

L86–87: a special case where one warpgroup uses RS and the other SS:

  // Slightly faster in this case to have WG1 use RS instead of SS to avoid waiting for the P smem write
  static constexpr bool MmaPV_use_RS_WG1 = !MmaPV_is_RS && kHeadDim == 64 && kHeadDimV == 512;

hopper/tile_size.h, tile_size_fwd_sm90 returns {kBlockM, kBlockN, MmaPV_is_RS, IntraWGOverlap} (L9). For FP16/BF16 (element_size == 2):

Head dim (Q·K)	Condition	MmaPV_is_RS	Source lines
≤ 64	headdim_v = 512	false (SS)	L19–20
≤ 64	headdim_v = 256	true (RS)	L21–22
≤ 64	otherwise	true only if causal, local or paged-KV (non-TMA); else SS	L25–26
≤ 96	—	false (SS)	L30–31
≤ 128	—	true (RS)	L32–34
≤ 192, > 192	—	true (RS)	L37–41

For FP8 (the else branch, L43–54), every case returns true (RS), which matches the L83 assert.

2. The conversion sequence: accumulator S → operand P

mainloop_fwd_sm90_tma_gmma_ws.hpp, L1155–1162 (the first iteration; the same steps repeat at L1202–1206 and L1287–1293):


softmax.template online_softmax</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
convert_type_out(tOrP_acc, tOrP);
if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }
if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
if constexpr (!MmaPV_is_RS) { arrive_on_P_write_barrier(); }

Then the P·V GEMM chooses its A operand (L1182):


flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_pv,
    cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP),   // RS: registers; SS: shared memory
    tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);

Step by step:

#	Step	Code	What happens
1	softmax in registers	online_softmax(tSrS)	tSrS = the Q·Kᵀ accumulator (FP32) becomes P, still FP32
2	reshape (layout only)	make_tensor(tSrS.data(), convert_layout_acc_Aregs<…>(…))	same registers, new layout: C-fragment view → A-fragment view
3	new tensor of the element type	make_tensor_like<Element>(tOrP_acc)	register tensor for P in FP16 / BF16 / FP8
4	convert type	convert_type_out(tOrP_acc, tOrP)	FP32 → Element
5	FP8 only: register shuffle	permute_Cregs_fp8 / permute_Aregs_fp8	data moved between threads to match the FP8 A layout
6	SS only: shared-memory detour	write_P_to_smem + arrive_on_P_write_barrier	the store + barrier that RS avoids
7	GEMM	conditional_return<MmaPV_is_RS>(tOrP, tOsP)	RS reads tOrP (registers); SS reads tOsP (smem)
3. The functions, in hopper/utils.h

convert_layout_acc_Aregs, L150–186. The comment above it (L150–153) states the layout change:

// For SM90, FP16/BF16, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16, MMA_N))
// For SM90, FP8, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((4, 2, 2), MMA_M, (N / 32, MMA_N))

For 16-bit types, it’s a logical_divide of the N/8 mode by 2 (L162–163). That’s the divide operation you studied. It only regroups the same values, without moving them. In step 2, it’s applied to tSrS.data(), the same registers. My reading of why it’s needed (an interpretation, not from a comment): the accumulator layout groups N in pieces of 8, while an A operand with K = 16 per atom needs pairs of those pieces.

For FP8 (L164–176), the source says the reshape alone isn’t correct:

// This combines the first two modes (<0, 0> and <0, 1>) into one mode.
// Will require register shuffling later to be correct.

convert_type_out, L211–224. FP32 → FP16/BF16/FP8, vectorized with CUTLASS’s NumericArrayConverter. The comment at L212 is worth quoting:

// Somehow if we allocate out inside this function and return it, e2e is slower and the output can be wrong.

permute_Aregs_fp8, L453–483. The comment at L453–454:

// Byte permute and shuffle to match register layout of
// (FP8 downcasted) accumulator of GEMM-I to FP8 operand A of GEMM-II.

It uses __shfl_sync within groups of 4 threads (threadIdx.x % 4) and __byte_perm. So for FP8, data really moves between threads, unlike the 16-bit case, which only reshapes.

Summary for your document
Point	Source
Q·Kᵀ is always SS (“register bandwidth is a bottleneck”)	mainloop L80–81
P·V is RS or SS, chosen per head dim / causal / dtype	mainloop L33; tile_size.h L9–54
FP8 or transposed V → P·V must be RS	mainloop L83–84
large head-dim V → P·V must be SS	mainloop L77
trade-off: SS saves registers, costs smem and a P write + barrier	mainloop L82; L1161–1162
FP16/BF16: P = same registers reshaped + converted (no shuffle)	utils.h L150–163, L211–224
FP8: additionally shuffled across threads	utils.h L453–483

Direction check: this covers what the FA document needs on SS vs RS. The source files are saved in /tmp/claude-0/-home-claude/053414b2-e0ce-5679-b0cb-75a776566795/scratchpad/fa3 for this session, if you want me to look up more. The next step on your main track is still the end of the wgmma_sm90.cu main loop.
