# FlyDSL API cheat sheet

Imports: `import flydsl.compiler as flyc`, `import flydsl.expr as fx`, and from `flydsl.expr` the modules
`arith`, `gpu`, `rocdl`, `math`, `const_expr`, `range_constexpr`. Vectors: `Vec = fx.Vector`.
Names follow ROCm/FlyDSL `main`. Some have an older spelling, noted below.

## 1. Kernel, launcher, arguments

```python
@flyc.kernel
def my_kernel(A: fx.Tensor, B: fx.Tensor, N: fx.Int32, BLOCK: fx.Constexpr[int]):
    ...

@flyc.jit
def launch(A: fx.Tensor, B: fx.Tensor, N: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    my_kernel(A, B, N, 256).launch(grid=(N // 256,), block=(256,), stream=stream)
```

| Type | Meaning |
|---|---|
| `fx.Tensor` | A torch tensor passed through DLPack as a memref descriptor |
| `fx.Constexpr[T]` | Baked into the IR. Each distinct value is a separate compiled kernel |
| `fx.Int32` | A runtime i32 (Python ints convert automatically) |
| `fx.Stream` | `fx.Stream(None)` = default stream; pass the current torch stream for graph capture |

- `launch(grid=..., block=..., smem=..., stream=...)`: `grid` and `block` take an int, an IR value or a
  1–3 tuple. Grids may be dynamic (`M // 256` with `M: fx.Int32`).
- Hot loops: `compiled = flyc.compile(launch, *args)`, then `compiled(*runtime_args)`.
- Dynamic leading dimension: `flyc.from_dlpack(t).mark_layout_dynamic(leading_dim=0, divisibility=4)`.
- Compile hints used by shipped kernels: `waves_per_eu`, fast / unsafe FP math (see `qk_norm_rope_quant.py`).

## 2. Indices

`gpu.thread_id("x")`, `gpu.block_id("x")` (preferred), or the typed `gpu.thread_idx.x`, `gpu.block_idx.x`,
`gpu.block_dim.x`, `gpu.grid_dim.x`, `fx.thread_idx.x`, `fx.block_idx.x`. `bid_x, bid_y, _ = fx.block_idx`.

## 3. Values, math, vectors

- Constants: `fx.Int32(0xFF)`, `fx.Int64(42)`, `fx.Float32(3.14)` (`fx.Index` is deprecated → `fx.Int64`).
- Operators: `+ - * // % & ^ <<`, comparison, `cond.select(t, f)`.
- `fx.max` / `fx.min` (NaN-propagating), `fx.maxnumf` / `fx.minnumf` (non-NaN wins), `fx.ceildiv`
  (not the layout op `fx.ceil_div`), `abs(v)` / `fx.absf(v)`.
- `math.rsqrt(x, fastmath=...)`; single-instruction, less precise: `rocdl.exp2(T.f32, x)`, `rocdl.rcp(T.f32, x)`.
- Fast-math scope: `with fx.fastmath(arith.FastMathFlags.fast): ...`.
- Vectors: `Vec.filled(4, 0.0, fx.Float32)`, `Vec.from_elements([a, b], fx.Float32)`, `v[i]`,
  `v.to(fx.BFloat16)`, `v.bitcast(fx.Int32)`, `v.reduce(ReductionOp.ADD, fastmath=...)`, `v.shuffle(v, [0, 1, 2, 3])`,
  `v.store(memref, [idx])`.
- fp8 packing: `rocdl.cvt_pk_fp8_f32(i32, a, b, old, hi)` packs two f32 into an i32 half.
- Debug print: `fx.printf("tid={} v={}", tid, v)`.

## 4. Control flow

| Form | Meaning |
|---|---|
| `for i in range_constexpr(N)` | Unrolled at compile time; the index can index Python lists |
| `for i in range(runtime_n)` | Runtime loop (`scf.for`); the index is an IR value |
| `for iv, st in range(fx.Int64(0), fx.Int64(n), fx.Int64(1), init=init):` … `results = yield [...]` | Loop-carried state (software pipelining). Bounds MUST be typed: Python ints unroll and drop `init=` silently |
| `if const_expr(flag):` | Compile-time branch |
| `if lane == c0:` | Runtime branch (`scf.if`) |

Frontend rules: do not define a variable inside `if`/`else` and use it after the branch; no early
`return` and no `return` inside a branch; nested helpers must not mutate captured outer variables;
never `const_expr` a thread-derived value. `FLYDSL_DEBUG_AST_DIFF=1` shows the rewritten AST.

## 5. Layouts

A layout is `(Shape, Stride)`: index = coordinate · stride.
`fx.make_layout(shape, stride)`, `make_shape`, `make_stride`, `make_coord`, `fx.crd2idx`, `fx.idx2crd`,
`size`, `cosize`, `rank`, `composition`, `logical_divide`, `zipped_divide`, `raked_product`, `fx.slice`,
`fx.make_view(fx.get_iter(buf), layout)`, `fx.get(coord, i)`.

Element-wise idiom: `fx.logical_divide(A, fx.make_layout(BLOCK, 1))` → `fx.slice(t, (None, bid))` →
divide again per thread (`fx.make_layout(VEC, 1)`) → `fx.copy(atom, src_slice, reg)`.

## 6. Memory and copies

| API | Use |
|---|---|
| `fx.rocdl.make_buffer_tensor(t)` | Buffer descriptor (`buffer_load` path). OOB checks: `num_records_bytes=` (RDNA) or `max_size=False` |
| `fx.make_copy_atom(op, dtype_or_bits)` | One hardware copy. `fx.UniversalCopy32b()`, `fx.UniversalCopy(64/128)`, `fx.rocdl.BufferCopy32b/64b/128b()`, `fx.rocdl.BufferCopy(64)` |
| `fx.copy(atom, src, dst)` / `fx.copy_atom_call(...)` | Execute a copy |
| `fx.make_rmem_tensor(n, dtype)`, `fx.make_fragment_like(src)` | Register tensors and fragments |
| `fx.memref_load_vec(r)`, `fx.memref_store_vec(v, r)` | Vector access to a register tensor |
| `fx.make_layout_tv(thr, val)` → `fx.make_tiled_copy(atom, tv, tile_mn)`, or `fx.make_tiled_copy_tv(atom, thr, val)` | Tiled copy. Do not replace `make_layout_tv` with a plain `raked_product` |
| `tiled_copy.get_slice(tid).partition_S(src)` / `.partition_D(dst)` | Per-thread views |
| `tiled_copy.set_value("soffset", off)` | New tiled copy with a field set |
| `fx.rocdl.s_buffer_load(rsrc, off, count=1)` | Wave-uniform scalar load (offset in i32 elements) |
| `rocdl.cdna5.make_tiled_tdm_atom(rocdl.TensorLoad(), gA, box, (TM, TK))`, `tdm_partition`, `s_wait_tensorcnt(0)` | gfx1250 async global ↔ LDS (TDM) |

- Copy width: `copy_bits = VEC_WIDTH * elem_bits`; the per-thread divide and the register tensor use the
  same `VEC_WIDTH`.
- Raw `buffer_load` / `buffer_store` offsets are in **elements**, not bytes (legacy helpers now live in
  `kernels/common/buffer_ops.py`).

## 7. LDS

```python
@fx.struct
class SharedStorage:
    a: fx.Array[fx.Float16, 8192]
lds = fx.SharedAllocator().allocate(SharedStorage).peek()      # at the TOP of the kernel
lds_a = lds.a.view(fx.make_layout((64, 128), (128, 1)))
```
Static mode (default) sizes LDS for you: leave `launch(smem=...)` unset. Capacity per CU: 64 KB (gfx942,
gfx1201), 160 KB (gfx950), 320 KB (gfx1250).

## 8. Reductions and cooperative ops

- Wave reduce: `fx.coop.warp_reduce(x, fx.ReductionOp.ADD, width=W)`; manual form
  `peer = gpu.shuffle_xor(v, sh, fx.Int32(64))` for `sh` in 32, 16, 8, 4, 2, 1.
- Block reduce: `BlockReduce = fx.coop.BlockReduce[fx.Float32, BLOCK_THREADS]`, storage
  `BlockReduce.SharedStorage` allocated in LDS, call `BlockReduce(v, fx.ReductionOp.ADD, storage=lds)`.
  Two values in one reduction: `fx.coop.BlockReduce[fx.Float32x2, ...]` with a 2-element vector.
- Cross-lane pair (RoPE partner): `ds_bpermute`, address = partner lane × 4 bytes, 16-bit values widened
  to i32 first.

## 9. MMA

| API | Target |
|---|---|
| `fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.Float16))` | CDNA3/4 MFMA (fp32 accumulator) |
| `rocdl.MFMA(m=16, n=16, k=32, elem_ty_ab=fx.Float8E4M3FNUZ)` | fp8 MFMA |
| `fx.rocdl.cdna4.MFMA_Scale` | CDNA4 scaled MFMA |
| `fx.make_mma_atom(rocdl.WMMA(16, 16, 32, fx.Float16))` | gfx1250 (wave32); RDNA4 only 16×16×16 |
| `rocdl.WMMAScale(16, 16, 128, fx.Float8E4M3FN, block_size=32)` + `fx.atom_set_value(mma, "scale_a", s)` | Scaled WMMA |
| `make_tiled_mma(atom, layout)` with layout `(M_rep, N_rep, K_rep)`, `tiled_mma.thr_slice(tid)` | Tiled MMA |
| `fx.gemm(mma, frag_C, frag_A, frag_B, frag_C)` (d, a, b, c) | Lowers repeated atoms into the m/n/k loops |
| Raw: `rocdl.mfma_f32_16x16x16f16(T.vec(4, T.f32), [a, b, acc, 0, 0, 0])` | Accumulator passed in and reassigned |

`retile(frag)` gives the copy view of an MMA fragment: copies use the retiled view, MMA uses the
fragment view. Per-tile scales: pass `[frag_A, scales_A]`.

## 10. Synchronization and scheduling

`gpu.barrier()` (every thread must reach it); `fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)`
(vmcnt + lgkmcnt does not cover expcnt); gfx1250 split counters (`s_wait_loadcnt(0)`);
scheduling hints `rocdl.sched_mfma(n)`, `sched_vmem(n)`, `sched_dsrd(n)`, `sched_dswr(n)`, `sched_barrier(0)`.

## 11. Cache and environment

| Variable | Effect |
|---|---|
| `FLYDSL_RUNTIME_ENABLE_CACHE=0` | Disable the kernel cache (`~/.flydsl/cache/`, or `$FLYDSL_RUNTIME_CACHE_DIR`) |
| `FLYDSL_DUMP_IR=1` | Dump IR (bypasses the cache) |
| `FLYDSL_DEBUG_AST_DIFF=1` | Show original vs rewritten AST |
| `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` | Debug info for ATT / source mapping |
| `FLYDSL_GPU_ARCH` | Force the target arch when torch cannot see the GPU |

The cache key is a SHA-256 of the prepared MLIR, target, pipeline and options: Python argument values
alone do not select an artifact, and edited Python helpers need a reload or a new process.
