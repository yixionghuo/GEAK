// stamp_demo.hip — minimal persistent kernel with region stamps, and a cross-XCD clock-skew probe.
//
//   hipcc -O3 --offload-arch=gfx950 -DREGION_STAMP=1 -I.. stamp_demo.hip -o stamp_demo
//   ./stamp_demo out_dir            # writes out_dir/stamps_rank0_rep<i>.bin + meta.json
//   python3 ../region_timeline.py out_dir --wait wait:publish
//   ./stamp_demo out_dir --skew     # entry/exit spread across CTAs after a grid-wide join
//
// Regions (slot: name):
//   0 entry   1 load (stream a contiguous chunk, HBM bound)   2 publish (partial + counter add)
//   3 wait (spin until every CTA published this epoch)        4 reduce (read all partials)
// CTA k streams (1 + k % 4) chunks, so the producers are deliberately unbalanced and the wait
// region of every CTA is gated by the slowest producer — the critical path must show that.
#include "region_stamp.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

typedef float f4 __attribute__((ext_vector_type(4)));  // native vector: nontemporal builtin needs it

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(1); } } while (0)

constexpr int NSLOTS = 5;
constexpr int THREADS = 256;

__global__ void __launch_bounds__(THREADS)
fused_demo(const f4* __restrict__ src, size_t chunk_f4, float* partial, unsigned* counter,
           unsigned epoch, float* out, uint64_t* stamps) {
  RS_MARK_NOSYNC(stamps, NSLOTS, 0);
  const int cta = blockIdx.x, ncta = gridDim.x;
  // load: stream (1 + cta % 4) chunks
  float acc = 0.f;
  const int nchunks = 1 + cta % 4;
  for (int c = 0; c < nchunks; ++c) {
    const f4* p = src + ((size_t)cta * 4 + c) * chunk_f4;
    for (size_t i = threadIdx.x; i < chunk_f4; i += THREADS) {
      f4 v = __builtin_nontemporal_load(p + i);
      acc += v.x + v.y + v.z + v.w;
    }
  }
  __shared__ float red[THREADS];
  red[threadIdx.x] = acc;
  RS_MARK(stamps, NSLOTS, 1);
  // publish: partial, release, one counter add per CTA
  for (int s = THREADS / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial[cta] = red[0];
    __threadfence();
    atomicAdd(counter, 1u);
  }
  RS_MARK(stamps, NSLOTS, 2);
  // wait: monotone target, never reset (epoch * ncta)
  if (threadIdx.x == 0) {
    const unsigned target = epoch * (unsigned)ncta;
    while (__hip_atomic_load(counter, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < target) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  RS_MARK(stamps, NSLOTS, 3);
  // reduce: every CTA reads all partials (stand-in for a consumer stage)
  float s = 0.f;
  for (int i = threadIdx.x; i < ncta; i += THREADS) s += __hip_atomic_load(partial + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  red[threadIdx.x] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f;
    for (int i = 0; i < THREADS; ++i) t += red[i];
    out[cta] = t;
  }
  RS_MARK(stamps, NSLOTS, 4);
}

// grid-wide join, then stamp: spread of slot 1 = cross-CTA (cross-XCD) clock skew + release jitter
__global__ void skew_probe(unsigned* counter, unsigned epoch, uint64_t* stamps) {
  if (threadIdx.x == 0) {
    stamps[blockIdx.x * 2] = wall_clock64();
    atomicAdd(counter, 1u);
    const unsigned target = epoch * gridDim.x;
    while (__hip_atomic_load(counter, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < target) { }
    stamps[blockIdx.x * 2 + 1] = wall_clock64();
  }
}

static void write_bin(const std::string& path, const std::vector<uint64_t>& v) {
  FILE* f = fopen(path.c_str(), "wb");
  fwrite(v.data(), sizeof(uint64_t), v.size(), f);
  fclose(f);
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s out_dir [--skew]\n", argv[0]); return 1; }
  const std::string out_dir = argv[1];
  const bool skew = argc > 2 && !strcmp(argv[2], "--skew");
  mkdir(out_dir.c_str(), 0755);
  int dev = 0, ncu = 0, wc_khz = 0;
  CHECK(hipGetDevice(&dev));
  CHECK(hipDeviceGetAttribute(&ncu, hipDeviceAttributeMultiprocessorCount, dev));
  CHECK(hipDeviceGetAttribute(&wc_khz, hipDeviceAttributeWallClockRate, dev));
  const int ncta = ncu;  // one CTA per CU: whole grid resident (required by the spin-wait)
  unsigned* counter; CHECK(hipMalloc(&counter, sizeof(unsigned)));
  CHECK(hipMemset(counter, 0, sizeof(unsigned)));
  const int reps = 5;

  if (skew) {
    uint64_t* st; CHECK(hipMalloc(&st, sizeof(uint64_t) * ncta * 2));
    std::vector<uint64_t> h(ncta * 2);
    for (int r = 1; r <= reps; ++r) {
      skew_probe<<<ncta, 64>>>(counter, r, st);
      CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(h.data(), st, h.size() * 8, hipMemcpyDeviceToHost));
      uint64_t lo = ~0ull, hi = 0;
      for (int i = 0; i < ncta; ++i) { lo = std::min(lo, h[2 * i + 1]); hi = std::max(hi, h[2 * i + 1]); }
      printf("rep %d: exit spread after join = %llu ticks = %.0f ns\n", r,
             (unsigned long long)(hi - lo), (hi - lo) * 1e6 / wc_khz);
    }
    return 0;
  }

  const size_t chunk_f4 = (size_t)1 << 16;  // 1 MiB per chunk
  const size_t n_f4 = (size_t)ncta * 4 * chunk_f4;
  f4* src; CHECK(hipMalloc(&src, n_f4 * sizeof(f4)));
  CHECK(hipMemset(src, 0, n_f4 * sizeof(f4)));
  float *partial, *out; CHECK(hipMalloc(&partial, ncta * 4)); CHECK(hipMalloc(&out, ncta * 4));
  uint64_t* st; CHECK(hipMalloc(&st, sizeof(uint64_t) * ncta * NSLOTS));
  std::vector<uint64_t> h(ncta * NSLOTS);
  for (int r = 1; r <= reps; ++r) {  // back to back, like the A/B; read the buffer between reps
    fused_demo<<<ncta, THREADS>>>(src, chunk_f4, partial, counter, r, out, st);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(h.data(), st, h.size() * 8, hipMemcpyDeviceToHost));
    write_bin(out_dir + "/stamps_rank0_rep" + std::to_string(r) + ".bin", h);
  }
  FILE* f = fopen((out_dir + "/meta.json").c_str(), "w");
  fprintf(f, "{\"nslots\": %d, \"ncta\": %d, \"tick_hz\": %lld, "
             "\"names\": [\"entry\", \"load\", \"publish\", \"wait\", \"reduce\"]}\n",
          NSLOTS, ncta, (long long)wc_khz * 1000);
  fclose(f);
  printf("wrote %d reps, %d CTAs, wall clock %d kHz -> %s\n", reps, ncta, wc_khz, out_dir.c_str());
  return 0;
}
