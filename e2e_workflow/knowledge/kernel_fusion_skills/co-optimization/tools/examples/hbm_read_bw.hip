// hbm_read_bw.hip — practical read bandwidth for floor calculations (09 §3).
//
//   hipcc -O3 --offload-arch=gfx950 hbm_read_bw.hip -o hbm_read_bw && ./hbm_read_bw
//
// Each CTA reads one contiguous chunk with 16 B nontemporal loads (the pattern that reached the
// read ceiling on MegaMoE). Report the number at sizes far above the 256 MB MALL as the HBM floor
// bandwidth. Small sizes read lower here (launch overhead dominates); this program does not show the
// MALL effect seen on graph-replayed small working sets (`09` §3.1).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

typedef float f4 __attribute__((ext_vector_type(4)));  // native vector: nontemporal builtin needs it

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(1); } } while (0)

__global__ void __launch_bounds__(256) read_chunks(const f4* __restrict__ p, size_t per_cta, float* out) {
  const f4* q = p + blockIdx.x * per_cta;
  float acc = 0.f;
  #pragma unroll 4
  for (size_t i = threadIdx.x; i < per_cta; i += 256) {
    f4 v = __builtin_nontemporal_load(q + i);
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 1234.5f) out[blockIdx.x] = acc;  // keep the loads alive
}

int main() {
  int dev = 0, ncu = 0;
  CHECK(hipGetDevice(&dev));
  CHECK(hipDeviceGetAttribute(&ncu, hipDeviceAttributeMultiprocessorCount, dev));
  const size_t max_bytes = (size_t)8 << 30;
  f4* buf; float* out;
  CHECK(hipMalloc(&buf, max_bytes)); CHECK(hipMalloc(&out, 4096 * 4));
  CHECK(hipMemset(buf, 0, max_bytes));
  hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
  const size_t sizes_mb[] = {16, 64, 128, 512, 2048, 8192};
  for (int grid_mult : {4, 8}) {
    const int grid = ncu * grid_mult;
    for (size_t mb : sizes_mb) {
      const size_t bytes = mb << 20, per_cta = bytes / 16 / grid;
      read_chunks<<<grid, 256>>>(buf, per_cta, out);
      CHECK(hipDeviceSynchronize());
      const int iters = 20;
      CHECK(hipEventRecord(a));
      for (int i = 0; i < iters; ++i) read_chunks<<<grid, 256>>>(buf, per_cta, out);
      CHECK(hipEventRecord(b)); CHECK(hipEventSynchronize(b));
      float ms; CHECK(hipEventElapsedTime(&ms, a, b));
      const double tbs = (double)per_cta * 16 * grid * iters / (ms * 1e-3) / 1e12;
      printf("grid %4d  %6zu MB  %8.1f us/read  %5.2f TB/s\n", grid, mb, ms * 1e3 / iters, tbs);
    }
  }
  return 0;
}
