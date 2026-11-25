// region_stamp.h — per-region end timestamps inside one fused HIP kernel.
//
// Layout: one u64 slot per (CTA, slot). Slot 0 is the CTA's entry time, slots 1..N-1 are the end
// times of the regions in program order. Host reads the buffer back and runs
// tools/region_timeline.py.
//
// Build: compile the stamps in with -DREGION_STAMP=1. With the default (0) every macro is empty,
// so the timed build has no extra instructions; take latency numbers only from that build.
//
// Clock: wall_clock64() reads s_memrealtime, a constant-rate counter (100 MHz on MI300X/MI355X;
// read the rate with hipDeviceAttributeWallClockRate, kHz). Do not use clock64() (shader clock,
// changes with DPM). Measure the cross-XCD skew once (tools/examples/stamp_demo.hip --skew) before
// comparing CTAs on different XCDs.
//
// Two marking forms:
//   RS_MARK(buf, nslots, slot)       __syncthreads(), then thread 0 stamps: "the whole CTA has left
//                                    the region". The barrier can change the schedule; use it where
//                                    the kernel already has a barrier or the region ends in one.
//   RS_MARK_NOSYNC(buf, nslots, slot) thread 0 stamps when wave 0 gets there. No barrier. Use inside
//                                    wave-specialized code or where a barrier would deadlock.
// Stores issued in the region may still be in flight at the stamp; stamp after the s_waitcnt /
// release that publishes them if the region's end means "visible to the consumer".
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef REGION_STAMP
#define REGION_STAMP 0
#endif

#if REGION_STAMP
__device__ __forceinline__ void rs_store(uint64_t* buf, int nslots, int slot) {
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    const size_t cta = blockIdx.x + (size_t)gridDim.x * (blockIdx.y + (size_t)gridDim.y * blockIdx.z);
    buf[cta * nslots + slot] = wall_clock64();
  }
}
#define RS_MARK(buf, nslots, slot) do { __syncthreads(); rs_store((buf), (nslots), (slot)); } while (0)
#define RS_MARK_NOSYNC(buf, nslots, slot) rs_store((buf), (nslots), (slot))
#else
#define RS_MARK(buf, nslots, slot) do { } while (0)
#define RS_MARK_NOSYNC(buf, nslots, slot) do { } while (0)
#endif
