# Mega skill link (placeholder)

This file is where KernelFusion points a kernel lane that writes an **L3 MegaKernel** fused kernel (one
persistent kernel that absorbs a heavy expert/GEMM or attention stage together with its dispatch, combine or
communication, e.g. MegaMoE) for the skill that says how to write one.

**The Mega skill is not here yet.** The user will supply its address and files later. When they do, replace
the link below with the real address or file path, or pass the skill's path as `kf_mega_skill`
(handoff `kernel_fusion.mega_skill`, env `GEAK_KF_MEGA_SKILL`), which overrides this file.

- Mega skill: `<to be supplied: address or file path>`

Until then, a lane writing an L3 MegaKernel follows the Mega chapter of `../fusion-kernel/SKILL.md` and the
`[multi-rank]` items of this directory's co-optimization skill (`references/02_in_fusion_levers.md` B10, B14,
C10–C11; `references/04_rejected_and_pitfalls.md` trap 13; `references/08_shipping_a_fused_kernel.md`).
