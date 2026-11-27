#!/usr/bin/env node
// KernelFusion in e2e_workflow.js (kernel_fusion=true; default OFF), on CPU: no GPU, no model, no LLM.
// kernel_fusion_mockrun.mjs runs the WHOLE workflow on GEAK's own runtime with scripted agents and stubbed
// kernel_lane.js runs, so these checks see what a real run would send and return:
//   - switched off (unset or "false"), no prompt, log, lane or return field carries any trace of it;
//   - switched on, the fusion profile reads Profile's own trace and leaves Profile as it is, Strategize routes
//     its regions as kernel candidates, and the Milestone loop builds, optimizes (kernel_lane.js) and
//     measures each one with GEAK's own extract, integrate A/B, Finalize-gate and Validate;
//   - HeadKernel runs gain-ordered slots: a head with the regions on its own kernels, every other region a slot of
//     its own; inside a slot the candidates run best estimate first and meet in one A/B (KF off: GEAK's own order);
//   - an L3 MegaKernel region runs its UT and lane on a GPU group (or is unprepared when the run has too few GPUs),
//     and a recommended region outside kf_implementable_levels is dropped;
//   - every outcome (accepted, pending, overlay rejected, unknown region, failed or unconfigured profile,
//     fast/deep mode, a given topk, AgentX, a resumed run) ends cleanly and the run goes on to Validate.
//
// Run:  node e2e_workflow/scripts/test_kernel_fusion.js
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const SCRIPTS = __dirname;
const WF_DIR = path.resolve(SCRIPTS, '..');
const src = fs.readFileSync(path.join(WF_DIR, 'e2e_workflow.js'), 'utf8');
const EV = '/tmp/kf_mock/eval';

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };
const runs = {};
const run = (scenario, env = {}) => {
  if (runs[scenario]) return runs[scenario];
  const p = spawnSync(process.execPath, [path.join(SCRIPTS, 'kernel_fusion_mockrun.mjs'), scenario],
    { encoding: 'utf8', timeout: 180000, maxBuffer: 256 * 1024 * 1024, env: { ...process.env, ...env } });
  if (p.status !== 0) throw new Error(`mock run ${scenario} exited ${p.status}: ${p.stderr}`);
  const r = JSON.parse(p.stdout.trim().split('\n').pop());
  if (!r.ok) throw new Error(`mock run ${scenario} threw: ${r.error}`);
  runs[scenario] = r;
  return r;
};
const labels = (r) => r.calls.map((c) => c.label);
const kfOf = (r) => (r.result && r.result.kernel_fusion) || {};
const promptOf = (r, label) => (r.calls.find((c) => c.label === label) || {}).prompt || '';
const banked = (r) => ((r.result && r.result.accepted_kernels) || []);
const names = (r) => banked(r).map((k) => k.short_name);
const before = (r, a, b) => labels(r).indexOf(a) >= 0 && labels(r).indexOf(a) < labels(r).indexOf(b);
const laneOf = (r, short) => r.lanes.find((l) => String(l.kernel_path || '').endsWith(`/${short}`)) || {};
// What a run carries of KernelFusion, wherever it would show: none of these appears in a run switched off.
const KF_MARKS = /FUSION_STRATEGY|FUSION_CANDIDATES|FUSION_SCOPE|FUSION REGION|fusion_analyst|fusion_ut_contract|fusion-kernel|KERNEL_FUSION|fusion_overlay_check|"op_kind": ?"fusion"|\bkf_c\d|\[kf\]/;
// A failed fusion profile must leave the run as off_kernels': same calls (but the analyst's own, and the
// final persists, which carry the return), same logs (agent counters aside), same result (but the record).
const END = /^(persist-workflow-return|file_writer:persist_llm_stats)/;
// The report's input line carrying the record is the record too (§2d says why the profile did not run).
const bareCalls = (r) => r.calls.filter((c) => !c.label.startsWith('fusion_analyst') && !END.test(c.label))
  .map((c) => `${c.label}\n${c.prompt.replace(/\n- KERNEL_FUSION: [^\n]*/, '')}`);
const bareLogs = (r) => r.logs.filter((l) => !l.startsWith('[kf]') && !l.includes('fusion_analyst'))
  .map((l) => l.replace(/\d+ spawned/, 'N spawned').replace(/\d+ agent attempts/, 'N agent attempts'));
const bareResult = (r) => {
  const o = JSON.parse(JSON.stringify(r.result));
  delete o.kernel_fusion;
  if (o.state) delete o.state.kernel_fusion;
  return o;
};
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const HOT = [{ kernel: 'gemm_kernel', pct_gpu_time: 12, single_kernel_gain_pct_expected: 1.2, basis: 'untuned' },
  { kernel: 'gemm_bf16_kernel', pct_gpu_time: 20, single_kernel_gain_pct_expected: 2.5, basis: 'tuned lib' }];

console.log('## A. structure: no phase, hooks into the stages GEAK already has');
const meta = (src.match(/phases:\s*\[[\s\S]*?\n {2}\],/) || [''])[0];
ok(meta.length > 0 && !/KernelFusion|Fusion/.test(meta), 'meta.phases is GEAK\'s own: KernelFusion adds no phase');
ok(!/phase\('KernelFusion'\)|want\('fusion'\)/.test(src), 'no KernelFusion phase() or phase selector');
ok(/const KF_ON = String\(A\.kernel_fusion != null \? A\.kernel_fusion : 'false'\) === 'true';/.test(src),
  'the switch defaults to off and only the string "true" turns it on');
ok(/label: 'profiler:baseline'[^\n]*\n\s*log\(`Baseline profiled\.[^\n]*\n\s*if \(KF_ON\) await kfProfile\('Profile'\);/.test(src),
  'fixed load: the fusion profile follows profiler:baseline, in the Profile step');
ok(/profile = strategy;\s*if \(KF_ON\) \{\s*await kfProfile\('KernelExtract'\);\s*strategy = await kfPrepareRegions\(/.test(src),
  'AgentX: the fusion profile follows the baseline preparation, then the region UTs join the inventory');
ok((src.match(/\.\.\.\(KF_ON && KF_INPUTS\),/g) || []).length === 4,
  'its regions reach Strategize, both re-strategizes and plan_milestone (KF_INPUTS, {} until there are regions)');
ok(['...(KF_ON ? kfExpand(c) : c), idx: i',
  "(KF_ON && kfExtractIntro(c)) || 'Capture shapes + oracle; emit an immutable unittest task dir.'",
  '...(KF_ON && kfExtractInputs(c)),', "task: (KF_ON && kfLaneTask(c)) || 'Compare candidate backends '",
  '...(KF_ON && kfResultFields(c)) },', '...(KF_ON && kfAccuracyFor(c)),'].every((s) => src.split(s).length === 2),
  'the Milestone loop treats a region as a kernel candidate: one hook each in plan, extract, lane and integrate');
const count = (s) => src.split(s).length - 1;
ok(count('if (KF_ON && slot.kf) headCands.push(...(await kfSlotCands(slot, true)));') === 1 &&
  count('if (KF_ON && slot.kf) headCands.push(...(await kfSlotCands(slot, false)));') === 1 &&
  count('const slots = (KF_ON && !FAST_MODE && kfRegions.size) ? kfSlots() : heads.map((h) => ({ h }));') === 1 &&
  count('const ext = (KF_ON && !headExt) ? kfSlotExt(headCands) : headExt;') === 1 && count('break headRoute;') === 4 &&
  count('kernel_candidates: KF_ON ? kfByGain(kernelQueue) : kernelQueue') === 2 &&
  count('...(KF_ON && kfHeadResult(cand)) },') === 1 && count('...(KF_ON && kfAccuracyFor(cand.kf)),') === 1 &&
  count('(KF_ON && kfHeadPct(cand)) || h.pct_gpu_time') === 4 && count('...(KF_ON && kfHeadSpec(cand)),') === 2,
  'HeadKernel takes a region as one more candidate: its A/B inputs, gate share, pending entry and correctives');
// Outside its own blocks every KernelFusion reference sits behind KF_ON, so a GEAK test that slices one
// function out of the workflow needs only KF_ON = false for it.
const cut = (s, from, to) => {
  const a = s.indexOf(from), b = s.indexOf(to, a);
  return a < 0 || b < 0 ? '' : s.slice(0, a) + s.slice(b);
};
const outside = [['// ---- KERNEL FUSION (opt-in switch', '// Kernel-layer warm start:'],
  ['// ---- KernelFusion (kernel_fusion=true)', "if (want('setup')) {"],
  ['// KernelFusion record (result["kernel_fusion"]', '// State to carry into the NEXT phase invocation']]
  .reduce((s, [from, to]) => s && cut(s, from, to), src);
const KF_NAME = /\bkf[A-Z]\w*|\bKF_(?!ON\b)[A-Z_]+/;
let guardIndent = null;   // inside an `if (KF_ON) {` block
const kfLines = [], unguarded = [];
for (const l of outside.split('\n')) {
  const indent = l.search(/\S/);
  if (guardIndent != null && l.trim() === '}' && indent === guardIndent) { guardIndent = null; continue; }
  if (guardIndent == null && /\bif \(KF_ON\) \{$/.test(l.trim())) { guardIndent = indent; continue; }
  if (!KF_NAME.test(l)) continue;
  kfLines.push(l);
  const g = l.search(/\bKF_ON\b/);
  if (guardIndent == null && !(g >= 0 && g < l.search(KF_NAME))) unguarded.push(l);
}
ok(outside.length > 0 && kfLines.length >= 15 && unguarded.length === 0,
  `outside its own blocks, all ${kfLines.length} lines naming KernelFusion read KF_ON first` +
  (unguarded.length ? `: ${unguarded.map((l) => l.trim()).join(' | ')}` : ''));
ok(/integ = await reviewSpeedup\(integ, withTuning, label, phaseName\);\s*if \(KF_ON\) integ = await kfOverlayGate\(integ, withTuning, phaseName\);/.test(src),
  'every integrate A/B (Milestone and Finalize-gate alike) runs the overlay check on an accepted region');
ok(/if \(KF_ON && k\.op_kind === 'fusion'\) kfBanked\(list\[list\.length - 1\]\);/.test(src) &&
  !/\bkind: 'fusion'/.test(src), 'a banked region stays kind "patch" (the KB replay reads it) with an op_kind tag');
ok(/fusion_analyst:analyze', schema: KF_OPPORTUNITY_SCHEMA \}, 2, kfTimeout\(3600000\)\)/.test(src) &&
  (src.match(/1, kfTimeout\(Infinity\)\)/g) || []).length === 2,
  'what runs ahead of the search is bounded: the analysis 2 x <= 1 h, each region UT call one attempt');
const block = src.slice(src.indexOf('const KF_SKILL_DIR ='), src.indexOf("if (want('setup')) {"));
ok(block.length > 10000 && !/Date\.now|require\(|process\./.test(block),
  'the KernelFusion helpers use no construct the workflow sandbox forbids');
ok(!fs.existsSync(path.join(WF_DIR, 'kernel_fusion')) && !/kf_workflow\.js/.test(src),
  'no standalone KernelFusion workflow or module dir');
const role = fs.readFileSync(path.join(WF_DIR, 'roles', 'fusion_analyst.md'), 'utf8');
ok(role.includes('## PHASE=analyze') && (role.match(/## PHASE=/g) || []).length === 1,
  'roles/fusion_analyst.md: one phase, the fusion profile (the reviews moved to the architect)');
const strat = fs.readFileSync(path.join(WF_DIR, 'roles', '_fragments', 'fusion_strategy.md'), 'utf8');
ok(/Feasibility Review/.test(strat) && /never `head_candidates`/.test(strat) && /exclusive_with/.test(strat) &&
  /route at most one of them at a time/.test(strat), 'roles/_fragments/fusion_strategy.md: the routing rules');
ok(/stays in `head_candidates` exactly as it would without the\s+region/.test(strat) && /`<head short_name>:kf_<id>`/.test(strat),
  'roles/_fragments/fusion_strategy.md: a head op keeps its head route, and HISTORY names a region tried in its slot');
const arch = fs.readFileSync(path.join(WF_DIR, 'roles', 'system_architect.md'), 'utf8');
ok(/1e\. \*\*Fusion regions \(only if `FUSION_STRATEGY` is in your inputs; otherwise skip this step entirely/.test(arch),
  'the architect routes fusion regions only when FUSION_STRATEGY is in its inputs');

console.log('\n## B. switched off: no trace of it anywhere');
for (const s of ['off', 'off_false', 'off_kernels', 'off_agentx', 'off_head', 'off_agentx_head', 'off_head_nowin', 'off_head_unrun']) {
  const r = run(s);
  const hits = [...r.calls.map((c) => c.prompt), ...r.logs, ...r.lanes.map((l) => JSON.stringify(l))]
    .filter((t) => KF_MARKS.test(t));
  ok(hits.length === 0, `${s}: no prompt, log or lane mentions KernelFusion`);
  ok(!('kernel_fusion' in r.result) && !('kernel_fusion' in (r.result.state || {})), `${s}: no kernel_fusion return or state`);
}
ok(same(run('off').calls, run('off_false').calls) && same(run('off').logs, run('off_false').logs) &&
  same(run('off').result, run('off_false').result), 'kernel_fusion "false" is exactly unset');
ok(labels(run('off_kernels')).includes('finish-integrate mk_gemm'), 'off_kernels: GEAK\'s own Finalize-gate replay ran');
ok(labels(run('off_head')).includes('integrate h_gemm:aiter') && labels(run('off_agentx_head')).includes('integrate h_gemm:aiter'),
  'off_head, off_agentx_head: GEAK\'s own head track ran');

console.log('\n## C. switched on: profile, route, optimize, measure');
const on = run('on');
const offk = run('off_kernels');
ok(before(on, 'profiler:baseline', 'fusion_analyst:analyze') && before(on, 'fusion_analyst:analyze', 'architect:strategize'),
  'profiler:baseline -> fusion_analyst:analyze -> architect:strategize');
ok(promptOf(on, 'profiler:baseline') === promptOf(offk, 'profiler:baseline'), 'the Profile round is the one GEAK runs anyway');
const an = promptOf(on, 'fusion_analyst:analyze');
ok(an.includes(`${EV}/profile/round_0/profile`) && an.includes(`${EV}/profile/round_0/profile_topN.json`) &&
  an.includes(`${EV}/kernel_fusion/analysis`), 'the analysis reads Profile\'s own trace and Top-N, and writes under EVAL_DIR');
const st = promptOf(on, 'architect:strategize');
ok(['FUSION_STRATEGY', 'fusion_strategy.md', 'FUSION_CANDIDATES', 'FUSION_RECOMMENDED', 'FUSION_TOPK_JSON',
  'IMPLEMENTABLE_LEVELS', 'UNSUPPORTED_PATTERNS'].every((k) => st.includes(k)), 'Strategize gets the fusion profile');
ok(!st.includes('fused_design') && !st.includes('one pass: normalize'), 'the architect routes on the slim region facts');
const ex = promptOf(on, 'extract kf_c2');
ok(/FUSION REGION/.test(ex) && ex.includes('fusion_ut_contract.md') && ex.includes('pkg.norm:fused_norm_quant') &&
  ex.includes('quant_kernel'), 'extract kf_c2: the region UT under the fusion UT contract, with the region\'s facts');
ok(promptOf(on, 'extract mk_gemm') === promptOf(offk, 'extract mk_gemm'), 'a single kernel beside it is extracted as before');
const lane = laneOf(on, 'kf_c2');
ok(/kernel_workflow\/kernel_lane\.js$/.test(lane.script) && /^FUSION REGION c2 \(L1\): write ONE fused triton kernel/.test(lane.task) &&
  lane.task.includes('kernel_fusion_skills/fusion-kernel/SKILL.md') && lane.task.includes('one pass: normalize'),
  'the fused kernel is optimized by GEAK\'s kernel_lane.js with the region task and the fusion-kernel skill');
const co = (f) => `kernel_fusion_skills/co-optimization/${f}`;
ok(lane.task.indexOf(co('fusion_kernel_prompt.md')) >= 0 && lane.task.indexOf(co('fusion_kernel_prompt.md')) < lane.task.indexOf(co('SKILL.md')) &&
  lane.task.indexOf(co('SKILL.md')) < lane.task.indexOf('fusion-kernel/SKILL.md') && lane.task.includes(`${WF_DIR}/knowledge/`) &&
  lane.task.includes('"est_gain_pct_expected":1.5'),
  'the lane reads the co-optimization prompt, then its SKILL.md, then the fusion-kernel skill (absolute paths)');
const coDir = path.join(WF_DIR, 'knowledge', 'kernel_fusion_skills', 'co-optimization');
const prm = fs.readFileSync(path.join(coDir, 'fusion_kernel_prompt.md'), 'utf8');
const fks = fs.readFileSync(path.join(WF_DIR, 'knowledge', 'kernel_fusion_skills', 'fusion-kernel', 'SKILL.md'), 'utf8');
ok(!/\{\{SKILL_DIR\}\}/.test(prm) && !/\{\{[^}]*\}\}/.test(prm.replace('`{{...}}`', '')) &&
  ['meta.json', 'kernel_src/_handback/design.md', 'kernel_src/_handback/baseline.json', 'refusals', 'notes.md'].every((k) => prm.includes(k)) &&
  ['SKILL.md', 'references/01_decide_and_order.md', 'references/09_mid_fusion_diagnosis.md', 'tools/region_stamp.h']
    .every((f) => fs.existsSync(path.join(coDir, f))),
  'co-optimization: the skill package is in place, the prompt\'s input slots and outputs are mapped onto GEAK\'s');
ok(['t_fused <= t_floor', '§0.1', '§0.2', '`02`', '`04`', '`09`', 'five columns', 'one of three verdicts', 'not a reason to fuse',
  'Parity before levers', 'baseline.json'].every((k) => fks.includes(k)) &&
  !fs.existsSync(path.join(WF_DIR, 'knowledge', 'kernel_fusion_skills', 'fusion-kernel', 'references', 'tune-then-fuse.md')),
  'fusion-kernel: steps 0-7 of the prompt (tune-then-fuse folded into step 0)');
ok(fs.existsSync(path.join(WF_DIR, 'knowledge', 'kernel_fusion_skills', 'fusion-kernel', 'SKILL.md')) &&
  !fs.existsSync(path.join(WF_DIR, 'roles', '_fragments', 'fused_kernel_task.md')),
  'writing the fused kernel is a skill (fusion-kernel), not a prompt fragment');
const rep = promptOf(on, 'architect:report');
ok(rep.includes('KERNEL_FUSION') && rep.includes('"routed":["kf_c2"]') && !promptOf(offk, 'architect:report').includes('KERNEL_FUSION'),
  'the report gets the KernelFusion record (its §2d section) only when the switch is on');
ok(/^Compare candidate backends/.test(laneOf(on, 'mk_gemm').task), 'the single kernel\'s lane task is GEAK\'s own');
const ik = promptOf(on, 'integrate kf_c2');
ok(/"op_kind": ?"fusion"/.test(ik) && /"fusion_id": ?"c2"/.test(ik) && ik.includes('live_call_seam') &&
  /ACCURACY_GATE/.test(ik), 'integrate kf_c2: GEAK\'s A/B with the region in KERNEL_RESULT and the accuracy gate');
ok(before(on, 'integrate kf_c2', 'overlay-check kf_c2') && before(on, 'overlay-check kf_c2', 'integrate mk_gemm') &&
  promptOf(on, 'overlay-check kf_c2').includes(`--overlay "${EV}/overlay/cand_kf_c2"`),
  'the accepted overlay is checked on disk before anything else runs on it');
const kc2 = banked(on).find((k) => k.short_name === 'kf_c2') || {};
ok(kc2.kind === 'patch' && kc2.op_kind === 'fusion' && kc2.fusion_id === 'c2' && names(on).includes('mk_gemm'),
  'banked: kf_c2 (kind patch, op_kind fusion) and mk_gemm');
ok(/ACCURACY_GATE/.test(promptOf(on, 'integrate mk_gemm')) && /ACCURACY_GATE/.test(promptOf(on, 'director:validate')) &&
  !/ACCURACY_GATE/.test(promptOf(offk, 'integrate mk_gemm')) && !/ACCURACY_GATE/.test(promptOf(offk, 'director:validate')),
  'once a fused kernel is in, every later A/B and Validate hold the run to task accuracy');
ok(promptOf(on, 'architect:plan m2').includes('FUSION_STRATEGY'), 'later milestone plans see the fusion profile too');
const k = kfOf(on);
ok(k.status === 'accepted' && same(k.routed, ['kf_c2']) && k.accepted[0].fusion_id === 'c2' && k.soft_parity === true &&
  same(k.outcomes.map((o) => [o.short_name, o.verdict]), [['kf_c2', 'confirmed']]) &&
  k.trace_dir === `${EV}/profile/round_0/profile` && on.result.state.kernel_fusion.status === 'accepted',
  'the record: profile, routed regions, accepts, verdicts (history.ledger); carried in state');
ok(labels(on).includes('director:validate'), 'the run goes on to Validate');

ok(ex.includes('FUSION_ANALYSIS') && ex.includes('"est_gain_pct_expected":1.5') && !promptOf(on, 'extract mk_gemm').includes('FUSION_ANALYSIS'),
  'the region UT gets the region\'s analysis entry (written into the task as analysis.json)');
const utc = fs.readFileSync(path.join(WF_DIR, 'roles', '_fragments', 'fusion_ut_contract.md'), 'utf8');
ok(['## The unified task dir (L1–L4 share it)', '`level`', '`acceptance`', '`regime.speculative`', '**`analysis.json`**',
  'controls/controls.json', 'run/controls_result.json', '**`selfcheck.json`** (every level)', '0.90–1.10',
  '## The hand-back the UT is accepted against'].every((k) => utc.includes(k)) && !utc.includes('No `selfcheck.json`'),
  'fusion_ut_contract.md: the unified task dir, its self-check at every level, and the hand-back it is accepted against');
ok(['## L3 MegaKernel', '`required_gpu_count`', '`tp`', '`ep`', 'weights/rank<r>/', 'reference_io/rank<r>/', 'ONLINE',
  'single_rank_deviation', 'missing_communication', 'stale_output', '4·tol', 'scripts/mp_harness.py']
  .every((k) => utc.includes(k)) && !/add their layer-level fields later/.test(utc),
  'fusion_ut_contract.md: the L3 MegaKernel part (per-rank weights and online reference I/O, multi-rank controls)');
const hbp = promptOf(on, 'handback kf_c2');
ok(before(on, 'handback kf_c2', 'integrate kf_c2') && hbp.includes('fusion_handback.py') && hbp.includes(`--task-dir '${EV}/tasks/kf_c2'`) &&
  hbp.includes(`--out '${EV}/kernel_fusion/handback/kf_c2'`) && !labels(on).includes('handback mk_gemm') &&
  kfOf(on).handbacks.length === 1 && kfOf(on).handbacks[0].ok === true,
  'a region\'s lane result is assembled into the unified hand-back and checked before its A/B; a single kernel has none');

console.log('\n## D. every outcome ends cleanly');
const bhb = run('on_bad_handback');
ok(!labels(bhb).includes('integrate kf_c2') && !names(bhb).includes('kf_c2') && names(bhb).includes('mk_gemm') &&
  kfOf(bhb).handbacks[0].ok === false && labels(bhb).includes('director:validate'),
  'a rejected hand-back never reaches the A/B, and the run goes on');
const pend = run('on_pending');
const fin = promptOf(pend, 'finish-integrate kf_c2');
ok(/"op_kind": ?"fusion"/.test(fin) && /ACCURACY_GATE/.test(fin) && before(pend, 'finish-integrate kf_c2', 'overlay-check kf_c2'),
  'pending: the Finalize-gate replay keeps the region and the accuracy gate, then checks the overlay');
ok(names(pend).includes('kf_c2') && !/ACCURACY_GATE/.test(promptOf(pend, 'integrate mk_gemm')) &&
  /ACCURACY_GATE/.test(promptOf(pend, 'director:validate')),
  'pending: banked at the Finalize-gate; the gate applies from then on, not before');
const bad = run('on_bad_overlay');
ok(!names(bad).includes('kf_c2') && names(bad).includes('mk_gemm') && kfOf(bad).soft_parity === false &&
  !/ACCURACY_GATE/.test(promptOf(bad, 'integrate mk_gemm')) && !/ACCURACY_GATE/.test(promptOf(bad, 'director:validate')),
  'overlay rejected: the region is not banked and the rest of the run is not held to task accuracy');
ok(kfOf(bad).outcomes[0].verdict === 'dead_end' && /^overlay: accepted overlay failed validation/.test(kfOf(bad).outcomes[0].lesson) &&
  !labels(bad).some((l) => /kernel_surgeon|corrective/.test(l)), 'overlay rejected: a final verdict, no corrective re-author');
const unk = run('on_unknown');
ok(!/FUSION REGION/.test(promptOf(unk, 'extract k_x')) && /^Compare candidate backends/.test(laneOf(unk, 'k_x').task) &&
  !/"op_kind": ?"fusion"/.test(promptOf(unk, 'integrate k_x')) && same(kfOf(unk).routed, []),
  'a fusion_id the profile does not have is an ordinary kernel candidate');
const err = run('on_error');
ok(kfOf(err).status === 'error' && /returned nothing/.test(kfOf(err).reason) &&
  labels(err).filter((l) => l === 'fusion_analyst:analyze').length === 2, 'failed profile: two attempts, recorded');
ok(same(bareCalls(err), bareCalls(offk)) && same(bareLogs(err), bareLogs(offk)) && same(bareResult(err), bareResult(offk)),
  'failed profile: the rest of the run is off_kernels\' run, call for call');
const cfg = run('on_config_error');
ok(kfOf(cfg).status === 'error' && /^config: kf_trace_dir does not exist/.test(kfOf(cfg).reason) &&
  !labels(cfg).includes('fusion_analyst:analyze') && same(bareCalls(cfg), bareCalls(offk)) && same(bareResult(cfg), bareResult(offk)),
  'unusable knob: no fusion profile, and otherwise off_kernels\' run');
for (const s of ['on_fast', 'on_deep']) {
  const r = run(s);
  ok(kfOf(r).status === 'skipped' && /runs no Milestone loop/.test(kfOf(r).reason) &&
    !labels(r).includes('fusion_analyst:analyze') && labels(r).includes('director:validate'),
  `${s}: skipped (no Milestone loop to optimize a region in)`);
}
const tk = run('on_topk');
ok(!labels(tk).includes('fusion_analyst:analyze') && promptOf(tk, 'architect:strategize').includes('FUSION_CANDIDATES') &&
  promptOf(tk, 'extract kf_c2').includes('pkg.norm:fused_norm_quant') && kfOf(tk).topk_json === '/data/topk.json' &&
  names(tk).includes('kf_c2'), 'a given topk is reused, not re-analyzed (a bare-string seam is read as the seam)');

console.log('\n## E. AgentX: the region UTs join the prepared inventory');
const ax = run('on_agentx');
ok(before(ax, 'kernel_extractor:prepare baseline', 'fusion_analyst:analyze') &&
  before(ax, 'fusion_analyst:analyze', 'kernel_extractor:prepare kf_c1') &&
  promptOf(ax, 'fusion_analyst:analyze').includes(`${EV}/kernel_extraction/round_baseline`),
  'the analysis reads the baseline preparation\'s captures (KX_EVIDENCE_DIR)');
const pk = promptOf(ax, 'kernel_extractor:prepare kf_c2');
ok(/FUSION_SCOPE: \[\{"task":"kf_c2","fusion_level":"L1"/.test(pk) && pk.includes('fusion_ut_contract.md') &&
  pk.includes('"analysis":{"id":"c2"') &&
  pk.includes('"device_kernels":["rms_norm_kernel","quant_kernel"]'),
  'each recommended region gets one kernel-extract invocation scoped to it');
ok(same(kfOf(ax).unprepared, [{ id: 'c3', reason: 'no seam.target_callable' }]) &&
  !labels(ax).includes('kernel_extractor:prepare kf_c3'), 'a region without a seam gets no UT, and says why');
ok(!labels(ax).some((l) => /^extract /.test(l)) &&
  same(ax.lanes.map((l) => l.kernel_path.split('/').pop()), ['kf_c1', 'kf_c2', 'k_norm']) &&
  /^FUSION REGION c1/.test(laneOf(ax, 'kf_c1').task) && !/FUSION REGION/.test(laneOf(ax, 'k_norm').task),
  'the prepared UTs are reused, best estimate first: kf_c1 2.7%, kf_c2 1.5%, k_norm (Amdahl of 5% at 1.2x) 0.84%');
ok(same(names(ax), ['kf_c1', 'kf_c2', 'k_norm']) && labels(ax).includes('overlay-check kf_c1') &&
  !labels(ax).includes('overlay-check k_norm') && kfOf(ax).accepted.length === 2, 'banked all three; only regions get the overlay check');

const axr = run('on_agentx_route');
ok(!labels(axr).includes('kernel_extractor:prepare kf_c1') && labels(axr).includes('kernel_extractor:prepare kf_c2') &&
  same(kfOf(axr).single_kernel_route, ['c1']) && !kfOf(axr).recommended.includes('c1') && !names(axr).includes('kf_c1'),
  'a region the analysis routes to single-kernel work is never prepared, routed or banked as a region');
ok(same(kfOf(on).hotspots, HOT) && /hotspots: \{ type: 'array'/.test(src) && kfOf(axr).hotspots.length === 2,
  'the analysis\' hotspots (every TopK single kernel\'s gain alone) are kept in the record');
const fa = fs.readFileSync(path.join(WF_DIR, 'knowledge', 'kernel_fusion_skills', 'fusion-analysis', 'SKILL.md'), 'utf8');
ok(/### Step 5 — Fusion against the single-kernel route/.test(fa) && /Recommend only candidates whose route is `fusion`/.test(fa) &&
  /route \("fusion" when est_gain_pct_expected is larger/.test(role),
  'the fusion-analysis skill compares each region with the single-kernel route; the analyst returns the route');
ok(['~2 µs', '7 TB/s', 't_floor', 'D3', 'D4', 'Launch-only regions', 'Coverage', 'gain_basis', '07_kernel_to_e2e.md', 'Every single-kernel hotspot']
  .every((k) => fa.includes(k)) && /"hotspots"/.test(role) && /gain_basis/.test(role) && /co-optimization/.test(role),
  'gains are estimated by the co-optimization method; the analyst returns gain_basis and every TopK hotspot');

console.log('\n## E2. L3 MegaKernel on a whole GPU group; the recommended set held to KF_LEVELS');
ok(/const KF_UNSUPPORTED = String\(A\.kf_unsupported_patterns \|\| 'L4 \(whole-layer MonoKernel\)'\);/.test(src) &&
  /budget: KERNEL_BUDGET, gpu_ids: c\.gpu_id, exp_root: `\$\{EVAL_DIR\}\/kernels\/_exp`, \.\.\.kfLaneGpus\(c\),/.test(src) &&
  count('...(KF_ON && kfLaneGpus(c)),') === 1,
  'only L4 is unsupported by default; a Mega lane gets its group in a HeadKernel slot and in the Milestone loop');
const mg = run('on_agentx_mega');
const pm = promptOf(mg, 'kernel_extractor:prepare kf_c5'), lm = laneOf(mg, 'kf_c5');
ok(pm.includes('- GPU_GROUP: 0,1') && pm.includes('- REQUIRED_GPU_COUNT: 2') && /"mega":true,"required_gpu_count":2,"comm":\["all_reduce"\]/.test(pm) &&
  !/GPU_GROUP|REQUIRED_GPU_COUNT/.test(promptOf(mg, 'kernel_extractor:prepare kf_c2')),
  'Mega: its region UT is prepared on a leased group of required_gpu_count GPUs (GPU_GROUP); other regions keep one');
ok(lm.gpus_per_job === 2 && lm.gpu_ids === '0,1' && /L3 MEGAKERNEL on 2 ranks/.test(lm.task) &&
  lm.task.includes('co-optimization/MEGA_SKILL_LINK.md') && !('gpus_per_job' in laneOf(mg, 'kf_c2')) &&
  !/MEGAKERNEL/.test(laneOf(mg, 'kf_c2').task),
  'Mega: its lane runs on the group (gpu_ids, gpus_per_job) and reads the kf_mega_skill link first');
const mm = run('on_agentx_mega_megamode'), lmm = laneOf(mm, 'kf_c5');
ok(lmm.mode === 'mega' && lmm.region_ut && lmm.region_ut === lmm.kernel_path && lmm.mega_candidate_timeout_s === 7200 &&
  lmm.gpus_per_job === 2 && lmm.gpu_ids === '0,1' && /optimized by mode=mega on the region UT/.test(lmm.task) &&
  !lmm.task.includes('MEGA_SKILL_LINK') && !('mode' in laneOf(mm, 'kf_c2')) && !('mode' in lm),
  'kf_mega_mode: a Mega lane is kernel_lane mode=mega on its region UT with kf_mega_args; other lanes and the default are unchanged');
ok(promptOf(mg, 'handback kf_c5').includes('"required_gpu_count":2') && names(mg).includes('kf_c5'),
  'Mega: the hand-back states the ranks, and the region is A/B\'d and banked like any other');
ok(kfOf(mg).unprepared.some((u) => u.id === 'c6' && u.reason === 'mega_needs_4_gpus') &&
  !labels(mg).includes('kernel_extractor:prepare kf_c6') && !kfOf(mg).recommended.includes('c6'),
  'Mega: a region needing more GPUs than the run has is unprepared (mega_needs_4_gpus), never prepared');
const lv = run('on_agentx_levels');
ok(same(kfOf(lv).recommended, ['c1', 'c2']) && !labels(lv).includes('kernel_extractor:prepare kf_c3') &&
  !kfOf(lv).unprepared.some((u) => u.id === 'c3') &&
  lv.logs.some((l) => l.startsWith('[kf] c3: level L3 is not in kf_implementable_levels (L1,L2)')),
  'a recommended region outside kf_implementable_levels is dropped by the code, not only by the prompt');
const fkm = fs.readFileSync(path.join(WF_DIR, 'knowledge', 'kernel_fusion_skills', 'fusion-kernel', 'SKILL.md'), 'utf8');
const link = path.join(WF_DIR, 'knowledge', 'kernel_fusion_skills', 'co-optimization', 'MEGA_SKILL_LINK.md');
ok(/## L3 MegaKernel/.test(fkm) && /kf_mega_skill/.test(fkm) && /B10/.test(fkm) && /B14/.test(fkm) && /C10/.test(fkm) &&
  /trap 13/.test(fkm) && fs.existsSync(link) && /will supply/.test(fs.readFileSync(link, 'utf8')),
  'fusion-kernel: a Mega chapter that starts from the kf_mega_skill link (a placeholder until the user supplies it)');

console.log('\n## F. phase by phase: a resumed run keeps the fusion profile');
const setup = run('on_setup');
ok(kfOf(setup).status === 'profiled' && labels(setup).filter((l) => !END.test(l)).slice(-1)[0] === 'architect:strategize' &&
  setup.result.state.kernel_fusion.candidates.length === 4, 'setup: profiled and carried in state');
const stateFile = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'kf_state_')), 'state.json');
fs.writeFileSync(stateFile, JSON.stringify(setup.result.state));
const res = run('on_resume', { KF_MOCK_STATE: stateFile });
ok(!labels(res).includes('fusion_analyst:analyze') && /FUSION REGION/.test(promptOf(res, 'extract kf_c2')) &&
  promptOf(res, 'architect:plan m2').includes('FUSION_STRATEGY') && names(res).includes('kf_c2') &&
  kfOf(res).status === 'accepted', 'resume: the carried regions are routed, optimized and banked without a new profile');

console.log('\n## G. HeadKernel: slots of heads and fusion regions, best estimate first; one A/B per slot');
const krIn = (r, label) => {
  const p = promptOf(r, label), i = p.indexOf('- KERNEL_RESULT: ');
  return i < 0 ? {} : JSON.parse(p.slice(i + 17, p.indexOf('\n', i)));
};
const heads = (r) => (r.result && r.result.accepted_heads) || [];
const once = (r, label) => labels(r).filter((l) => l === label).length === 1;
const hd = run('on_head');
ok(before(hd, 'extract kf_c4', 'extract_op h_gemm') && before(hd, 'bakeoff h_gemm', 'integrate h_gemm:kf_c4') &&
  once(hd, 'extract kf_c4') && !labels(hd).includes('integrate kf_c4') &&
  /FUSION REGION/.test(promptOf(hd, 'extract kf_c4')) && promptOf(hd, 'extract kf_c4').includes('pkg.mlp:MLP.forward'),
  'on_head: the region (3%) runs before its head (2.5%) in the head\'s slot, and the Milestone loop never gets it');
const sched = (r) => (kfOf(r).schedule || []).map((sl) => [sl.slot, sl.candidates.map((c) => `${c.short_name}:${c.kind}:${c.est_gain_pct}`)]);
ok(same(sched(hd), [['h_gemm', ['kf_c4:fusion:3', 'h_gemm:head:2.5']], ['kf_c2', ['kf_c2:fusion:1.5']]]) &&
  kfOf(hd).schedule[0].candidates[1].basis === 'hotspot: tuned lib' &&
  hd.logs.some((l) => l.startsWith('[kf] HeadKernel schedule (best estimate first, 2 of 2 slot(s)): h_gemm [kf_c4 3%, h_gemm 2.5%]; kf_c2')),
  'on_head: the schedule (slot, candidates, estimates and their basis) is recorded and logged');
const ok2 = krIn(hd, 'integrate kf_c2:kf_c2');
ok(before(hd, 'integrate h_gemm:triton', 'extract kf_c2') && before(hd, 'handback kf_c2', 'integrate kf_c2:kf_c2') &&
  ok2.short_name === 'kf_c2' && ok2.task_dir === `${EV}/tasks/kf_c2` && ok2.op_kind === 'fusion' && ok2.fusion_id === 'c2' &&
  ok2.pct_gpu_time === 6 && !/REUSE_REF: true/.test(promptOf(hd, 'integrate kf_c2:kf_c2')) &&
  before(hd, 'integrate kf_c2:kf_c2', 'overlay-check kf_c2') && !labels(hd).includes('integrate kf_c2') &&
  heads(hd).some((x) => x.short_name === 'kf_c2' && x.fusion_id === 'c2'),
  'on_head: a region on no head\'s kernel is a slot of its own and reaches the same A/B (its own reference leg)');
ok(/^FUSION REGION c4 \(L2\)/.test(laneOf(hd, 'kf_c4').task) && /^Author\+optimize a triton/.test(laneOf(hd, 'h_gemm').task),
  'on_head: one fused kernel lane beside the head\'s own author lane');
const fk = krIn(hd, 'integrate h_gemm:kf_c4'), ak = krIn(hd, 'integrate h_gemm:aiter');
ok(fk.short_name === 'h_gemm' && fk.task_dir === `${EV}/tasks/kf_c4` && fk.op_kind === 'fusion' && fk.fusion_id === 'c4' &&
  fk.winner_kind === 'patch' && fk.pct_gpu_time === 22 && fk.live_call_seam === 'pkg.mlp:MLP.forward' &&
  /ACCURACY_GATE/.test(promptOf(hd, 'integrate h_gemm:kf_c4')),
  'on_head: the fused candidate\'s A/B carries the region\'s UT, seam, GPU-time share and accuracy gate');
ok(ak.task_dir === `${EV}/tasks/h_gemm` && ak.op_kind === 'gemm' && ak.pct_gpu_time === 20 && !('fusion_id' in ak) &&
  /REUSE_REF: true/.test(promptOf(hd, 'integrate h_gemm:aiter')) && !/ACCURACY_GATE/.test(promptOf(hd, 'integrate h_gemm:aiter')),
  'on_head: the head\'s own candidates are A/B\'d as before, against the reference leg the fused one measured');
ok(before(hd, 'integrate h_gemm:kf_c4', 'overlay-check h_gemm') && before(hd, 'overlay-check h_gemm', 'integrate h_gemm:aiter'),
  'on_head: the fused candidate\'s accepted overlay is checked on disk before the next candidate runs');
const hk = heads(hd)[0] || {};
ok(heads(hd).length === 2 && hk.backend === 'kf_c4' && hk.kind === 'patch' && hk.op_kind === 'fusion' && hk.fusion_id === 'c4' &&
  same(kfOf(hd).routed, ['kf_c4', 'kf_c2']) && kfOf(hd).accepted.some((a) => a.short_name === 'h_gemm' && a.fusion_id === 'c4') &&
  kfOf(hd).outcomes.some((o) => o.short_name === 'h_gemm:kf_c4' && o.verdict === 'candidate_passed'),
  'on_head: the best measured candidate, the fused one, is banked as the head\'s win; the record says where it ran');
const lose = run('on_head_loses');
const lk = heads(lose)[0] || {};
ok(lk.backend === 'aiter' && lk.op_kind === 'gemm' && !('fusion_id' in lk) && !kfOf(lose).accepted.some((a) => a.fusion_id === 'c4') &&
  kfOf(lose).outcomes.some((o) => o.short_name === 'h_gemm:kf_c4' && o.verdict === 'candidate_passed') &&
  once(lose, 'extract kf_c4') && !labels(lose).includes('integrate kf_c4'),
  'on_head_loses: a fused candidate measured below another is not banked, and the region is not tried again');
const unrun = run('on_head_unrun');
const uk = krIn(unrun, 'integrate h_gemm:kf_c4');
ok(!labels(unrun).includes('bakeoff h_gemm') && !labels(unrun).includes('integrate kf_c4') && uk.task_dir === `${EV}/tasks/kf_c4` &&
  uk.op_kind === 'fusion' && (heads(unrun).find((x) => x.short_name === 'h_gemm') || {}).fusion_id === 'c4',
  'on_head_unrun: a failed head extract ends only the head\'s route: its slot A/Bs the region with the region\'s UT');
const nw = run('on_head_nowin');
ok(before(nw, 'bakeoff h_gemm', 'extract kf_c4') && labels(nw).includes('integrate h_gemm:kf_c4') &&
  !labels(nw).some((l) => /^integrate h_gemm:(aiter|triton)/.test(l)) && !labels(nw).includes('integrate kf_c4') &&
  same(sched(nw)[0], ['h_gemm', ['h_gemm:head:5', 'kf_c4:fusion:3']]) &&
  nw.logs.some((l) => l.startsWith('  h_gemm: no win and nothing worth authoring')),
  'on_head_nowin: the head (5%) runs first; its no-win bake-off ends its route, and the slot\'s region still runs and is A/B\'d');
const kh = run('on_head_kfhigh');
ok(same(sched(kh).map((x) => x[0]), ['kf_c2', 'h_gemm']) && before(kh, 'integrate kf_c2:kf_c2', 'extract kf_c4') &&
  before(kh, 'integrate kf_c2:kf_c2', 'extract_op h_gemm'),
  'on_head_kfhigh: a region estimated above every head (9%) runs its slot first');
const ne = run('on_head_noest');
const nh = (kfOf(ne).schedule || [{ candidates: [] }])[0].candidates.find((c) => c.kind === 'head') || {};
ok(Math.abs(nh.est_gain_pct - 3.448) < 1e-3 && /^amdahl\(pct_gpu_time 20, 1\.2x\)$/.test(nh.basis) &&
  before(ne, 'bakeoff h_gemm', 'extract kf_c4'),
  'on_head_noest: a head without an estimate takes the Amdahl ceiling of its share at 1.2x (3.45% > the region\'s 3%)');
const b1 = run('on_head_budget1');
ok(same(sched(b1).map((x) => x[0]), ['h_gemm']) && labels(b1).includes('integrate kf_c2') && !labels(b1).includes('integrate kf_c2:kf_c2'),
  'on_head_budget1: HEAD_BUDGET slots run; the region of a slot not run goes to the Milestone loop');
// Off, HeadKernel is GEAK's own: the same agent calls in the same order as before KernelFusion ordered slots
// (recorded from the base branch's script, BASE_REV), and, where git has that script, the same prompts, logs and result.
const GOLD = {
  off_head: ['extract_op h_gemm', 'bakeoff h_gemm', 'integrate h_gemm:aiter', 'integrate h_gemm:triton', 'profiler:post-head',
    'extract mk_gemm', 'integrate mk_gemm'],
  off_head_nowin: ['extract_op h_gemm', 'bakeoff h_gemm', 'extract mk_gemm', 'integrate mk_gemm'],
  off_head_unrun: ['extract_op h_gemm', 'extract mk_gemm', 'integrate mk_gemm'],
  off_agentx_head: ['bakeoff h_gemm', 'integrate h_gemm:aiter', 'integrate h_gemm:triton', 'integrate k_norm'],
};
const BASE_REV = '26d2cb3';
const gb = spawnSync('git', ['-c', 'safe.directory=*', 'show', `${BASE_REV}:e2e_workflow/e2e_workflow.js`],
  { cwd: WF_DIR, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
const baseScript = gb.status === 0 ? path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'kf_base_')), 'e2e_workflow.js') : '';
if (baseScript) fs.writeFileSync(baseScript, gb.stdout);
for (const [s, want] of Object.entries(GOLD)) {
  const r = run(s);
  const got = labels(r).filter((l) => /^(extract|bakeoff|integrate|profiler:post-head)/.test(l));
  let full = true;
  if (baseScript) {
    const p = spawnSync(process.execPath, [path.join(SCRIPTS, 'kernel_fusion_mockrun.mjs'), s, baseScript],
      { encoding: 'utf8', timeout: 180000, maxBuffer: 256 * 1024 * 1024 });
    const b = JSON.parse(p.stdout.trim().split('\n').pop());
    full = b.ok && same(b.calls, r.calls) && same(b.logs, r.logs) && same(b.result, r.result);
  }
  ok(same(got, want) && full, `${s}: KernelFusion off, HeadKernel runs as before${baseScript ? ` (call for call, log for log, as ${BASE_REV})` : ''}`);
}
const hp = run('on_head_pending');
const pr = krIn(hp, 'finish-integrate h_gemm');
ok(pr.op_kind === 'fusion' && pr.task_dir === `${EV}/tasks/kf_c4` && /ACCURACY_GATE/.test(promptOf(hp, 'finish-integrate h_gemm')) &&
  before(hp, 'finish-integrate h_gemm', 'overlay-check h_gemm') && (heads(hp).find((x) => x.short_name === 'h_gemm') || {}).fusion_id === 'c4',
  'on_head_pending: the Finalize-gate replays the fused A/B as it was, checks the overlay, and banks it');
const fx = run('on_head_fix');
const fixLane = fx.lanes.find((l) => /^CORRECTIVE FIX/.test(String(l.task))) || {};
const fr = krIn(fx, 'integrate h_gemm corrective');
ok(String(fixLane.kernel_path).endsWith('/_exp/kf_c4') && (fixLane.op_spec || {}).op_kind !== 'gemm' &&
  fr.op_kind === 'fusion' && fr.task_dir === `${EV}/tasks/kf_c4` && fr.pct_gpu_time === 22 &&
  /ACCURACY_GATE/.test(promptOf(fx, 'integrate h_gemm corrective')) && before(fx, 'integrate h_gemm corrective', 'overlay-check h_gemm') &&
  (heads(fx)[0] || {}).fusion_id === 'c4' && (heads(fx)[0] || {}).corrective === true,
  'on_head_fix: a fixable live failure of the fused kernel gets the head\'s corrective re-author, on the fused kernel');
const axh = run('on_agentx_head');
ok(!labels(axh).some((l) => /^extract/.test(l)) && labels(axh).includes('kernel_extractor:prepare kf_c4') &&
  axh.lanes.filter((l) => l.kernel_path.endsWith('/kf_c4')).length === 1 &&
  before(axh, 'integrate h_gemm:kf_c4', 'integrate kf_c1:kf_c1') && before(axh, 'integrate kf_c1:kf_c1', 'integrate kf_c2:kf_c2') &&
  before(axh, 'integrate kf_c2:kf_c2', 'integrate k_norm') && !labels(axh).some((l) => /^integrate kf_c\d$/.test(l)) &&
  (heads(axh)[0] || {}).fusion_id === 'c4',
  'AgentX: every slot takes its region\'s prepared UT (nothing is extracted), best estimate first; the Milestone loop gets the single kernel');

console.log(failures === 0 ? '\nPASS: KernelFusion runs inside GEAK\'s own stages; off, it leaves no trace.'
  : `\n${failures} check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);
