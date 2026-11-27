// CPU-only regression for #522. Execute the shipped workflow helpers and integration wrapper.
// Run: node --test e2e_workflow/scripts/test_speedup_review.js
const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const src = fs.readFileSync(path.join(__dirname, '..', 'e2e_workflow.js'), 'utf8');
const core = src.slice(src.indexOf('const CORRECTNESS_REJECT_RX ='), src.indexOf('// ---- DEEP MODE'));
const wrapper = src.slice(src.indexOf('const SPEEDUP_REVIEW_RESULTS ='), src.indexOf('// Lightweight SURGICAL FIX'));
const abDone = src.match(/^const abDone = .*;$/m)[0];
const abLegs = src.match(/^const abLegs = .*;$/m)[0];

const measurement = (name, extra = {}) => ({
  measurement_id: name, candidate_digest: 'candidate-content', reference_digest: 'h0-content',
  workload_digest: 'workload', ref_summary: `/bench/${name}/ref.json`, cand_summary: `/bench/${name}/cand.json`,
  ref_replicas: 2, cand_replicas: 2, fresh_pair: true, configs_matched: true, workload_matched: true, ...extra,
});
const candidate = (extra = {}) => ({
  short_name: 'blockscale', gate: 'accepted', ab_complete: true, output_parity: 'pass', parity_kind: 'accuracy',
  pct_gpu_time: 7.18, isolated_speedup: 1.4596, ref_med: 335.171, cand_med: 361.051,
  e2e_throughput_tok_s: 361.051, e2e_delta_pct: 7.72, accepted_overlay: '/overlay/h1',
  measurement_evidence: measurement('original'), ...extra,
});
const inputs = {
  KERNEL_RESULT: { pct_gpu_time: 7.18, verified_isolated_speedup: 1.4596, code_patch: '/h1.diff' },
  CURRENT_OVERLAY: '/overlay/h0', CURRENT_FLAGS: '--tp 8', CURRENT_ENV: '', WORKLOAD: { isl: 8192, osl: 1024, conc: 4 },
  REUSE_REF: true, SHARED_REF_MED: 335.171, RESUME_AB: true,
};
const coverage = (rows = [7.18, 3.47, 1.37]) => ({
  candidate_digest: 'candidate-content',
  profile_path: '/profile/h0.json', reference_digest: 'h0-content', workload_digest: 'workload',
  kernels: rows.map((pct, i) => ({ id: `path-${i}`, pct_gpu_time: pct })),
});

function harness(original = candidate(), fresh = candidate({ measurement_evidence: measurement('confirmation') }), closed = false,
  short = () => false, clock = {}) {
  const calls = [], checkpoints = [], windows = [];
  const safeAgent = async (request, opts, tries, timeoutFor) => {
    calls.push({ ...request, opts, tries, timeoutMs: timeoutFor ? timeoutFor() : null });
    return structuredClone(request.inputs.FORCE_FRESH_AB ? fresh : original);
  };
  const abWindowShort = (legs, what, passes = 1) => {
    windows.push({ legs, what, passes });
    return short(legs, what, passes);
  };
  const api = new Function('safeAgent', 'persistE2EValidationCheckpoint', 'searchClosed', 'abWindowShort', `
    const A = {}, ACCURACY_GATE = 'gsm8k', FIXABLE_REJECT_RX = /no_engagement|signature_mismatch/;
    const SEARCH_REPLICAS = 1, ACCURACY_INPUTS = {}, MEASUREMENT_MODE = 'warm_server';
    const EVAL_DIR = '/eval', WORKLOAD = {}, INTEGRATE_SCHEMA = {}, profile = {profile_topN_json:'/profile/h0.json'};
    const AB_FINISH_RETRIES = 3, log = () => {}, tuningIntegrateInputs = () => ({}), KF_ON = false;
    const roleAgent = (role, phase, intro, inputs) => ({role, phase, intro, inputs});
    const adoptAbandonedVerdict = async r => r;
    const IS_AGENTX = ${!!clock.agentx}, AGENT_TIMEOUT_MS = ${clock.agentTimeoutMs || 7200000};
    const TIME_BUDGET_MS = ${clock.remainingMs == null ? 'null' : 1}, FINAL_PHASE_STARTED = false;
    const FINAL_RESERVE_MS = ${clock.reserveMs || 0}, remainingMs = () => ${clock.remainingMs || 0};
    const agentTimeoutFor = () => TIME_BUDGET_MS == null ? AGENT_TIMEOUT_MS
      : Math.max(120000, Math.min(AGENT_TIMEOUT_MS, remainingMs() - FINAL_RESERVE_MS));
    const agentxLegS = (w) => w + 1200, AGENTX_AB_OVERHEAD_S = 600;
    const agentxAbMs = (legs, passes = 1) => 1000 * (legs * passes * agentxLegS(900) + AGENTX_AB_OVERHEAD_S);
    ${core}\n${abDone}\n${abLegs}\n${wrapper}
    return { speedupEstimate, speedupNeedsReview, confirmedSpeedup, integAccepted, gateRejectReason,
      rejectClass, reviewSpeedup, runIntegrateBothLegs };
  `)(safeAgent, async (p, value) => checkpoints.push({ path: p, ...value }), () => closed, abWindowShort);
  return { ...api, calls, checkpoints, windows };
}

test('workflow remains syntactically valid in its async workflow wrapper', () => {
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  assert.doesNotThrow(() => new AsyncFunction(src.replace('export const meta', 'const meta')));
});

test('#522 whole-module coverage includes the second shape and prefill without changing the head', async () => {
  const c = candidate({ affected_gpu_time: coverage() });
  const h = harness(c);
  const estimate = h.speedupEstimate(7.18, 1.4596, c);
  assert.ok(Math.abs(estimate.pct_gpu_time - 12.02) < 1e-9);
  assert.ok(Math.abs(estimate.estimate_pct - 3.9337544649) < 1e-8);
  assert.equal(estimate.source, 'coverage_uniform_estimate');
  assert.equal(c.pct_gpu_time, 7.18);
  const result = await h.runIntegrateBothLegs('test', inputs, 'h1', 'HeadKernel');
  assert.equal(h.integAccepted(result, 7.18, 1.4596), true);
  assert.equal(h.calls.length, 1);
  assert.equal(h.checkpoints[0].committed, true);
  assert.equal(h.calls[0].inputs.PROFILE_TOPN_JSON, '/profile/h0.json');
});

test('#524 feedback: profile and A/B of the same workload share one workload identity digest', () => {
  // Qwen3.5 run: the role used to ask for a profile-manifest digest (with profile/trace file hashes)
  // in coverage and a bench-protocol digest in the measurement, so coverage could never match.
  const role = fs.readFileSync(path.join(__dirname, '..', 'roles', 'e2e_integrator.md'), 'utf8');
  assert.doesNotMatch(role, /profile workload manifest/);
  assert.match(role, /reuse the SAME\s+value in `measurement_evidence` and `affected_gpu_time`/);
  const sha = (o) => require('node:crypto').createHash('sha256').update(JSON.stringify(o)).digest('hex');
  const identity = { dataset: 'random', seed: 0, isl: 8192, osl: 1024, random_range_ratio: 0.8, conc: 4 };
  const workload = sha(identity);
  const evidence = measurement('original', { workload_digest: workload, protocol_digest: sha({
    bench_client: 'inferencex', measurement_mode: 'warm_server', num_prompts: 40, num_warmups: 8,
    metric_basis: 'aggregate_output_tok_s' }) });
  const c = candidate({ pct_gpu_time: 6.49, isolated_speedup: 1.7746, e2e_delta_pct: 6.2,
    measurement_evidence: evidence, affected_gpu_time: { ...coverage([6.49, 2.84, 2.84]), workload_digest: workload } });
  const h = harness();
  const estimate = h.speedupEstimate(6.49, 1.7746, c);
  assert.equal(estimate.source, 'coverage_uniform_estimate');
  assert.ok(Math.abs(estimate.pct_gpu_time - 12.17) < 1e-9);
  assert.ok(Math.abs(estimate.estimate_pct - 5.61) < 0.01);
  assert.equal(h.speedupNeedsReview(6.49, 1.7746, c), false, '+6.2% is within twice the coverage estimate');
  const headOnly = { ...c, affected_gpu_time: undefined };
  assert.equal(h.speedupNeedsReview(6.49, 1.7746, headOnly), true, 'the head-only estimate would review');
});

test('per-path estimate includes a removed reduce and uses measured speedups, including a slowdown', () => {
  const h = harness();
  const rows = [
    { id: 'decode-main', pct_gpu_time: 4.53, isolated_speedup: 1.28 },
    { id: 'reduce', pct_gpu_time: 2.65, removed: true },
    { id: 'other-shape', pct_gpu_time: 3.47, isolated_speedup: 0.9 },
    { id: 'prefill', pct_gpu_time: 1.37, isolated_speedup: 1.94 },
  ];
  const c = candidate({ affected_gpu_time: { ...coverage(), kernels: rows } });
  const saved = .0453 * (1 - 1 / 1.28) + .0265 + .0347 * (1 - 1 / .9) + .0137 * (1 - 1 / 1.94);
  const estimate = h.speedupEstimate(7.18, 1.4596, c);
  assert.equal(estimate.source, 'coverage_per_path');
  assert.ok(Math.abs(estimate.estimate_pct - (1 / (1 - saved) - 1) * 100) < 1e-9);
  c.affected_gpu_time.kernels.push({ ...rows[1] });
  assert.deepEqual(h.speedupEstimate(7.18, 1.4596, c), estimate, 'duplicate reduction is counted once');
});

test('stale, conflicting or invalid coverage falls back to an explicitly labelled head estimate', () => {
  const h = harness();
  for (const cov of [
    { ...coverage(), candidate_digest: 'another-overlay' },
    { ...coverage(), reference_digest: 'pre-h0-content' },
    { ...coverage(), workload_digest: 'different-workload' },
    coverage([60, 60]), coverage([-1]), coverage([NaN]),
    { ...coverage(), kernels: [{ id: 'same', pct_gpu_time: 7 }, { id: 'same', pct_gpu_time: 8 }] },
  ]) {
    assert.equal(h.speedupEstimate(7.18, 1.4596, candidate({ affected_gpu_time: cov })).source, 'head_estimate');
  }
});

test('a confirmed +7.72% win is banked despite the 2.31% head estimate; all sites use the wrapper', async () => {
  const h = harness();
  const result = await h.runIntegrateBothLegs('test', inputs, 'h1', 'HeadKernel');
  assert.equal(result.speedup_review.status, 'confirmed');
  assert.equal(h.integAccepted(result, 7.18, 1.4596), true);
  assert.equal(h.calls.length, 2);
  const review = h.calls[1];
  assert.equal(review.tries, 1);
  assert.equal(review.inputs.FORCE_FRESH_AB, true);
  assert.equal(review.inputs.RESUME_AB, false);
  assert.equal(review.inputs.REUSE_REF, false);
  assert.equal('SHARED_REF_MED' in review.inputs, false);
  assert.equal(review.inputs.REPLICAS, 2);
  assert.equal(review.inputs.CAND_OVERLAY_DIR, '/overlay/h1');
  assert.equal(h.checkpoints[0].committed, true);
  assert.deepEqual(h.checkpoints[0].integration_result, result);
  assert.match(h.checkpoints[0].path, /candidate_blockscale_confirmation\//);
  // Deep mode used to call the agent directly, bypassing the common review/checkpoint path.
  assert.match(src, /deepInputs, `integrate \$\{c\.uid\} g\$\{e2eGateCount\}`, 'HeadKernel'/);
  assert.doesNotMatch(src, /isImplausibleSpeedup|implausible2/);
});

test('equal performance numbers from independent artifacts confirm; repeating the old A/B never does', async () => {
  const h = harness(candidate(), candidate());
  const first = await h.runIntegrateBothLegs('test', inputs, 'h1', 'HeadKernel');
  const second = await h.runIntegrateBothLegs('test', inputs, 'h1 finish', 'Finalize-gate');
  assert.equal(first.speedup_review.status, 'inconclusive');
  assert.deepEqual(second, first);
  assert.equal(h.calls.filter(c => c.inputs.FORCE_FRESH_AB).length, 1);
  assert.equal(h.integAccepted(first, 7.18, 1.4596), false);
  assert.equal(h.rejectClass(h.gateRejectReason(first, 7.18, 1.4596)), '');
  assert.equal(h.checkpoints[0].committed, false);
  assert.equal(h.checkpoints[0].validation_status, 'speedup_review_pending');
});

test('changed candidate/baseline/workload, reused legs and insufficient samples cannot confirm', async () => {
  for (const extra of [
    { candidate_digest: 'new-patch' }, { reference_digest: 'new-stack' }, { workload_digest: 'other' },
    { protocol_digest: 'different-bench-protocol' }, { ref_summary: '/bench/original/ref.json' }, { cand_summary: '/bench/original/cand.json' },
    { fresh_pair: false }, { configs_matched: false }, { workload_matched: false },
    { ref_replicas: 1 }, { cand_replicas: 1 }, { ref_replicas: '2' }, { cand_replicas: 2.5 },
  ]) {
    const h = harness(candidate(), candidate({ measurement_evidence: measurement('confirmation', extra) }));
    const result = await h.runIntegrateBothLegs('test', inputs, 'h1', 'HeadKernel');
    assert.equal(result.speedup_review.status, 'inconclusive', JSON.stringify(extra));
    assert.equal(h.integAccepted(result, 7.18, 1.4596), false);
  }
});

test('confirmed evidence already on disk avoids another review', async () => {
  const c = candidate({ measurement_evidence: measurement('confirmation'),
    speedup_review: { status: 'confirmed', original_measurement: measurement('original') } });
  const h = harness(c);
  assert.equal(h.integAccepted(await h.runIntegrateBothLegs('test', inputs, 'h1', 'HeadKernel'), 7.18, 1.4596), true);
  assert.equal(h.calls.length, 1);
});

test("an integrator's own 'pending' flag does not block a gain within the coverage estimate", async () => {
  const c = candidate({ pct_gpu_time: 13.6954, isolated_speedup: 1.5265, e2e_delta_pct: 12.79,
    affected_gpu_time: coverage([13.6954, 3.2377, 2.2092]),
    speedup_review: { status: 'pending', estimate: { amdahl_e2e_ceiling_pct: 4.96 } } });
  const kernel = { ...inputs, KERNEL_RESULT: { pct_gpu_time: 13.6954, verified_isolated_speedup: 1.5265 } };
  const h = harness(c);
  assert.ok(Math.abs(h.speedupEstimate(13.6954, 1.5265, c).estimate_pct - 7.069) < 0.001);
  assert.equal(h.speedupNeedsReview(13.6954, 1.5265, c), false, '+12.79% is within twice the coverage estimate');
  const result = await h.runIntegrateBothLegs('test', kernel, 'h1', 'HeadKernel');
  assert.equal('speedup_review' in result, false);
  assert.equal(h.integAccepted(result, 13.6954, 1.5265), true);
  assert.equal(h.calls.length, 1);
  assert.equal(h.checkpoints[0].gate, 'accepted');
  assert.equal(h.checkpoints[0].committed, true);
  assert.equal(h.checkpoints[0].validation_status, 'accepted_intermediate');
});

test("an integrator's own 'pending' flag still gets the confirmation the estimate requires", async () => {
  const h = harness(candidate({ speedup_review: { status: 'pending' } }));
  const result = await h.runIntegrateBothLegs('test', inputs, 'h1', 'HeadKernel');
  assert.equal(h.calls.length, 2);
  assert.equal(h.calls[1].inputs.FORCE_FRESH_AB, true);
  assert.equal(result.speedup_review.status, 'confirmed');
  assert.equal(h.integAccepted(result, 7.18, 1.4596), true);
  assert.equal(h.checkpoints[0].committed, true);
});

test('a resumed inconclusive review is not re-run and backend candidates keep distinct checkpoints', async () => {
  const pending = candidate({ speedup_review: { status: 'inconclusive', original_measurement: measurement('original') } });
  const resumed = harness(pending);
  const result = await resumed.runIntegrateBothLegs('resume', inputs, 'h1', 'Finalize-gate');
  assert.equal(result.speedup_review.status, 'inconclusive');
  assert.equal(resumed.calls.length, 1);
  const original = candidate(), fresh = candidate({ measurement_evidence: measurement('confirmation') });
  const h = harness(original, fresh);
  await h.runIntegrateBothLegs('test', inputs, 'h1:triton', 'HeadKernel');
  original.measurement_evidence = measurement('hip-original', { candidate_digest: 'hip' });
  fresh.measurement_evidence = measurement('hip-confirmation', { candidate_digest: 'hip' });
  await h.runIntegrateBothLegs('test', inputs, 'h1:hip', 'HeadKernel');
  assert.notEqual(h.checkpoints[0].path, h.checkpoints[1].path);
  assert.equal(h.checkpoints[0].integration_result.speedup_review.original_measurement.candidate_digest, 'candidate-content');
  assert.equal(h.checkpoints[1].integration_result.speedup_review.original_measurement.candidate_digest, 'hip');
});

test('real accuracy failures and measured throughput regressions remain rejected', async () => {
  for (const fail of [
    { gate: 'rejected', output_parity: 'fail', reason: 'accuracy_regression' },
    { gate: 'rejected', e2e_delta_pct: -1, reason: 'throughput_regression' },
  ]) {
    const h = harness(candidate(), candidate({ ...fail, measurement_evidence: measurement('confirmation') }));
    const result = await h.runIntegrateBothLegs('test', inputs, 'h1', 'HeadKernel');
    assert.equal(result.gate, 'rejected');
    assert.equal(h.integAccepted(result, 7.18, 1.4596), false);
    assert.equal(h.checkpoints[0].committed, false);
    assert.equal(h.rejectClass(h.gateRejectReason(result, 7.18, 1.4596)), fail.output_parity === 'fail' ? 'correctness' : '');
  }
});

test('timeouts, missing provenance and the final reserve stop review without a corrective loop', async () => {
  for (const [original, fresh, closed] of [
    [candidate(), null, false], [candidate(), candidate({ gate: 'incomplete', ab_complete: false }), false],
    [candidate(), candidate({ ab_complete: undefined, measurement_evidence: measurement('confirmation') }), false],
    [candidate(), candidate({ gate: 'stack', measurement_evidence: measurement('confirmation') }), false],
    [candidate({ measurement_evidence: undefined }), candidate(), false], [candidate(), candidate(), true],
  ]) {
    const h = harness(original, fresh, closed);
    const result = await h.runIntegrateBothLegs('test', inputs, 'h1', 'HeadKernel');
    assert.equal(result.speedup_review.status, 'inconclusive');
    assert.equal(h.rejectClass(h.gateRejectReason(result, 7.18, 1.4596)), '');
    assert.ok(h.calls.length <= 2);
    if (!original.measurement_evidence || closed) assert.equal(h.calls.length, 1);
  }
});

test('a confirmation that cannot finish before the final reserve is deferred, not started', async () => {
  const h = harness(candidate(), candidate({ measurement_evidence: measurement('confirmation') }), false,
    (legs, what) => what.endsWith(': speedup review'));
  const result = await h.runIntegrateBothLegs('test', inputs, 'h1', 'HeadKernel');
  assert.deepEqual(h.windows.map(({ legs, passes }) => [legs, passes]), [[1, 1], [2, 2]],
    'the original A/B is checked for its one fresh leg, the confirmation for two fresh legs of two passes');
  assert.equal(h.calls.length, 1);
  assert.equal(result.speedup_review.status, 'inconclusive');
  assert.match(result.reason, /^speedup_review_inconclusive: deferred/);
  assert.equal(h.integAccepted(result, 7.18, 1.4596), false);
  assert.equal(h.rejectClass(h.gateRejectReason(result, 7.18, 1.4596)), '');
  assert.equal(h.checkpoints[0].committed, false);
  assert.equal(h.checkpoints[0].validation_status, 'speedup_review_pending');
});

test('a claimed confirmation still requires an explicit completed correctness gate', () => {
  const h = harness();
  for (const extra of [{ output_parity: undefined }, { parity_kind: 'none' }, { ab_complete: undefined }]) {
    const c = candidate({ measurement_evidence: measurement('confirmation'), ...extra,
      speedup_review: { status: 'confirmed', original_measurement: measurement('original') } });
    assert.equal(h.confirmedSpeedup(c), false);
  }
});

test('ordinary and byte-exact wins retain their acceptance behavior', async () => {
  for (const c of [candidate({ e2e_delta_pct: 2 }), candidate({ parity_kind: 'byte_exact' })]) {
    const h = harness(c);
    const result = await h.runIntegrateBothLegs('test', inputs, 'h1', 'HeadKernel');
    assert.equal(h.integAccepted(result, 7.18, 1.4596), true);
    assert.equal(h.calls.length, 1);
  }
  const h = harness();
  assert.equal(h.rejectClass('implausible_speedup: likely corruption'), '');
  assert.equal(h.rejectClass('speedup_review_inconclusive: configuration mismatch'), '');
  assert.equal(h.rejectClass('signature_mismatch'), 'integration');
  assert.equal(h.rejectClass('accuracy_regression'), 'correctness');
});

test('a trace-replay review gets the time its two-replica A/B needs, not the 2h per-agent cap', async () => {
  // GLM-5.2 c40 run (2026-11-26): the review of a +7.33% Mega candidate needed ~2.5h (2 legs x 2 replicas of a
  // 900s replay + warmups), was abandoned at the 2h cap, and its valid confirmation arrived after the slot chose.
  const needMs = 1000 * (2 * 2 * (900 + 1200) + 600);
  const big = candidate({ e2e_delta_pct: 30, cand_med: 435.7, e2e_throughput_tok_s: 435.7 });
  const confirm = { ...big, measurement_evidence: measurement('confirmation') };
  const h = harness(big, confirm, false, () => false, { agentx: true, remainingMs: 20 * 3600000, reserveMs: 3 * 3600000 });
  await h.reviewSpeedup(big, inputs, 'h1', 'HeadKernel');
  const review = h.calls.find((c) => c.inputs.FORCE_FRESH_AB);
  assert.ok(review, 'the review ran');
  assert.ok(review.timeoutMs >= needMs, `review timeout ${review.timeoutMs} < ${needMs}`);
  assert.ok(review.timeoutMs > 7200000);
  // Never past the search window before the final reserve.
  const tight = harness(big, confirm, false, () => false, { agentx: true, remainingMs: 4 * 3600000, reserveMs: 2 * 3600000 });
  await tight.reviewSpeedup(big, inputs, 'h1', 'HeadKernel');
  assert.equal(tight.calls.find((c) => c.inputs.FORCE_FRESH_AB).timeoutMs, 2 * 3600000);
  // A fixed-sequence run keeps the ordinary cap.
  const plain = harness(big, confirm, false, () => false, { agentx: false });
  await plain.reviewSpeedup(big, inputs, 'h1', 'HeadKernel');
  assert.equal(plain.calls.find((c) => c.inputs.FORCE_FRESH_AB).timeoutMs, 7200000);
});

test('a checkpoint is named by a real measurement digest, never by placeholder text', async () => {
  const placeholder = 'computed as sha256(ref_summary path+content, cand_summary path+content) - see integrate_result.json';
  // byte-exact parity: no review, so the checkpoint is named by this A/B's own (placeholder) id
  const h = harness(candidate({ parity_kind: 'byte_exact', measurement_evidence: measurement(placeholder) }));
  await h.runIntegrateBothLegs('test', { ...inputs, CAND_TAG: 'c1_kf_region' }, 'h1', 'HeadKernel');
  assert.match(h.checkpoints[0].path, /overlay\/candidate_blockscale_c1_kf_region\/e2e_validation\.json$/);
  const hex = 'a'.repeat(64);
  const g = harness(candidate({ parity_kind: 'byte_exact', measurement_evidence: measurement(hex) }));
  await g.runIntegrateBothLegs('test', inputs, 'h1', 'HeadKernel');
  assert.match(g.checkpoints[0].path, new RegExp(`candidate_blockscale_${hex}/`));
});

test("the analysis' serving-layer upper bound raises the review threshold; without it the device-time estimate stands", async () => {
  // +7.72% on a 7.18% / 1.46x head: the device-time estimate (2.31%) asks for a review; an analysis upper bound of 9%
  // (tail metrics, queueing already included) does not, and a bound below the device estimate never lowers it.
  const c = candidate();
  const h = harness(c);
  assert.equal(h.speedupNeedsReview(7.18, 1.4596, c), true);
  assert.equal(h.speedupNeedsReview(7.18, 1.4596, c, 9), false);
  assert.equal(h.speedupNeedsReview(7.18, 1.4596, c, 1), true);
  assert.equal(h.integAccepted(c, 7.18, 1.4596, 9), true);
  assert.equal(h.integAccepted(c, 7.18, 1.4596), false);
  // reviewSpeedup reads it from KERNEL_RESULT.est_gain_pct_upper: no confirmation A/B is started.
  const g = harness(c);
  const res = await g.reviewSpeedup(c, { ...inputs, KERNEL_RESULT: { ...inputs.KERNEL_RESULT, est_gain_pct_upper: 9 } },
    'h1', 'HeadKernel');
  assert.equal(g.calls.length, 0);
  assert.equal(res.gate, 'accepted');
});
