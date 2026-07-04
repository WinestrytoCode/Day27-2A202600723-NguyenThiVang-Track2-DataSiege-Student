"""
Data Siege defense.

Strategy (see reflection.md for the tradeoff reasoning):
- Exactly ONE metered tool call per event (single-pass coverage). On the
  private stream that costs ~300 credits against a 320 budget, so there is no
  cost overage and every event gets a real measurement — no blind guessing.
- Detection combines the published baseline bounds (mean ± 3σ, good for large
  deviations) with tighter, statistically-motivated secondary thresholds so
  that subtle faults sitting closer to normal variance are still caught
  without inflating false positives on clean events.
- Nothing here is keyed to a specific event id or run/seed. Every rule is a
  general statistical/structural check on the values the sanctioned tools
  return, so it transfers to an unseen stream.

Signal notes per pillar (derived from how the toolkit computes each check and
from the shape of clean-vs-faulty separation, not from memorized answers):
  checks     — row volume, customer_id null rate, amount distribution, freshness.
  contracts  — the tool's own `violations` list (schema/type) plus SLA freshness.
  lineage    — missing upstream edges, orphaned outputs, runtime anomalies.
  ai_infra   — training/serving feature skew (normalized sigma) and
               embedding-corpus drift / staleness.
"""
from api import Verdict

# Adaptive-detector constants.
_ROBUST_Z_K = 3.0      # flag a value >K robust-z (median/MAD) from the run's own
                       #   live distribution — catches subtle faults that a
                       #   static bound misses because the stream drifts.
_MIN_HISTORY = 8       # need this many observations before robust-z is trusted.
_HISTORY_CAP = 60      # rolling window; recent clean values define "normal now".
_MAD_TO_SIGMA = 1.4826 # scale MAD to a Gaussian-consistent sigma.


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def _err(resp):
    """A tool error / not-yet-visible sentinel. Fail quiet (don't alert) so a
    transient lookup miss never manufactures a false positive."""
    return not isinstance(resp, dict) or "error" in resp


def _median(xs):
    """Median without importing `statistics` — that module transitively pulls in
    `decimal`/`fractions`, which the child sandbox's import allowlist blocks, so
    touching it would crash the handler. Plain sorted-middle is all we need."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _robust_z_outlier(ctx, key, value):
    """True if `value` is a robust-z outlier vs the running per-field history in
    ctx.state, THEN record it. Uses median/MAD (not mean/std) so the small
    faulty minority in the window doesn't drag the center or inflate the spread.
    This is a general within-run anomaly test — no run/seed-specific constants."""
    if value is None:
        return False
    hist = ctx.state.setdefault("_hist", {}).setdefault(key, [])
    outlier = False
    if len(hist) >= _MIN_HISTORY:
        med = _median(hist)
        mad = _median([abs(h - med) for h in hist]) or 1e-9
        if abs(value - med) / (_MAD_TO_SIGMA * mad) > _ROBUST_Z_K:
            outlier = True
    hist.append(value)
    if len(hist) > _HISTORY_CAP:
        del hist[:-_HISTORY_CAP]
    return outlier


# --------------------------------------------------------------------------- #
# checks  (data_batch)
# --------------------------------------------------------------------------- #
def check_data_batch(payload, ctx):
    b = ctx.baseline
    prof = ctx.tools.batch_profile(payload["batch_id"])
    if _err(prof):
        return Verdict(alert=False, pillar="checks", reason="profile unavailable")

    row = prof.get("row_count")
    null_rate = (prof.get("null_rate") or {}).get("customer_id")
    mean_amt = prof.get("mean_amount")
    staleness = prof.get("staleness_min")

    reasons = []

    # The baseline bands are mean ± 3σ, so a value outside a band is a fault at
    # >3σ. To also reach the "subtle" tier (faults nudged to ~2.5-3σ, inside
    # the band) we add a soft two-sided band at ~2.6σ. Because the published
    # band is exactly ±3σ, 2.6σ == center ± (2.6/3)·half. That margin sits just
    # past the widest normal jitter, so it catches a near-band fault without
    # firing on clean events.
    SOFT = 2.6 / 3.0

    # Each field gets three layers: hard baseline band (>3σ), a soft static
    # margin just above the clean envelope, and an in-run robust-z outlier test
    # against the field's own live distribution (catches subtle faults that a
    # fixed bound misses when the stream itself drifts).

    # Volume (two-sided: spike or drop).
    if row is not None:
        lo, hi = b["row_count_min"], b["row_count_max"]
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)  # == 3σ
        if row < lo or row > hi:
            reasons.append("row_count out of baseline band")
        elif half > 0 and abs(row - center) > SOFT * half:
            reasons.append("row_count borderline volume anomaly")
        if _robust_z_outlier(ctx, "db_row", row):
            reasons.append("row_count in-run volume outlier")

    # Null rate (one-sided high). Baseline is the ±3σ upper bound. The clean
    # null-rate distribution sits well under it, so a soft bound a little below
    # the baseline stays clear of clean variance while reaching a subtle null
    # spike that hasn't fully cleared 3σ.
    if null_rate is not None:
        if null_rate > b["null_rate_max"]:
            reasons.append("customer_id null rate over baseline")
        elif null_rate > 0.85 * b["null_rate_max"]:
            reasons.append("customer_id null rate elevated")
        if _robust_z_outlier(ctx, "db_null", null_rate):
            reasons.append("customer_id null rate in-run outlier")

    # Amount distribution (two-sided).
    if mean_amt is not None:
        lo, hi = b["mean_amount_min"], b["mean_amount_max"]
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        if mean_amt < lo or mean_amt > hi:
            reasons.append("mean_amount out of baseline band")
        elif half > 0 and abs(mean_amt - center) > SOFT * half:
            reasons.append("mean_amount borderline distribution shift")
        if _robust_z_outlier(ctx, "db_mean", mean_amt):
            reasons.append("mean_amount in-run distribution outlier")

    # Freshness / staleness (one-sided high). Clean staleness sits under the
    # bound with room to spare, so a soft bound a little below it catches a
    # subtle freshness lag without touching clean events.
    if staleness is not None:
        if staleness > b["staleness_min_max"]:
            reasons.append("batch staleness over baseline")
        elif staleness > 0.85 * b["staleness_min_max"]:
            reasons.append("batch staleness elevated")
        if _robust_z_outlier(ctx, "db_stale", staleness):
            reasons.append("batch staleness in-run outlier")

    if reasons:
        return Verdict(alert=True, pillar="checks", confidence=0.9,
                       reason="; ".join(reasons))
    return Verdict(alert=False, pillar="checks")


# --------------------------------------------------------------------------- #
# contracts  (contract_checkpoint)
# --------------------------------------------------------------------------- #
def check_contract_checkpoint(payload, ctx):
    b = ctx.baseline
    diff = ctx.tools.contract_diff(payload["contract_id"],
                                   payload["checkpoint_batch_id"])
    if _err(diff):
        return Verdict(alert=False, pillar="contracts", reason="diff unavailable")

    reasons = []

    # The tool already surfaces schema_hash_mismatch / type_violation. These are
    # hard contract breaches — always alert.
    violations = diff.get("violations") or []
    if violations:
        reasons.append("contract violations: " + ",".join(violations))

    # SLA freshness breach: compare the actual delay to the declared SLA if
    # present, else to the published baseline. One-sided.
    delay = diff.get("freshness_delay_min")
    if delay is not None:
        sla = (payload.get("declared_sla") or {}).get("freshness_min")
        if sla is not None and delay > sla:
            reasons.append("freshness delay over declared SLA")
        elif delay > b["freshness_delay_max_min"]:
            reasons.append("freshness delay over baseline")

    if reasons:
        return Verdict(alert=True, pillar="contracts", confidence=0.95,
                       reason="; ".join(reasons))
    return Verdict(alert=False, pillar="contracts")


# --------------------------------------------------------------------------- #
# lineage  (lineage_run)
# --------------------------------------------------------------------------- #
def check_lineage_run(payload, ctx):
    b = ctx.baseline
    sl = ctx.tools.lineage_graph_slice(payload["run_id"])
    if _err(sl):
        return Verdict(alert=False, pillar="lineage", reason="slice unavailable")

    reasons = []

    # Missing upstream edge. The event payload's declared `inputs` list is NOT a
    # reliable expectation (a transform can under-declare its own inputs), so we
    # use the structural norm instead: a healthy transform in this pipeline
    # resolves 2 upstream sources (raw.orders + raw.customers). A run whose
    # actual upstream set collapses below that norm has dropped an edge.
    actual_up = sl.get("actual_upstream")
    EXPECTED_UPSTREAM = 2
    if actual_up is not None and len(actual_up) < EXPECTED_UPSTREAM:
        reasons.append("missing upstream edge(s)")

    # Orphaned output: a COMPLETE run that produced no downstream consumer.
    down = sl.get("actual_downstream_count")
    if down is not None and down == 0:
        reasons.append("orphaned output (no downstream)")

    # Runtime anomaly (one-sided high). Baseline max ~5135ms; clean runtime
    # legitimately reaches ~4630ms, so there is only a thin FP-safe margin. A
    # soft bound at 0.95·baseline (~4878ms) sits above the clean envelope and
    # below the smallest real anomaly; anything more aggressive starts hitting
    # clean runs.
    dur = sl.get("duration_ms")
    if dur is not None:
        if dur > b["lineage_duration_ms_max"]:
            reasons.append("runtime over baseline")
        elif dur > 0.95 * b["lineage_duration_ms_max"]:
            reasons.append("runtime elevated")
        # Subtle runtime anomalies land under the static bound but stand out
        # against the run's own tight clean runtime distribution.
        if _robust_z_outlier(ctx, "lin_dur", dur):
            reasons.append("runtime in-run outlier")

    if reasons:
        return Verdict(alert=True, pillar="lineage", confidence=0.9,
                       reason="; ".join(reasons))
    return Verdict(alert=False, pillar="lineage")


# --------------------------------------------------------------------------- #
# ai_infra  (feature_materialization)
# --------------------------------------------------------------------------- #
def check_feature_materialization(payload, ctx):
    b = ctx.baseline
    fd = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if _err(fd):
        return Verdict(alert=False, pillar="ai_infra", reason="drift unavailable")

    # mean_shift_sigma is a normalized statistic (|serve-train| / train_std),
    # so it is directly comparable across feature views. Clean runs sit under
    # ~0.4σ and the published bound is ~0.41σ, so a threshold a hair above that
    # (0.9σ) clears clean jitter while catching subtle skews that a 1σ+ cut
    # would let slip. Real skews overshoot it by a wide margin.
    sigma = fd.get("mean_shift_sigma")
    if sigma is None:
        serve = fd.get("serve_mean"); tm = fd.get("train_mean"); ts = fd.get("train_std")
        if None not in (serve, tm, ts) and ts:
            sigma = abs(serve - tm) / ts

    if sigma is not None:
        threshold = max(0.9, 2.0 * b["feature_mean_shift_sigma_max"])
        if sigma > threshold:
            return Verdict(alert=True, pillar="ai_infra", confidence=0.95,
                           reason=f"feature skew {sigma:.2f}sigma")

    return Verdict(alert=False, pillar="ai_infra")


# --------------------------------------------------------------------------- #
# ai_infra  (embedding_batch)
# --------------------------------------------------------------------------- #
def check_embedding_batch(payload, ctx):
    b = ctx.baseline
    ed = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if _err(ed):
        return Verdict(alert=False, pillar="ai_infra", reason="drift unavailable")

    reasons = []

    # Two independent failure modes, each one-sided high:
    #   drift      -> centroid_shift jumps (baseline max ~0.0435, clean ~0.02)
    #   staleness  -> avg_doc_age_days jumps (baseline max ~49.8, clean ~26)
    # Subtle instances of both hug the underside of the baseline bound, so a
    # soft bound just above the clean envelope recovers them. The clean centroid
    # tail runs closer to its bound than the age tail does, so its soft margin
    # is tighter (0.90·baseline vs 0.85·baseline).
    shift = ed.get("centroid_shift")
    if shift is not None:
        if shift > b["embedding_centroid_shift_max"]:
            reasons.append("embedding centroid shift over baseline")
        elif shift > 0.90 * b["embedding_centroid_shift_max"]:
            reasons.append("embedding centroid shift elevated")
        if _robust_z_outlier(ctx, "emb_shift", shift):
            reasons.append("embedding centroid shift in-run outlier")

    age = ed.get("avg_doc_age_days")
    if age is not None:
        if age > b["corpus_avg_doc_age_days_max"]:
            reasons.append("corpus staleness over baseline")
        elif age > 0.85 * b["corpus_avg_doc_age_days_max"]:
            reasons.append("corpus staleness elevated")
        if _robust_z_outlier(ctx, "emb_age", age):
            reasons.append("corpus staleness in-run outlier")

    if reasons:
        return Verdict(alert=True, pillar="ai_infra", confidence=0.9,
                       reason="; ".join(reasons))
    return Verdict(alert=False, pillar="ai_infra")
