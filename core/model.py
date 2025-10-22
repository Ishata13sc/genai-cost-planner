from dataclasses import dataclass, asdict
import math
from typing import Dict, Any

@dataclass
class Params:
    T_ctx: float
    T_prompt: float
    T_resp: float
    qps: float
    cache_hit: float
    cache_savings: float
    batch: int
    price_in: float
    price_out: float
    tps_prefill: float
    tps_decode: float
    net_ms_one_way: float = 0.0

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _safe_pos(x, eps=1e-9):
    return max(eps, x)

def plan(p: Params) -> Dict[str, Any]:
    h = _clamp(p.cache_hit, 0.0, 1.0)
    s = _clamp(p.cache_savings, 0.0, 1.0)
    b = max(1, int(p.batch))
    T_ctx = max(0.0, float(p.T_ctx))
    T_prompt = max(0.0, float(p.T_prompt))
    T_resp = max(0.0, float(p.T_resp))
    lam = max(0.0, float(p.qps))
    P_in = max(0.0, float(p.price_in))
    P_out = max(0.0, float(p.price_out))
    tps_prefill = _safe_pos(float(p.tps_prefill))
    tps_decode = _safe_pos(float(p.tps_decode))
    net_rtt_s = max(0.0, 2.0 * float(p.net_ms_one_way)) / 1000.0
    T_ctx_eff = (1.0 - h * s) * T_ctx
    T_ctx_eff_batch = T_ctx_eff / b
    T_in_per_query = T_ctx_eff_batch + T_prompt
    T_out_per_query = T_resp
    cost_in = (T_in_per_query / 1000.0) * P_in
    cost_out = (T_out_per_query / 1000.0) * P_out
    cost_q = cost_in + cost_out
    cost_1k = 1000.0 * cost_q
    s_prefill = T_in_per_query / tps_prefill
    s_decode = T_out_per_query / tps_decode
    s_base = s_prefill + s_decode + net_rtt_s
    mu = 1.0 / _safe_pos(s_base)
    rho = _clamp(lam / mu, 0.0, 0.999999)
    stable = lam < mu
    def wait_percentile(pct: float) -> float:
        if not stable:
            return float("inf")
        denom = max(1e-9, (mu - lam))
        return -math.log(max(1e-12, 1.0 - pct)) / denom
    L_p50 = s_base + wait_percentile(0.50)
    L_p95 = s_base + wait_percentile(0.95)
    qpd = lam * 86400.0
    cost_day = qpd * cost_q
    cost_month = 30.0 * cost_day
    recs = []
    if not stable:
        recs.append("Queue is unstable (λ ≥ μ): reduce QPS or scale out, or increase throughput.")
    if rho > 0.70 and stable:
        recs.append("High utilization (ρ>0.7): increase batch (if shared context) or scale out / lower QPS.")
    if T_ctx > 2000 and h < 0.5:
        recs.append("Input cost is dominant: prune context or improve cache hit/savings.")
    if L_p95 != float("inf") and L_p95 > 2.0:
        recs.append("p95 exceeds 2s: reduce response length (T_resp) or use a faster decode model.")
    if cost_out > cost_in:
        recs.append("Output cost dominates: reduce T_resp or return a more compact format.")
    safe_qps = mu * 0.7
    return {
        "inputs": asdict(p),
        "tokens": {
            "T_ctx_eff": T_ctx_eff,
            "T_ctx_eff_batch": T_ctx_eff_batch,
            "T_in_per_query": T_in_per_query,
            "T_out_per_query": T_out_per_query
        },
        "cost": {
            "in_per_query": cost_in,
            "out_per_query": cost_out,
            "per_query": cost_q,
            "per_1k": cost_1k,
            "per_day": cost_day,
            "per_month": cost_month
        },
        "latency": {
            "prefill_s": s_prefill,
            "decode_s": s_decode,
            "service_base_s": s_base,
            "p50_s": L_p50,
            "p95_s": L_p95,
            "rho": rho,
            "mu_qps": mu,
            "stable": stable,
            "safe_qps": safe_qps
        },
        "recommendations": recs,
        "version": "stage1-core-1.1"
    }
