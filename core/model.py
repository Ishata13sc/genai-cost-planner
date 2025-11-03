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
    servers: int = 1
    burst_factor: float = 1.0

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _safe_pos(x, eps=1e-9):
    return max(eps, x)

def _erlang_c(lam, mu, k):
    if k <= 1:
        rho = lam / mu if mu > 0 else 1.0
        rho = min(rho, 0.999999)
        wq = rho / (mu - lam) if lam < mu else float("inf")
        return rho, rho, max(0.0, 1.0 - rho), wq
    a = lam / mu if mu > 0 else float("inf")
    rho_k = a / k if k > 0 else 1.0
    if rho_k >= 1.0:
        return rho_k, 1.0, 0.0, float("inf")
    s = 0.0
    for n in range(k):
        s += (a**n) / math.factorial(n)
    pn = (a**k) / math.factorial(k) * (1.0 / (1.0 - rho_k))
    p0 = 1.0 / (s + pn)
    pw = pn * p0
    wq = pw / (k * mu - lam)
    return rho_k, pw, p0, wq

def plan(p: Params) -> Dict[str, Any]:
    h = _clamp(p.cache_hit, 0.0, 1.0)
    s = _clamp(p.cache_savings, 0.0, 1.0)
    b = max(1, int(p.batch))
    k = max(1, int(p.servers))
    T_ctx = max(0.0, float(p.T_ctx))
    T_prompt = max(0.0, float(p.T_prompt))
    T_resp = max(0.0, float(p.T_resp))
    lam = max(0.0, float(p.qps)) * max(1.0, float(p.burst_factor))
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
    rho_k, p_wait, p0, wq = _erlang_c(lam, mu, k)
    stable = lam < k * mu
    w_p50 = wq
    w_p95 = wq * 3.0
    L_p50 = s_base + w_p50
    L_p95 = s_base + w_p95
    if not stable:
        L_p50 = float("inf")
        L_p95 = float("inf")
    qpd = (lam / max(1.0, float(p.burst_factor))) * 86400.0
    cost_day = qpd * cost_q
    cost_month = 30.0 * cost_day
    mu_total = k * mu
    rho_total = lam / mu_total if mu_total > 0 else 1.0
    safe_qps = mu_total * 0.7
    recs = []
    if not stable:
        recs.append("Queue is unstable: increase servers or reduce QPS.")
    if T_ctx > 2000 and h < 0.5:
        recs.append("Input cost is dominant: prune context or improve cache hit/savings.")
    if L_p95 != float("inf") and L_p95 > 2.0:
        recs.append("p95 exceeds 2s: reduce response length or use a faster decode model.")
    if cost_out > cost_in:
        recs.append("Output cost dominates: reduce T_resp or return a more compact format.")
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
            "rho": rho_total,
            "mu_qps": mu_total,
            "stable": stable,
            "safe_qps": safe_qps,
            "p_wait": p_wait
        },
        "recommendations": recs,
        "version": "stage1-core-2.0"
    }
from functools import lru_cache

def _key_from_params(p: Params):
    return (
        p.T_ctx, p.T_prompt, p.T_resp, p.qps, p.cache_hit, p.cache_savings, int(p.batch),
        p.price_in, p.price_out, p.tps_prefill, p.tps_decode, p.net_ms_one_way,
        int(getattr(p, "servers", 1)), float(getattr(p, "burst_factor", 1.0))
    )

@lru_cache(maxsize=8192)
def _plan_cached_key(key):
    return plan(Params(
        T_ctx=key[0], T_prompt=key[1], T_resp=key[2], qps=key[3],
        cache_hit=key[4], cache_savings=key[5], batch=key[6],
        price_in=key[7], price_out=key[8], tps_prefill=key[9], tps_decode=key[10],
        net_ms_one_way=key[11], servers=key[12], burst_factor=key[13]
    ))

def plan_cached(p: Params):
    return _plan_cached_key(_key_from_params(p))
