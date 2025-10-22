from .model import plan, Params

def suggest(params, res, sla_p95=2.0, rho_target=0.7):
    rec = []
    if res["latency"]["rho"] > rho_target:
        rec.append("Utilization exceeds target: increase batch or reduce QPS / scale out.")
    if res["latency"]["p95_s"] > sla_p95:
        rec.append(f"p95 exceeds SLA {sla_p95:.2f}s: reduce response length, choose a faster decode model, or increase batch.")
    if res["cost"]["out_per_query"] > res["cost"]["in_per_query"]:
        rec.append("Output cost dominates: shorten responses or return a more compact format.")
    if res["cost"]["in_per_query"] > res["cost"]["out_per_query"]:
        rec.append("Input cost dominates: prune context, improve cache hit/savings, or share context via batching.")
    if not rec:
        rec.append("Configuration looks balanced for the current load.")
    return rec

def optimize(params, sla_p95=2.0, rho_target=0.7):
    base = plan(params)
    base_cost = base["cost"]["per_query"]
    T_resp_opts = sorted(set([max(1, int(params.T_resp * x)) for x in (0.5, 0.75, 1.0, 1.25)]))
    T_ctx_opts = sorted(set([max(0, int(params.T_ctx * x)) for x in (1.0, 0.75, 0.5)]))
    batch_opts = list(range(1, 65))
    best = None
    count = 0
    for ctx in T_ctx_opts:
        for resp in T_resp_opts:
            for b in batch_opts:
                p = Params(**{**params.__dict__, "T_ctx": ctx, "T_resp": resp, "batch": b})
                r = plan(p)
                if r["latency"]["p95_s"] <= sla_p95 and r["latency"]["rho"] <= rho_target:
                    count += 1
                    score = (r["cost"]["per_query"], r["latency"]["p95_s"])
                    item = {"params": p, "result": r, "score": score}
                    if best is None or score < best["score"]:
                        best = item
    return {"best": best, "base_cost": base_cost, "count": count}
