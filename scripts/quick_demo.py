import json, pathlib
from core.model import Params, plan

path = pathlib.Path(__file__).resolve().parents[1] / "examples" / "scenarios.json"
scenarios = json.loads(path.read_text())

for s in scenarios:
    p = Params(**s["params"])
    r = plan(p)
    status = "UNSTABLE" if not r["latency"]["stable"] else "OK"
    print(f"== {s['name']} == [{status}]")
    print(f"cost/query: ${r['cost']['per_query']:.4f} | cost/1k: ${r['cost']['per_1k']:.2f}")
    p50 = "∞" if r["latency"]["p50_s"] == float("inf") else f"{r['latency']['p50_s']:.3f}s"
    p95 = "∞" if r["latency"]["p95_s"] == float("inf") else f"{r['latency']['p95_s']:.3f}s"
    print(f"p50: {p50} | p95: {p95} | rho: {r['latency']['rho']:.2f} | mu: {r['latency']['mu_qps']:.2f} qps")
    if not r["latency"]["stable"]:
        print(f"safe_qps≈{r['latency']['safe_qps']:.2f}")
    print("")
