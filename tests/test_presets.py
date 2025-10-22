from core.presets import get_presets, list_presets, get_preset
from core.model import plan, Params

def test_presets_exist():
    names = set(list_presets())
    assert {"POC","Pilot","Prod"}.issubset(names)

def test_poc_plan_runs():
    p = get_preset("POC")
    res = plan(p)
    assert res["cost"]["per_query"] > 0
    assert res["latency"]["p95_s"] > 0

def test_presets_produce_positive_costs():
    presets = get_presets()
    for name, p in presets.items():
        res = plan(p)
        assert res["cost"]["per_query"] > 0

def test_cost_monotonic_with_context_isolated():
    base = Params(
        T_ctx=1000,
        T_prompt=120,
        T_resp=180,
        qps=5.0,
        cache_hit=0.4,
        cache_savings=0.8,
        batch=4,
        price_in=0.5,
        price_out=1.5,
        tps_prefill=20000,
        tps_decode=150,
        net_ms_one_way=50
    )
    a = plan(base)["cost"]["per_query"]
    b = plan(Params(**{**base.__dict__,"T_ctx":2000}))["cost"]["per_query"]
    c = plan(Params(**{**base.__dict__,"T_ctx":3000}))["cost"]["per_query"]
    assert b >= a
    assert c >= b
