from core.model import Params, plan
from core.recommend import suggest, optimize

def test_suggest_runs():
    p = Params(1000,100,150,1.0,0.2,0.8,2,0.5,1.5,20000,150,50)
    r = plan(p)
    s = suggest(p, r, sla_p95=2.0, rho_target=0.8)
    assert isinstance(s, list)

def test_optimize_basic():
    p = Params(1000,100,150,1.0,0.2,0.8,2,0.5,1.5,20000,150,50)
    out = optimize(p, sla_p95=10.0, rho_target=0.95)
    assert isinstance(out, dict)
