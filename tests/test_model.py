from core.model import Params, plan

def test_cost_increases_with_more_context():
    a = plan(Params(1000,100,150,1,0.0,0.0,1,0.5,1.5,20000,150,50))
    b = plan(Params(2000,100,150,1,0.0,0.0,1,0.5,1.5,20000,150,50))
    assert b['cost']['per_query'] > a['cost']['per_query']

def test_cache_and_batch_reduce_input_cost():
    no_opt = plan(Params(1000,100,150,1,0.0,0.0,1,0.5,1.5,20000,150,50))
    cached_batched = plan(Params(1000,100,150,1,0.6,1.0,5,0.5,1.5,20000,150,50))
    assert cached_batched['cost']['in_per_query'] < no_opt['cost']['in_per_query']

def test_latency_grows_with_qps():
    low = plan(Params(1000,100,150,0.5,0.2,0.8,5,0.5,1.5,20000,150,50))
    high = plan(Params(1000,100,150,5.0,0.2,0.8,5,0.5,1.5,20000,150,50))
    assert high['latency']['p95_s'] >= low['latency']['p95_s']

def test_nonnegative_and_stability():
    res = plan(Params(0,0,0,0,1,1,64,0,0,1e9,1e9,0))
    assert res['cost']['per_query'] >= 0
    assert res['latency']['p50_s'] >= 0
    assert 0 <= res['latency']['rho'] < 1
