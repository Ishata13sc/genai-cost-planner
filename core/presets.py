from .model import Params

def get_presets():
    return {
        "POC": Params(
            T_ctx=1000,
            T_prompt=100,
            T_resp=150,
            qps=1.0,
            cache_hit=0.2,
            cache_savings=0.8,
            batch=2,
            price_in=0.5,
            price_out=1.5,
            tps_prefill=20000,
            tps_decode=150,
            net_ms_one_way=50
        ),
        "Pilot": Params(
            T_ctx=2000,
            T_prompt=150,
            T_resp=200,
            qps=5.0,
            cache_hit=0.4,
            cache_savings=0.8,
            batch=4,
            price_in=0.5,
            price_out=1.5,
            tps_prefill=20000,
            tps_decode=150,
            net_ms_one_way=50
        ),
        "Prod": Params(
            T_ctx=3000,
            T_prompt=120,
            T_resp=180,
            qps=20.0,
            cache_hit=0.6,
            cache_savings=0.9,
            batch=8,
            price_in=0.5,
            price_out=1.5,
            tps_prefill=20000,
            tps_decode=150,
            net_ms_one_way=40
        )
    }

def list_presets():
    return list(get_presets().keys())

def get_preset(name: str) -> Params:
    return get_presets()[name]
