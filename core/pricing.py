import json
import pathlib
from typing import Dict, Any
from .model import Params

def _profiles_path():
    return pathlib.Path(__file__).resolve().parents[1] / "data" / "pricing_profiles.json"

def load_profiles() -> Dict[str, Dict[str, float]]:
    p = _profiles_path()
    if not p.exists():
        return {
            "default": {"price_in": 0.5, "price_out": 1.5, "tps_prefill": 20000, "tps_decode": 150}
        }
    return json.loads(p.read_text())

def list_profiles():
    return sorted(load_profiles().keys())

def get_profile(name: str) -> Dict[str, float]:
    profiles = load_profiles()
    return profiles.get(name, profiles["default"])

def apply_profile(p: Params, profile_name: str) -> Params:
    prof = get_profile(profile_name)
    return Params(
        T_ctx=p.T_ctx, T_prompt=p.T_prompt, T_resp=p.T_resp, qps=p.qps,
        cache_hit=p.cache_hit, cache_savings=p.cache_savings, batch=p.batch,
        price_in=prof["price_in"], price_out=prof["price_out"],
        tps_prefill=prof["tps_prefill"], tps_decode=prof["tps_decode"],
        net_ms_one_way=p.net_ms_one_way, servers=p.servers, burst_factor=p.burst_factor
    )
