import io
import numpy as np
import pandas as pd
import streamlit as st

from core.model import Params, plan, plan_cached
from core.presets import get_preset, list_presets
from core.recommend import suggest, optimize
from core.pricing import (
    list_profiles as list_price_profiles,
    get_profile as get_price_profile,
)

# ---------- Streamlit setup ----------
st.set_page_config(page_title="GenAI Cost Planner", layout="wide")
st.title("GenAI Cost Planner — Token & Throughput Simulator")

# ---------- Helpers ----------
def apply_pricing_profile(base: Params, profile_name: str) -> Params:
    """Return Params with pricing/tps from pricing profile, keep the rest from base."""
    prof = get_price_profile(profile_name)
    return Params(
        T_ctx=base.T_ctx,
        T_prompt=base.T_prompt,
        T_resp=base.T_resp,
        qps=base.qps,
        cache_hit=base.cache_hit,
        cache_savings=base.cache_savings,
        batch=int(base.batch),
        price_in=float(prof["price_in"]),
        price_out=float(prof["price_out"]),
        tps_prefill=float(prof["tps_prefill"]),
        tps_decode=float(prof["tps_decode"]),
        net_ms_one_way=base.net_ms_one_way,
        servers=int(getattr(base, "servers", 1)),
        burst_factor=float(getattr(base, "burst_factor", 1.0)),
    )

def params_from_inputs(
    T_ctx, T_prompt, T_resp, qps, cache_hit, cache_savings, batch,
    price_in, price_out, tps_prefill, tps_decode, net_ms_one_way, servers, burst_factor
) -> Params:
    return Params(
        T_ctx=float(T_ctx),
        T_prompt=float(T_prompt),
        T_resp=float(T_resp),
        qps=float(qps),
        cache_hit=float(cache_hit),
        cache_savings=float(cache_savings),
        batch=int(batch),
        price_in=float(price_in),
        price_out=float(price_out),
        tps_prefill=float(tps_prefill),
        tps_decode=float(tps_decode),
        net_ms_one_way=float(net_ms_one_way),
        servers=int(servers),
        burst_factor=float(burst_factor),
    )

# ---------- Session state baseline ----------
if "current_params" not in st.session_state:
    st.session_state.current_params = get_preset(list_presets()[0])

# =========================================================
# Sidebar controls
# =========================================================
with st.sidebar:
    # Pricing profile (single, konsisten)
    st.header("Pricing Profile")
    price_profiles = list_price_profiles()
    if not price_profiles:
        st.warning("No pricing profiles found.")
        prof_name = None
    else:
        # default -> pilih index-nya kalau ada
        default_idx = price_profiles.index("default") if "default" in price_profiles else 0
        prof_name = st.selectbox("Profile", price_profiles, index=default_idx)

    if prof_name and st.button("Apply profile"):
        st.session_state.current_params = apply_pricing_profile(
            st.session_state.current_params, prof_name
        )
        st.success(f"Applied pricing profile: {prof_name}")

    # Preset (input template)
    st.header("Preset")
    preset_name = st.selectbox("Choose preset", list_presets(), index=0)
    if st.button("Load preset"):
        st.session_state.current_params = get_preset(preset_name)
        st.success(f"Loaded preset: {preset_name}")

    # Inputs
    st.header("Inputs")
    p0 = st.session_state.current_params

    T_ctx = st.number_input("Context length (tokens)", 0, 500000, int(p0.T_ctx), step=50)
    T_prompt = st.number_input("Prompt tokens", 0, 200000, int(p0.T_prompt), step=10)
    T_resp = st.number_input("Response tokens", 0, 200000, int(p0.T_resp), step=10)
    qps = st.number_input("QPS (queries/sec)", 0.0, 100000.0, float(p0.qps), step=0.1, format="%.2f")
    cache_hit = st.slider("Cache hit-rate", 0.0, 1.0, float(p0.cache_hit), 0.01)
    cache_savings = st.slider("Cache savings (0..1)", 0.0, 1.0, float(p0.cache_savings), 0.05)
    batch = st.number_input("Batch size", 1, 2048, int(p0.batch), step=1)
    price_in = st.number_input("Price input per 1k tokens ($)", 0.0, 100.0, float(p0.price_in), step=0.01, format="%.4f")
    price_out = st.number_input("Price output per 1k tokens ($)", 0.0, 100.0, float(p0.price_out), step=0.01, format="%.4f")
    tps_prefill = st.number_input("Prefill TPS (tok/s)", 1.0, 1e7, float(p0.tps_prefill), step=100.0, format="%.1f")
    tps_decode = st.number_input("Decode TPS (tok/s)", 1.0, 1e7, float(p0.tps_decode), step=10.0, format="%.1f")
    net_ms_one_way = st.number_input("Network one-way (ms)", 0.0, 10000.0, float(p0.net_ms_one_way), step=5.0, format="%.0f")
    servers = st.number_input("Servers (k)", 1, 1024, int(getattr(p0, "servers", 1)), step=1)
    burst_factor = st.slider("Burst factor (λ multiplier)", 1.0, 5.0, float(getattr(p0, "burst_factor", 1.0)), 0.1)

    # Optimizer knobs
    st.header("SLA & Utilization Targets")
    sla_p95 = st.number_input("SLA target p95 (s)", 0.1, 60.0, 2.0, step=0.1, format="%.1f")
    rho_target = st.slider("Utilization cap ρ", 0.50, 0.95, 0.70, 0.01)

    if st.button("Set as baseline"):
        st.session_state.current_params = params_from_inputs(
            T_ctx, T_prompt, T_resp, qps, cache_hit, cache_savings, batch,
            price_in, price_out, tps_prefill, tps_decode, net_ms_one_way, servers, burst_factor
        )
        st.success("Baseline updated.")

# =========================================================
# Compute current plan
# =========================================================
params = st.session_state.current_params
res = plan(params)

if not res["latency"]["stable"]:
    st.error(
        f"Queue is unstable (λ ≥ μ). Reduce QPS below {res['latency']['mu_qps']:.2f} "
        f"or scale out. Safe QPS ≈ {res['latency']['safe_qps']:.2f}."
    )

# Metrics: cost
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cost / query", f"${res['cost']['per_query']:.4f}")
c2.metric("Cost / 1k queries", f"${res['cost']['per_1k']:.2f}")
c3.metric("Daily cost", f"${res['cost']['per_day']:.2f}")
c4.metric("Monthly cost (30d)", f"${res['cost']['per_month']:.2f}")

# Metrics: latency/capacity
c5, c6, c7, c8 = st.columns(4)
c5.metric("p50 latency", "∞" if res["latency"]["p50_s"] == float("inf") else f"{res['latency']['p50_s']:.3f}s")
c6.metric("p95 latency", "∞" if res["latency"]["p95_s"] == float("inf") else f"{res['latency']['p95_s']:.3f}s")
c7.metric("Utilization ρ", f"{res['latency']['rho']:.2f}")
c8.metric("Service capacity μ", f"{res['latency']['mu_qps']:.2f} qps")

# Capacity planning (servers)
st.subheader("Capacity planning")
safe_qps_total = res["latency"]["safe_qps"]
safe_qps_per_instance = safe_qps_total / max(1, int(getattr(params, "servers", 1)))
required_instances = int(
    np.ceil((params.qps * max(1.0, getattr(params, "burst_factor", 1.0))) / max(1e-9, safe_qps_per_instance))
)
c9, c10, c11 = st.columns(3)
c9.metric("Max QPS total (μ)", f"{res['latency']['mu_qps']:.2f}")
c10.metric("Safe QPS total (ρ≤0.7)", f"{safe_qps_total:.2f}")
c11.metric("Instances needed", f"{required_instances}")

if required_instances > getattr(params, "servers", 1) and st.button("Apply scaling suggestion"):
    new_servers = required_instances
    st.session_state.current_params = Params(
        T_ctx=params.T_ctx, T_prompt=params.T_prompt, T_resp=params.T_resp,
        qps=params.qps, cache_hit=params.cache_hit, cache_savings=params.cache_savings,
        batch=int(params.batch), price_in=params.price_in, price_out=params.price_out,
        tps_prefill=params.tps_prefill, tps_decode=params.tps_decode, net_ms_one_way=params.net_ms_one_way,
        servers=int(new_servers), burst_factor=float(getattr(params, "burst_factor", 1.0)),
    )
    st.success(f"Applied: scale to {new_servers} servers.")

# Recommendations
st.subheader("Recommendations")
try:
    for r in suggest(params, res, sla_p95=sla_p95, rho_target=rho_target):
        st.write("• " + r)
except Exception as e:
    st.info(f"(suggest) {e}")

# What-if: p95 vs batch
st.subheader("What-if: p95 latency vs batch size")
b_values = np.arange(1, 33)
p95_list = []
for b in b_values:
    tmp = Params(
        T_ctx=params.T_ctx, T_prompt=params.T_prompt, T_resp=params.T_resp,
        qps=params.qps, cache_hit=params.cache_hit, cache_savings=params.cache_savings,
        batch=int(b), price_in=params.price_in, price_out=params.price_out,
        tps_prefill=params.tps_prefill, tps_decode=params.tps_decode, net_ms_one_way=params.net_ms_one_way,
        servers=int(getattr(params, "servers", 1)), burst_factor=float(getattr(params, "burst_factor", 1.0)),
    )
    p95_list.append(plan_cached(tmp)["latency"]["p95_s"])
df_batch = pd.DataFrame({"batch": b_values, "p95 (s)": p95_list}).set_index("batch")
st.line_chart(df_batch)

# What-if: Cost vs context
st.subheader("What-if: Cost per query vs context length")
ctx_max = max(1, int(params.T_ctx * 2))
ctx_values = np.linspace(0, ctx_max, num=30)
cost_list = []
for ctx in ctx_values:
    tmp = Params(
        T_ctx=float(ctx), T_prompt=params.T_prompt, T_resp=params.T_resp,
        qps=params.qps, cache_hit=params.cache_hit, cache_savings=params.cache_savings,
        batch=int(params.batch), price_in=params.price_in, price_out=params.price_out,
        tps_prefill=params.tps_prefill, tps_decode=params.tps_decode, net_ms_one_way=params.net_ms_one_way,
        servers=int(getattr(params, "servers", 1)), burst_factor=float(getattr(params, "burst_factor", 1.0)),
    )
    cost_list.append(plan_cached(tmp)["cost"]["per_query"])
df_ctx = pd.DataFrame({"context": ctx_values, "cost/query ($)": cost_list}).set_index("context")
st.line_chart(df_ctx)

# Export
st.subheader("Export")
if st.button("Download CSV (batch sweep)"):
    csv_buf = io.StringIO()
    pd.DataFrame({"batch": b_values, "p95 (s)": p95_list}).to_csv(csv_buf, index=False)
    st.download_button("Save CSV", csv_buf.getvalue(), file_name="batch_sweep.csv", mime="text/csv")

# Optimizer
st.subheader("Optimizer")
if st.button("Run optimizer (min cost subject to SLA)"):
    out = optimize(params, sla_p95=sla_p95, rho_target=rho_target)
    if out.get("best") is None:
        st.warning("No candidate meets SLA and ρ cap.")
    else:
        best = out["best"]
        bp = best["params"]
        br = best["result"]
        delta = br["cost"]["per_query"] - out["base_cost"]
        c1o, c2o, c3o = st.columns(3)
        c1o.metric("Best cost / query", f"${br['cost']['per_query']:.4f}", f"{delta:+.4f}")
        c2o.metric("Best p95", f"{br['latency']['p95_s']:.3f}s")
        c3o.metric("Batch", f"{bp.batch}")
        st.write(
            f"T_ctx={bp.T_ctx}, T_resp={bp.T_resp}, batch={bp.batch}, "
            f"hit={bp.cache_hit}, savings={bp.cache_savings}"
        )
        if st.button("Apply best as baseline"):
            st.session_state.current_params = bp

# Token breakdown
with st.expander("Token breakdown"):
    t = res["tokens"]
    st.write(
        f"T_ctx_eff: {t['T_ctx_eff']:.2f}, "
        f"T_ctx_eff_batch: {t['T_ctx_eff_batch']:.2f}, "
        f"T_in_per_query: {t['T_in_per_query']:.2f}, "
        f"T_out_per_query: {t['T_out_per_query']:.2f}"
    )
