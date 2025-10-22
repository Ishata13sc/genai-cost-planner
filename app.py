import io
import streamlit as st
import numpy as np
import pandas as pd
from core.model import Params, plan
from core.presets import get_preset, list_presets
from core.recommend import suggest, optimize

st.set_page_config(page_title="GenAI Cost Planner", layout="wide")
st.title("GenAI Cost Planner — Token & Throughput Simulator")

if "current_params" not in st.session_state:
    st.session_state.current_params = get_preset(list_presets()[0])

with st.sidebar:
    st.header("Preset")
    preset_name = st.selectbox("Pilih preset", list_presets(), index=0)
    if st.button("Load preset"):
        st.session_state.current_params = get_preset(preset_name)

    st.header("Inputs")
    p0 = st.session_state.current_params
    T_ctx = st.number_input("Context length (tokens)", min_value=0, max_value=500000, value=int(p0.T_ctx), step=50)
    T_prompt = st.number_input("Prompt tokens", min_value=0, max_value=200000, value=int(p0.T_prompt), step=10)
    T_resp = st.number_input("Response tokens", min_value=0, max_value=200000, value=int(p0.T_resp), step=10)
    qps = st.number_input("QPS (queries/sec)", min_value=0.0, max_value=100000.0, value=float(p0.qps), step=0.1, format="%.2f")
    cache_hit = st.slider("Cache hit-rate", min_value=0.0, max_value=1.0, value=float(p0.cache_hit), step=0.01)
    cache_savings = st.slider("Cache savings (0..1)", min_value=0.0, max_value=1.0, value=float(p0.cache_savings), step=0.05)
    batch = st.number_input("Batch size", min_value=1, max_value=2048, value=int(p0.batch), step=1)
    price_in = st.number_input("Price input per 1k tokens ($)", min_value=0.0, max_value=100.0, value=float(p0.price_in), step=0.01, format="%.4f")
    price_out = st.number_input("Price output per 1k tokens ($)", min_value=0.0, max_value=100.0, value=float(p0.price_out), step=0.01, format="%.4f")
    tps_prefill = st.number_input("Prefill TPS (tok/s)", min_value=1.0, max_value=1e7, value=float(p0.tps_prefill), step=100.0, format="%.1f")
    tps_decode = st.number_input("Decode TPS (tok/s)", min_value=1.0, max_value=1e7, value=float(p0.tps_decode), step=10.0, format="%.1f")
    net_ms_one_way = st.number_input("Network one-way (ms)", min_value=0.0, max_value=10000.0, value=float(p0.net_ms_one_way), step=5.0, format="%.0f")
    sla_p95 = st.number_input("SLA target p95 (s)", min_value=0.1, max_value=60.0, value=2.0, step=0.1, format="%.1f")
    rho_target = st.slider("Utilization cap ρ", min_value=0.5, max_value=0.95, value=0.7, step=0.01)
    if st.button("Set sebagai baseline"):
        st.session_state.current_params = Params(
            T_ctx=T_ctx, T_prompt=T_prompt, T_resp=T_resp, qps=qps,
            cache_hit=cache_hit, cache_savings=cache_savings, batch=batch,
            price_in=price_in, price_out=price_out,
            tps_prefill=tps_prefill, tps_decode=tps_decode, net_ms_one_way=net_ms_one_way
        )

params = st.session_state.current_params
res = plan(params)

if not res["latency"]["stable"]:
    st.error(f"Queue is unstable (λ ≥ μ). Reduce QPS below {res['latency']['mu_qps']:.2f} or scale out. Safe QPS ≈ {res['latency']['safe_qps']:.2f}.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Cost / query", f"${res['cost']['per_query']:.4f}")
c2.metric("Cost / 1k queries", f"${res['cost']['per_1k']:.2f}")
c3.metric("Daily cost", f"${res['cost']['per_day']:.2f}")
c4.metric("Monthly cost (30d)", f"${res['cost']['per_month']:.2f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("p50 latency", "∞" if res["latency"]["p50_s"] == float("inf") else f"{res['latency']['p50_s']:.3f}s")
c6.metric("p95 latency", "∞" if res["latency"]["p95_s"] == float("inf") else f"{res['latency']['p95_s']:.3f}s")
c7.metric("Utilization ρ", f"{res['latency']['rho']:.2f}")
c8.metric("Service capacity μ", f"{res['latency']['mu_qps']:.2f} qps")

st.subheader("Capacity planning")
safe_qps = res["latency"]["safe_qps"]
max_qps_instance = res["latency"]["mu_qps"]
instances_needed = int(np.ceil(params.qps / safe_qps)) if safe_qps > 0 else 0
c9, c10, c11 = st.columns(3)
c9.metric("Max QPS per instance (μ)", f"{max_qps_instance:.2f}")
c10.metric("Safe QPS/instance (ρ≤0.7)", f"{safe_qps:.2f}")
c11.metric("Instances needed", f"{instances_needed}")

if instances_needed > 0 and st.button("Apply scaling suggestion"):
    new_qps = params.qps / instances_needed if instances_needed > 0 else params.qps
    st.session_state.current_params = Params(
        T_ctx=params.T_ctx, T_prompt=params.T_prompt, T_resp=params.T_resp,
        qps=new_qps, cache_hit=params.cache_hit, cache_savings=params.cache_savings,
        batch=int(params.batch), price_in=params.price_in, price_out=params.price_out,
        tps_prefill=params.tps_prefill, tps_decode=params.tps_decode, net_ms_one_way=params.net_ms_one_way
    )
    st.success(f"Applied: total QPS split across {instances_needed} instances → per-instance QPS ≈ {new_qps:.3f}.")

st.subheader("Recommendations")
for r in suggest(params, res, sla_p95=sla_p95, rho_target=rho_target):
    st.write("• " + r)

st.subheader("What-if: p95 latency vs batch size")
b_values = np.arange(1, 33)
p95_list = []
for b in b_values:
    tmp = Params(
        T_ctx=params.T_ctx, T_prompt=params.T_prompt, T_resp=params.T_resp,
        qps=params.qps, cache_hit=params.cache_hit, cache_savings=params.cache_savings,
        batch=int(b), price_in=params.price_in, price_out=params.price_out,
        tps_prefill=params.tps_prefill, tps_decode=params.tps_decode, net_ms_one_way=params.net_ms_one_way
    )
    v = plan(tmp)["latency"]["p95_s"]
    p95_list.append(v if np.isfinite(v) else np.nan)
df_batch = pd.DataFrame({"batch": b_values, "p95 (s)": p95_list}).set_index("batch")
st.line_chart(df_batch)

st.subheader("What-if: Cost per query vs context length")
ctx_max = max(1, int(params.T_ctx * 2))
ctx_values = np.linspace(0, ctx_max, num=30)
cost_list = []
for ctx in ctx_values:
    tmp = Params(
        T_ctx=float(ctx), T_prompt=params.T_prompt, T_resp=params.T_resp,
        qps=params.qps, cache_hit=params.cache_hit, cache_savings=params.cache_savings,
        batch=int(params.batch), price_in=params.price_in, price_out=params.price_out,
        tps_prefill=params.tps_prefill, tps_decode=params.tps_decode, net_ms_one_way=params.net_ms_one_way
    )
    cost_list.append(plan(tmp)["cost"]["per_query"])
df_ctx = pd.DataFrame({"context": ctx_values, "cost/query ($)": cost_list}).set_index("context")
st.line_chart(df_ctx)

st.subheader("Export")
if st.button("Download CSV (batch sweep)"):
    csv_buf = io.StringIO()
    pd.DataFrame({"batch": b_values, "p95 (s)": p95_list}).to_csv(csv_buf, index=False)
    st.download_button("Save CSV", csv_buf.getvalue(), file_name="batch_sweep.csv", mime="text/csv")

st.subheader("Optimizer")
run_opt = st.button("Run optimizer (min cost subject to SLA)")
if run_opt:
    out = optimize(params, sla_p95=sla_p95, rho_target=rho_target)
    if out["best"] is None:
        st.warning("No candidate meets the SLA and utilization cap.")
    else:
        best = out["best"]
        bp = best["params"]
        br = best["result"]
        delta = br["cost"]["per_query"] - out["base_cost"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best cost / query", f"${br['cost']['per_query']:.4f}", f"{delta:+.4f}")
        c2.metric("Best p95", f"{br['latency']['p95_s']:.3f}s")
        c3.metric("Batch", f"{bp.batch}")
        st.write(f"T_ctx={bp.T_ctx}, T_resp={bp.T_resp}, batch={bp.batch}, hit={bp.cache_hit}, savings={bp.cache_savings}")
        if st.button("Apply best as baseline"):
            st.session_state.current_params = bp

with st.expander("Token breakdown"):
    t = res["tokens"]
    st.write(
        f"T_ctx_eff: {t['T_ctx_eff']:.2f}, "
        f"T_ctx_eff_batch: {t['T_ctx_eff_batch']:.2f}, "
        f"T_in_per_query: {t['T_in_per_query']:.2f}, "
        f"T_out_per_query: {t['T_out_per_query']:.2f}"
    )
