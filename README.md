![CI](https://github.com/Ishata13sc/genai-cost-planner/actions/workflows/ci.yml/badge.svg)


# GenAI Cost Planner — Token & Throughput Simulator

A vendor-neutral calculator to estimate GenAI cost and latency. Plug in context/prompt/response lengths, QPS, cache, batch size, and token prices. Get cost per query/1k/day/month, p50/p95 latency, utilization, and optimization hints.

## Why this exists
Teams need quick answers to “How much and how fast?” for POC→pilot→production. This tool compares scenarios such as cache vs no cache, batching, and context length while staying model/cloud agnostic.

## Features
- Cost breakdown: input vs output, per-query, per-1k, daily, monthly
- Latency estimation: p50, p95, utilization ρ, service capacity μ
- Presets: POC, Pilot, Prod
- What-if charts: p95 vs batch, cost vs context
- Lightweight optimizer: minimize cost under p95 SLA and ρ cap
- Streamlit UI and pure-Python core

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
streamlit run app.py
