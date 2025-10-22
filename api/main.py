import os
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
from core.model import Params, plan
from core.recommend import optimize
from core.presets import list_presets, get_preset
import io
import csv
import json

class PlanRequest(BaseModel):
    T_ctx: float
    T_prompt: float
    T_resp: float
    qps: float
    cache_hit: float
    cache_savings: float
    batch: int
    price_in: float
    price_out: float
    tps_prefill: float
    tps_decode: float
    net_ms_one_way: float = 0.0

class OptimizeRequest(BaseModel):
    params: PlanRequest
    sla_p95: float = 2.0
    rho_target: float = 0.7

class ExportItem(BaseModel):
    name: str
    params: PlanRequest

class ExportRequest(BaseModel):
    items: List[ExportItem]
    format: str = "csv"

app = FastAPI(title="GenAI Cost Planner API", version="1.1.0")

origins_env = os.getenv("ALLOWED_ORIGINS", "*")
origins = ["*"] if origins_env.strip() == "*" else [o.strip() for o in origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/presets")
def presets() -> Dict[str, Any]:
    return {"presets": list_presets()}

@app.get("/presets/{name}")
def preset(name: str):
    p = get_preset(name)
    return {"name": name, "params": p.__dict__}

@app.post("/plan")
def plan_endpoint(body: PlanRequest):
    p = Params(**body.model_dump())
    r = plan(p)
    return r

@app.post("/optimize")
def optimize_endpoint(body: OptimizeRequest):
    p = Params(**body.params.model_dump())
    out = optimize(p, sla_p95=body.sla_p95, rho_target=body.rho_target)
    payload = {"base_cost": out["base_cost"], "count": out["count"], "best": None}
    if out["best"] is not None:
        bp = out["best"]["params"]
        br = out["best"]["result"]
        payload["best"] = {"params": bp.__dict__, "result": br, "score": out["best"]["score"]}
    return payload

@app.post("/export")
def export_endpoint(body: ExportRequest):
    rows = []
    for it in body.items:
        p = Params(**it.params.model_dump())
        r = plan(p)
        rows.append({
            "name": it.name,
            "cost_per_query": r["cost"]["per_query"],
            "cost_per_1k": r["cost"]["per_1k"],
            "p50_s": r["latency"]["p50_s"],
            "p95_s": r["latency"]["p95_s"],
            "rho": r["latency"]["rho"],
            "mu_qps": r["latency"]["mu_qps"]
        })
    if body.format.lower() == "json":
        payload = json.dumps(rows, ensure_ascii=False)
        return Response(payload, media_type="application/json")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return Response(output.getvalue(), media_type="text/csv")
