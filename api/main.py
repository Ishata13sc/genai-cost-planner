import os
import io
import csv
import json
import time
import pathlib
from typing import Dict, Any, List

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from core.model import Params, plan
from core.recommend import optimize
from core.presets import list_presets, get_preset
from core.pricing import list_profiles as list_price_profiles, get_profile as get_price_profile


def _read_version() -> str:
    try:
        return (pathlib.Path(__file__).resolve().parents[1] / "VERSION").read_text().strip()
    except Exception:
        return "0.0.0"


APP_VERSION = _read_version()
app = FastAPI(title="GenAI Cost Planner API", version=APP_VERSION)

origins_env = os.getenv("ALLOWED_ORIGINS", "*")
origins = ["*"] if origins_env.strip() == "*" else [o.strip() for o in origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY", "").strip()
RATE_LIMIT_RPS = float(os.getenv("RATE_LIMIT_RPS", "0"))
_rate_state: Dict[str, Dict[str, float]] = {}
PUBLIC_PATHS = {"/", "/healthz", "/favicon.ico", "/docs", "/redoc", "/openapi.json", "/version"}


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
    servers: int = 1
    burst_factor: float = 1.0


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


class ApplyProfileRequest(BaseModel):
    profile: str
    params: PlanRequest


async def auth_and_rate(request: Request, call_next):
    path = request.url.path
    if API_KEY and not (path in PUBLIC_PATHS or path.startswith("/pricing/profiles")):
        if request.headers.get("x-api-key", "") != API_KEY:
            raise HTTPException(status_code=401, detail="invalid api key")
    if RATE_LIMIT_RPS > 0 and path not in PUBLIC_PATHS:
        now = time.time()
        ip = request.client.host if request.client else "unknown"
        state = _rate_state.get(ip, {"t": now, "c": 0})
        if now - state["t"] > 1.0:
            state = {"t": now, "c": 0}
        state["c"] += 1
        _rate_state[ip] = state
        if state["c"] > RATE_LIMIT_RPS:
            raise HTTPException(status_code=429, detail="rate limit exceeded")
    return await call_next(request)


@app.get("/")
def root():
    return RedirectResponse("/docs")


@app.get("/healthz")
def healthz() -> Dict[str, bool]:
    return {"ok": True}


@app.get("/version")
def version() -> Dict[str, str]:
    return {"version": APP_VERSION}


@app.get("/presets")
def presets() -> Dict[str, Any]:
    return {"presets": list_presets()}


@app.get("/presets/{name}")
def preset(name: str) -> Dict[str, Any]:
    p = get_preset(name)
    return {"name": name, "params": p.__dict__}


@app.get("/pricing/profiles")
def pricing_profiles() -> Dict[str, Any]:
    return {"profiles": list_price_profiles()}


@app.get("/pricing/profiles/{name}")
def pricing_profile(name: str) -> Dict[str, Any]:
    return {"name": name, "profile": get_price_profile(name)}


@app.post("/pricing/apply")
def pricing_apply(body: ApplyProfileRequest) -> Dict[str, Any]:
    prof = get_price_profile(body.profile)
    merged = body.params.model_dump()
    merged.update(prof)
    return {"params": merged, "profile": body.profile}


@app.post("/plan")
def plan_endpoint(body: PlanRequest) -> Dict[str, Any]:
    p = Params(**body.model_dump())
    return plan(p)


@app.post("/optimize")
def optimize_endpoint(body: OptimizeRequest) -> Dict[str, Any]:
    p = Params(**body.params.model_dump())
    out = optimize(p, sla_p95=body.sla_p95, rho_target=body.rho_target)
    payload: Dict[str, Any] = {"base_cost": out["base_cost"], "count": out["count"], "best": None}
    if out["best"] is not None:
        bp = out["best"]["params"]
        br = out["best"]["result"]
        payload["best"] = {"params": bp.__dict__, "result": br, "score": out["best"]["score"]}
    return payload


@app.post("/export")
def export_endpoint(body: ExportRequest):
    rows: List[Dict[str, Any]] = []
    for it in body.items:
        p = Params(**it.params.model_dump())
        r = plan(p)
        rows.append(
            {
                "name": it.name,
                "cost_per_query": r["cost"]["per_query"],
                "cost_per_1k": r["cost"]["per_1k"],
                "p50_s": r["latency"]["p50_s"],
                "p95_s": r["latency"]["p95_s"],
                "rho": r["latency"]["rho"],
                "mu_qps": r["latency"]["mu_qps"],
            }
        )
    if body.format.lower() == "json":
        return Response(json.dumps(rows, ensure_ascii=False), media_type="application/json")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return Response(output.getvalue(), media_type="text/csv")
