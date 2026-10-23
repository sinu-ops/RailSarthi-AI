# RailSarthi AI — Master Implementation & AI System Design Plan

**Project:** RailSarthi AI (Railway Journey Intelligence Platform)  
**Timeline:** October 23, 2026 – November 13, 2026 (4-Week Sprint)  
**Goal:** High-impact Final Year Engineering Project & AI/ML Engineering Portfolio  
**API Status:** RailRadar API verified working (Status 200 on Train Info & Live Tracking)

---

## 1. Executive Summary & Problem Statement

Existing railway platforms primarily provide passive train schedules and basic GPS status without predictive guidance. Passengers face uncertainty regarding:
- Cascading delay risks across multi-stop journeys.
- Station dwell-time bottlenecks and route reliability.
- Comparative evaluation between available trains on the same corridor.
- Actionable, explainable AI recommendations backed by empirical data.

RailSarthi AI bridges this gap by merging **real-time RailRadar telemetry**, **predictive XGBoost regression**, **historical route reliability analysis**, and an **agentic LangGraph Journey Copilot**.

---

## 2. Verified Data Source & Telemetry

### Active Endpoints (RailRadar v1)
- **Train Route & Schedule:** `GET https://api.railradar.in/v1/trains/{train_no}`
  - Verified response: Train metadata, distance (km), duration, run days, halt counts, full stop list with GPS coordinates (`lat`/`lng`), scheduled arrival/departure.
- **Live Status & Delay Telemetry:** `GET https://api.railradar.in/v1/trains/{train_no}/live`
  - Verified response: `delayMinutes`, `status` (`running`, `not-started`, `arrived`), `currentLocation`, `nextHalt`, stop-by-stop live updates (`route` array).

---

## 3. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAILSARTHI AI SYSTEM                     │
└─────────────────────────────────────────────────────────────┘

 [RailRadar API v1]
   ├── /trains/{no} (Schedule & Route)
   └── /trains/{no}/live (Live Delay & Location)
          │
          ▼
 [Data Collection & Validation Engine]
   ├── Ingestion Client (Circuit Breaker + Exponential Backoff)
   ├── Pydantic Schemas (Type safety, bounds checking)
   └── Rate Limiter (Token Bucket)
          │
          ├─────────────────────────┬─────────────────────────┐
          ▼                         ▼                         ▼
 [Raw JSON Audit Store]    [Processed Parquet]       [SQLite App DB]
 (data/raw/*.json)         (ML training & features)  (App state & cache)
                                    │
                                    ▼
                         [Feature Engineering]
                         ├── Stop sequence & distance %
                         ├── Cascading previous delay (lag)
                         ├── Temporal (day, month, festival)
                         └── Route reliability aggregates
                                    │
                                    ▼
                         [ML Prediction Models]
                         ├── Delay Regressor (XGBoost)
                         └── Crowd/Demand Risk Classifier
                                    │
                                    ▼
                         [FastAPI Serving Layer]
                         ├── /api/v1/trains/{no}/status
                         ├── /api/v1/trains/{no}/schedule
                         ├── /api/v1/predict/delay
                         └── /api/v1/agent/chat
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          ▼                                                   ▼
 [LangGraph Journey Copilot]                        [Frontend UI]
 ├── Intent Classification (Router)                 ├── Interactive Dashboard
 ├── Tool Calling (Status, Delay, Reliability)       ├── Route Delay Visualizer
 └── Evidence-Grounded Explainer                    └── Real-time Copilot Chat
```

---

## 4. AI System Design Concepts Matrix

| System Design Component | Theoretical Concept | Practical RailSarthi Implementation |
|---|---|---|
| **Data Ingestion** | Circuit Breaker, Exponential Backoff, Rate Limiting | `src/data_pipeline/api_client.py` handles API throttling and failures gracefully |
| **Data Integrity** | Schema validation, Type Safety, Outlier rejection | `src/data_pipeline/schemas.py` validates train numbers, station codes, and delay ranges |
| **Feature Store** | Online vs. Offline feature consistency | Shared feature generator used identically in training notebooks and inference endpoints |
| **Model Serving** | Low-latency inference, model caching, batching | Pre-warmed XGBoost model in FastAPI lifespan, TTLCache for live observations |
| **Agentic AI** | ReAct pattern, stateful graphs, tool calling | LangGraph agent with strict tool schemas to avoid hallucinations |
| **Observability** | Telemetry, latency percentiles, data drift | Request logging with execution time, confidence intervals on delay predictions |

---

## 5. Phased 4-Week Execution Roadmap

### Phase 1: Data Pipeline & Structured Storage (Days 1–5: Oct 23 – Oct 27)
- [x] Configure `.env` with RailRadar API key.
- [x] Verify API connection and inspect live payloads.
- [ ] Implement `src/data_pipeline/api_client.py` incorporating confirmed `/trains/{no}` and `/trains/{no}/live` routes.
- [ ] Build automated JSON-to-Parquet conversion pipeline:
  - `data/processed/train_schedules.parquet`
  - `data/processed/live_observations.parquet`
- [ ] Schedule regular live status snapshots across 10 key high-traffic trains (e.g., 12301, 12302, 12951, 12423, 12309).

### Phase 2: Feature Engineering & Delay Model Retraining (Days 6–10: Oct 28 – Nov 1)
- [ ] Migrate stop-by-stop lag feature generator from demo into production pipeline.
- [ ] Calculate empirical route reliability indices:
  - Historical median delay per station.
  - Delay cascade coefficient (how delay at stop $N$ propagates to stop $N+1$).
  - Severe delay probability ($>60$ min).
- [ ] Train production XGBoost delay regressor (`models/delay_model_v2.joblib`).
- [ ] Baseline evaluation: Compare XGBoost MAE against naive scheduled arrival and historical average.

### Phase 3: High-Performance FastAPI Backend (Days 11–15: Nov 2 – Nov 6)
- [ ] Setup FastAPI structure in `backend/`:
  - `GET /api/v1/trains/{train_no}/schedule`
  - `GET /api/v1/trains/{train_no}/live`
  - `POST /api/v1/predict/delay`
  - `GET /api/v1/trains/{train_no}/reliability`
- [ ] Implement in-memory TTLCache (300 seconds) for live status queries to protect API quota.
- [ ] Setup SQLite database for tracking user queries and historical observations.

### Phase 4: LangGraph Agentic Journey Copilot (Days 16–19: Nov 7 – Nov 10)
- [ ] Define LangChain tools wrapping backend endpoints:
  - `get_train_live_telemetry`
  - `predict_journey_delay`
  - `get_route_reliability_score`
- [ ] Construct LangGraph state machine:
  - State: `messages`, `extracted_entities`, `tool_results`, `grounded_explanation`.
  - Conditional routing based on user travel intent.
- [ ] Integrate Gemini API for cost-effective, hallucination-free journey guidance.

### Phase 5: User Interface, Verification & Submission Pack (Days 20–22: Nov 11 – Nov 13)
- [ ] Modernize dashboard interface (`app.py`):
  - Live train status tracker with station progress bar.
  - Interactive delay curve along route stops.
  - Side-by-side train comparison widget.
  - Integrated Copilot chat window.
- [ ] Create documentation:
  - `docs/ARCHITECTURE.md` (System design diagrams and rationale).
  - `docs/MODEL_REPORT.md` (Metrics, feature importances, ablation study).
  - 5-minute video demonstration script and slide deck.

---

## 6. Immediate Next Steps

1. Update `src/data_pipeline/api_client.py` to target the confirmed RailRadar `/trains/{train_no}` and `/trains/{train_no}/live` endpoints.
2. Build `src/data_pipeline/parser.py` to extract route stops, GPS coordinates, and live delays into structured DataFrames.
3. Verify test collection across 5 representative Indian Railway corridors.
