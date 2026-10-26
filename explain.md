---
name: step3-implementation-log
description: Records every edit made to implement Step 3 (Cache + Async + Validation)
metadata:
  type: project
---

STEP 3 IMPLEMENTATION LOG — RailSarthi AI FastAPI Backend
===========================================================

1. EDITED FILE: backend/main.py
   - Line 17: Added import from cachetools import TTLCache, cached
   - Line ~50 (after CORS middleware): Added live_status_cache = TTLCache(maxsize=100, ttl=300)
   - Line ~153 (get_live_status endpoint): Changed def -> async def; added cache_key check/return
   - End of endpoint: Added live_status_cache[cache_key] = result before return
   - Lines 118, 152, 268: Added Path(..., pattern=r"^\d{5}$") validation to all {train_no} endpoints
   - Lines 51-53, 59: Added get_api_client() dependency provider + DummyRailRadarClient import

2. FEATURE IMPLEMENTED: Cache-first live status (5-min TTL)
   WHY: Protects limited RailRadar API quota
   WHERE: /api/v1/trains/{train_no}/live endpoint

3. FEATURE IMPLEMENTED: Async I/O endpoint
   WHY: Prevents server blocking during external API/network delays
   WHERE: get_live_status converted to async def

4. FEATURE IMPLEMENTED: Pydantic Path validation
   WHY: Invalid 5-digit train numbers rejected automatically (HTTP 422) before ML runs
   WHERE: schedule, live, reliability endpoints

5. FEATURE IMPLEMENTED: Dependency Injection (Depends + Dummy override)
   WHY: Tests/demo can run with zero API quota using DummyRailRadarClient
   WHERE: get_api_client() provider; app.dependency_overrides mechanism documented

6. NO LIVE API CALLS MADE during Step 3 implementation (quota preserved)
7. Committed to GitHub: github.com/sinu-ops/RailSarthi-AI (commit c9ce45d)
