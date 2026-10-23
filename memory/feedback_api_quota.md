---
name: api-call-constraint
description: Strictly prohibit automatic or unauthorized API calls to RailRadar due to strict request limits (quota conservation)
metadata:
  type: feedback
---

**Rule:** NEVER make automated, test, or routine API calls to RailRadar. 
**Why:** The API has strict request quotas (~100 requests already consumed) and we must preserve remaining requests for essential validation and final demonstration.
**How to apply:** Whenever working on data pipelines, models, or backend services, use local cached JSON files in `data/raw/` or Parquet datasets in `data/processed/` instead of hitting the live API. Only make a live API call if the user explicitly asks for a live test.
