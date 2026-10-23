# AI System Design Learning Plan for RailSarthi AI
**Integrated with 4-Week Project Timeline | Oct 17 - Nov 13, 2026**

## Overview

This learning plan teaches AI System Design concepts through hands-on implementation in RailSarthi AI. Each week focuses on specific system design principles that you'll apply directly to your project, making you interview-ready while building your portfolio.

---

## Week 1: Data Pipeline & ML System Fundamentals (Oct 17-23)

### **Core Concepts to Learn**

#### 1. **Data Pipeline Architecture**
**Theory:**
- Batch vs streaming data ingestion
- ETL (Extract, Transform, Load) vs ELT patterns
- Data validation and quality checks
- Schema evolution and versioning
- Cold start problem and data bootstrapping

**Interview Questions You'll Be Ready For:**
- "Design a data pipeline for collecting real-time train status data"
- "How would you handle missing or corrupted API responses?"
- "What's your strategy for the cold start problem when you have no historical data?"

**Applied in RailSarthi (Days 1-7):**
- ✅ **Day 1-2**: Design Railway API collector with retry logic, rate limiting, timeout handling
- ✅ **Day 3-4**: Build Pydantic validation schemas for data quality
- ✅ **Day 5**: Implement deduplication and outlier detection
- ✅ **Day 6-7**: Create Parquet storage for efficient ML data access

**Hands-On Learning:**
```python
# You'll implement these patterns:

# 1. Retry with exponential backoff
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_train_status(train_no: str) -> dict:
    response = requests.get(f"{API_URL}/status/{train_no}")
    response.raise_for_status()
    return response.json()

# 2. Circuit breaker pattern
class APICircuitBreaker:
    def __init__(self, failure_threshold: int = 5):
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call_api(self):
        if self.state == "OPEN":
            raise Exception("Circuit breaker is OPEN")
        # Make API call...

# 3. Data validation with Pydantic
class TrainStatusResponse(BaseModel):
    train_no: str
    delay_minutes: int = Field(ge=0, le=1440)  # 0-24 hours
    timestamp: datetime
    
    @validator('train_no')
    def validate_train_no(cls, v):
        if not re.match(r'^\d{5}$', v):
            raise ValueError('Invalid train number format')
        return v
```

**Resources:**
- Read: "Designing Data-Intensive Applications" Chapter 3 (Storage and Retrieval)
- Watch: "System Design Interview: Design a Rate Limiter" (YouTube)
- Practice: Implement your Railway API collector with all error handling

---

#### 2. **ML System Design Fundamentals**

**Theory:**
- Online vs offline prediction
- Feature store architecture
- Model serving patterns (embedded, service, sidecar)
- Model versioning and AB testing
- Training vs inference pipeline separation

**Interview Questions You'll Be Ready For:**
- "How would you serve ML predictions in a production system?"
- "Explain the difference between online and offline feature computation"
- "Design a system to A/B test two delay prediction models"

**Applied in RailSarthi (Days 1-7):**
- ✅ **Day 6**: Separate training pipeline (notebooks) from inference (FastAPI)
- ✅ **Day 7**: Design feature engineering that works for both training and serving
- ✅ **Day 7**: Plan model versioning (v1.0 with synthetic data, v2.0 with real data)

**Key Patterns You'll Implement:**
```python
# Feature engineering consistency
class FeatureEngineering:
    """Same feature logic for training and serving"""
    
    @staticmethod
    def extract_temporal_features(journey_date: date) -> dict:
        return {
            'day_of_week': journey_date.weekday(),
            'is_weekend': journey_date.weekday() >= 5,
            'month': journey_date.month,
            'is_festival': is_festival_date(journey_date)
        }
    
    @staticmethod
    def compute_features(train_no: str, journey_date: date, 
                        historical_data: pd.DataFrame) -> dict:
        # Used in BOTH training notebook AND FastAPI endpoint
        temporal = FeatureEngineering.extract_temporal_features(journey_date)
        historical = historical_data[historical_data['train_no'] == train_no]
        
        return {
            **temporal,
            'historical_avg_delay': historical['delay_minutes'].mean(),
            'historical_median_delay': historical['delay_minutes'].median(),
            'historical_std_delay': historical['delay_minutes'].std()
        }
```

**Resources:**
- Read: "Machine Learning Systems Design" (Chip Huyen) - Chapter 4
- Watch: "ML System Design: Instagram Recommendation System"
- Practice: Draw architecture diagram showing training vs serving pipeline

---

### **Week 1 Deliverables for Interview Prep**

By end of Week 1, you should be able to:

1. **Whiteboard Exercise**: Design a data pipeline for collecting and validating API data
   - Show retry logic, rate limiting, caching
   - Explain trade-offs between pull vs push models
   - Discuss data quality checks

2. **Code Review**: Explain your Railway API collector implementation
   - Walk through error handling patterns
   - Justify Pydantic schema choices
   - Discuss scalability (what breaks at 1000 trains?)

3. **Architecture Diagram**: Draw RailSarthi's data flow
   - API → Validation → Storage → Feature Engineering → Model

**Practice Interview Questions:**
- "Your Railway API has rate limits of 100 requests/day. How do you design data collection for 50 trains with hourly updates?"
- "The API returns data in format X, but your model expects format Y. Where does transformation happen?"
- "How do you handle time zones when collecting train status from multiple regions?"

---

## Week 2: Backend API Design & ML Model Serving (Oct 24-30)

### **Core Concepts to Learn**

#### 3. **RESTful API Design for ML Systems**

**Theory:**
- API design principles (REST, versioning, pagination)
- Request/response schema design
- Caching strategies (Redis, in-memory, CDN)
- Rate limiting and authentication
- Async vs sync endpoints
- Error handling and status codes

**Interview Questions You'll Be Ready For:**
- "Design a REST API for a train delay prediction service"
- "How would you handle 1000 concurrent prediction requests?"
- "What caching strategy would you use for live train status?"

**Applied in RailSarthi (Days 8-14):**
- ✅ **Day 8-9**: Design FastAPI endpoints with proper schemas
- ✅ **Day 10-11**: Implement in-memory caching (5-min TTL for live status)
- ✅ **Day 12-13**: Add request validation and error responses
- ✅ **Day 14**: Set up SQLite for storing predictions and user queries

**Key API Design Patterns:**
```python
# 1. Versioned API with proper schemas
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from datetime import date

app = FastAPI(title="RailSarthi API", version="1.0.0")

class DelayPredictionRequest(BaseModel):
    train_no: str = Field(..., pattern=r'^\d{5}$', example="12302")
    journey_date: date = Field(..., example="2026-11-05")
    source_station: str = Field(..., min_length=3, max_length=10)
    dest_station: str = Field(..., min_length=3, max_length=10)

class DelayPredictionResponse(BaseModel):
    train_no: str
    journey_date: date
    predicted_delay_minutes: float
    confidence_interval: tuple[float, float]
    model_version: str
    features_used: dict
    prediction_timestamp: datetime

@app.post("/api/v1/predict/delay", response_model=DelayPredictionResponse)
async def predict_delay(
    request: DelayPredictionRequest,
    db: Session = Depends(get_db)
):
    # Load model, compute features, predict, cache result
    pass

# 2. Caching with TTL
from functools import lru_cache
from cachetools import TTLCache
from datetime import datetime, timedelta

live_status_cache = TTLCache(maxsize=100, ttl=300)  # 5 min TTL

@app.get("/api/v1/trains/{train_no}/status")
async def get_live_status(train_no: str):
    cache_key = f"status_{train_no}_{datetime.now().strftime('%Y%m%d%H%M')}"
    
    if cache_key in live_status_cache:
        return {"source": "cache", "data": live_status_cache[cache_key]}
    
    # Call Railway API
    status = await fetch_from_railway_api(train_no)
    live_status_cache[cache_key] = status
    
    return {"source": "api", "data": status}

# 3. Rate limiting
from fastapi_limiter.depends import RateLimiter

@app.get("/api/v1/predict/delay", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def predict_delay():
    # Max 10 requests per minute per IP
    pass
```

**Resources:**
- Read: "Web API Design: The Missing Link" (REST best practices)
- Watch: "System Design: Design TinyURL" (caching strategies)
- Practice: Design API for multi-model serving (delay + crowd predictions)

---

#### 4. **Model Serving & Inference Optimization**

**Theory:**
- Model loading strategies (lazy vs eager)
- Batch prediction vs real-time
- GPU vs CPU inference
- Model compression (quantization, pruning)
- Feature preprocessing pipeline
- Cold start latency

**Interview Questions You'll Be Ready For:**
- "Your XGBoost model takes 200ms per prediction. How do you optimize it?"
- "How would you serve models to handle 1000 QPS?"
- "Explain trade-offs between batch and real-time prediction"

**Applied in RailSarthi (Days 8-14):**
- ✅ **Day 8**: Load model once at startup (not per request)
- ✅ **Day 9**: Profile prediction latency with sample data
- ✅ **Day 10**: Implement batch prediction endpoint for multiple trains
- ✅ **Day 14**: Cache predictions in SQLite with expiry timestamp

**Optimization Patterns:**
```python
# 1. Lazy model loading at startup
import joblib
from contextlib import asynccontextmanager

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models at startup
    models['delay_v2'] = joblib.load("models/delay_model_v2.joblib")
    models['delay_features'] = joblib.load("models/delay_feature_cols.joblib")
    print(f"Loaded models: {list(models.keys())}")
    yield
    # Cleanup on shutdown
    models.clear()

app = FastAPI(lifespan=lifespan)

# 2. Feature computation caching
@lru_cache(maxsize=1000)
def get_historical_features(train_no: str) -> dict:
    """Cache historical aggregates (rarely change)"""
    # Query database once, cache result
    return compute_historical_stats(train_no)

# 3. Batch prediction endpoint
@app.post("/api/v1/predict/delay/batch")
async def predict_delay_batch(trains: List[DelayPredictionRequest]):
    # Compute features for all trains
    features_batch = [compute_features(req) for req in trains]
    features_df = pd.DataFrame(features_batch)
    
    # Single model.predict() call (faster than N individual calls)
    predictions = models['delay_v2'].predict(features_df)
    
    return [
        {"train_no": req.train_no, "predicted_delay": pred}
        for req, pred in zip(trains, predictions)
    ]
```

**Resources:**
- Read: "Serving ML Models in Production" (mlops.community)
- Watch: "Optimizing TensorFlow Models for Production"
- Practice: Profile your model's prediction time with `cProfile`

---

### **Week 2 Deliverables for Interview Prep**

By end of Week 2, you should be able to:

1. **API Design Exercise**: Design endpoints for a multi-model ML service
   - Show versioning, caching, rate limiting
   - Explain sync vs async trade-offs
   - Discuss error handling for model failures

2. **Latency Analysis**: Explain your prediction pipeline's bottlenecks
   - Measure: feature computation time, model inference time, DB query time
   - Identify: what's the slowest part?
   - Optimize: how would you speed it up 10x?

3. **Scalability Discussion**: "What happens at 100x load?"
   - Current: 10 requests/min, 1 server
   - Future: 1000 requests/min, need horizontal scaling
   - How would you redesign?

**Practice Interview Questions:**
- "Your prediction API has p99 latency of 500ms. Users need <100ms. How do you optimize?"
- "Design a caching strategy for a system with 1000 trains × 24 hours = 24K predictions/day"
- "Your model file is 500MB. How do you deploy to a lambda with 250MB limit?"

---

## Week 3: Agentic AI & LLM System Design (Oct 31-Nov 6)

### **Core Concepts to Learn**

#### 5. **LLM-Powered Systems Architecture**

**Theory:**
- Prompt engineering and prompt chaining
- RAG (Retrieval Augmented Generation) architecture
- Tool calling and function execution
- Agentic workflows (ReAct, Plan-and-Execute)
- LLM observability and debugging
- Cost optimization (caching, prompt compression)

**Interview Questions You'll Be Ready For:**
- "Design a chatbot that answers questions using external APIs"
- "How would you implement RAG for a knowledge base?"
- "Explain the architecture of an autonomous agent system"

**Applied in RailSarthi (Days 15-21):**
- ✅ **Day 15-16**: Retrain delay model (not agent work, but needed)
- ✅ **Day 17-18**: Build LangGraph agent with 3 tools
- ✅ **Day 19-20**: Implement tool calling with structured outputs
- ✅ **Day 21**: Add explainability (cite data sources in responses)

**Agentic Patterns You'll Implement:**
```python
# 1. LangGraph Agent Architecture
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain.tools import tool

# Define state
class AgentState(TypedDict):
    messages: List[dict]
    tool_results: dict
    final_answer: str

# Define tools
@tool
def get_live_status(train_no: str, date: str) -> dict:
    """Get current running status of a train."""
    response = requests.get(f"{FASTAPI_URL}/trains/{train_no}/status")
    return response.json()

@tool
def predict_delay(train_no: str, date: str) -> dict:
    """Predict delay for a train journey."""
    response = requests.post(f"{FASTAPI_URL}/predict/delay", json={
        "train_no": train_no,
        "journey_date": date
    })
    return response.json()

# Build graph
workflow = StateGraph(AgentState)

def router_node(state: AgentState) -> str:
    """Classify user intent and decide which tool to call."""
    last_message = state['messages'][-1]['content']
    
    # Simple intent classification (in real system, use LLM)
    if "live status" in last_message.lower():
        return "get_live_status"
    elif "delay" in last_message.lower() or "late" in last_message.lower():
        return "predict_delay"
    else:
        return "general_response"

def tool_executor_node(state: AgentState, tool_name: str) -> AgentState:
    """Execute the selected tool and update state."""
    tool_result = tools[tool_name].invoke(state['messages'][-1])
    state['tool_results'][tool_name] = tool_result
    return state

def explainer_node(state: AgentState) -> AgentState:
    """Generate final response citing tool results."""
    prompt = f"""
    User asked: {state['messages'][-1]['content']}
    Tool results: {json.dumps(state['tool_results'], indent=2)}
    
    Generate a helpful response citing the data sources.
    """
    
    response = llm.invoke(prompt)
    state['final_answer'] = response
    return state

workflow.add_node("router", router_node)
workflow.add_node("executor", tool_executor_node)
workflow.add_node("explainer", explainer_node)

workflow.set_entry_point("router")
workflow.add_edge("router", "executor")
workflow.add_edge("executor", "explainer")
workflow.add_edge("explainer", END)

agent = workflow.compile()

# 2. RAG Pattern (if you have time for ChromaDB)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

# Index railway knowledge base
docs = [
    "Indian Railways has 17 zones including Northern Railway, Western Railway...",
    "Rajdhani Express trains are premium AC trains connecting major cities...",
    "Fog in North India (Dec-Feb) causes significant delays..."
]

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.create_documents(docs)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

def search_knowledge(query: str) -> str:
    """RAG tool for agent."""
    relevant_docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in relevant_docs])
```

**Resources:**
- Read: "Building LLM Applications for Production" (Huyen)
- Watch: "LangChain & LangGraph Deep Dive" (YouTube)
- Read: "Anthropic Prompt Engineering Guide"
- Practice: Build a simple agent that calls 2 functions

---

#### 6. **Agent Design Patterns & Reliability**

**Theory:**
- Tool selection and execution
- Error recovery and fallbacks
- Multi-agent orchestration
- Human-in-the-loop patterns
- Agent observability (tracing, logging)
- Preventing hallucinations (grounding, citations)

**Interview Questions You'll Be Ready For:**
- "How would you prevent an LLM agent from hallucinating facts?"
- "Design an error recovery mechanism for failed tool calls"
- "How do you debug when an agent selects the wrong tool?"

**Applied in RailSarthi (Days 17-21):**
- ✅ **Day 17-18**: Implement structured tool calling (Pydantic schemas)
- ✅ **Day 19**: Add error handling (retry tool call, fallback to general response)
- ✅ **Day 20**: Add citations to agent responses ("Based on 45 historical journeys...")
- ✅ **Day 21**: Test edge cases (tool failures, ambiguous queries)

**Agent Reliability Patterns:**
```python
# 1. Structured tool calling with validation
from pydantic import BaseModel, ValidationError

class GetLiveStatusInput(BaseModel):
    train_no: str = Field(pattern=r'^\d{5}$')
    date: str = Field(pattern=r'^\d{4}-\d{2}-\d{2}$')

@tool(args_schema=GetLiveStatusInput)
def get_live_status(train_no: str, date: str) -> dict:
    """Tool with validated inputs - prevents malformed calls."""
    # LLM must provide correctly formatted inputs
    pass

# 2. Error recovery
def execute_tool_with_retry(tool_name: str, inputs: dict, max_retries: int = 2):
    for attempt in range(max_retries):
        try:
            return tools[tool_name].invoke(inputs)
        except ValidationError as e:
            if attempt == max_retries - 1:
                return {"error": f"Invalid inputs: {e}"}
            # Ask LLM to fix inputs
            corrected = llm.invoke(f"Fix these inputs: {inputs}. Error: {e}")
            inputs = corrected
        except APIError as e:
            if attempt == max_retries - 1:
                return {"error": f"API failed: {e}"}
            time.sleep(2 ** attempt)  # Exponential backoff

# 3. Citation and grounding
def generate_response_with_citations(tool_results: dict, user_query: str) -> str:
    prompt = f"""
    User query: {user_query}
    
    Data retrieved:
    {json.dumps(tool_results, indent=2)}
    
    Generate a response that:
    1. Directly answers the user's question
    2. Cites specific data points (e.g., "Based on 45 historical journeys...")
    3. Acknowledges uncertainty if data is insufficient
    4. Never invents information not present in tool results
    """
    
    response = llm.invoke(prompt)
    return response
```

**Resources:**
- Read: "ReAct: Synergizing Reasoning and Acting in Language Models" (paper)
- Watch: "Building Reliable LLM Applications" (Harrison Chase)
- Practice: Add tracing to your agent with LangSmith

---

### **Week 3 Deliverables for Interview Prep**

By end of Week 3, you should be able to:

1. **Agent Architecture Whiteboard**: Draw your LangGraph agent's flow
   - Show: Router → Tool Executor → Explainer
   - Explain: Why this architecture over simple prompt chaining?
   - Discuss: How would you add more tools?

2. **RAG vs Fine-tuning**: Explain when to use each
   - RAG: Dynamic knowledge that changes (train schedules)
   - Fine-tuning: Specific domain language or style
   - Hybrid: Fine-tuned model + RAG for facts

3. **Agent Debugging**: "Your agent predicts sunny weather when asked about train delays. How do you fix it?"
   - Step through: intent classification, tool selection, response generation
   - Identify: wrong tool called or hallucination in final response?
   - Fix: add structured outputs, improve prompts, add validation

**Practice Interview Questions:**
- "Design an AI agent that books train tickets. What tools does it need?"
- "Your agent costs $0.50 per query. How do you reduce costs by 10x?"
- "How would you test an LLM agent's reliability before production?"

---

## Week 4: Production ML & System Integration (Nov 7-13)

### **Core Concepts to Learn**

#### 7. **ML System Deployment & Monitoring**

**Theory:**
- Deployment patterns (blue-green, canary, shadow)
- Model monitoring (drift detection, performance degradation)
- Observability (metrics, logs, traces)
- Feature flags for models
- Rollback strategies
- A/B testing framework

**Interview Questions You'll Be Ready For:**
- "How would you deploy a new model version without downtime?"
- "Design a monitoring system for ML models in production"
- "How do you detect when a model's performance degrades?"

**Applied in RailSarthi (Days 22-28):**
- ✅ **Day 26**: Deploy backend to Render (understand cloud deployment)
- ✅ **Day 27**: Deploy frontend to Vercel (understand CI/CD basics)
- ✅ **Day 28**: Add basic logging (request counts, error rates, prediction latency)

**Deployment Patterns:**
```python
# 1. Model versioning and feature flags
class ModelRegistry:
    def __init__(self):
        self.models = {
            'delay_v1': joblib.load('models/delay_v1.joblib'),  # Synthetic data
            'delay_v2': joblib.load('models/delay_v2.joblib')   # Real data
        }
        self.active_model = 'delay_v1'  # Feature flag
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        model = self.models[self.active_model]
        return model.predict(features)
    
    def switch_model(self, version: str):
        """Hot-swap models without restart"""
        if version in self.models:
            self.active_model = version
            logging.info(f"Switched to {version}")

# 2. Request logging for monitoring
import structlog

logger = structlog.get_logger()

@app.post("/api/v1/predict/delay")
async def predict_delay(request: DelayPredictionRequest):
    start_time = time.time()
    
    try:
        prediction = model_registry.predict(features)
        latency = time.time() - start_time
        
        logger.info(
            "prediction_success",
            train_no=request.train_no,
            predicted_delay=prediction,
            latency_ms=latency * 1000,
            model_version=model_registry.active_model
        )
        
        return {"predicted_delay": prediction, "model_version": model_registry.active_model}
    
    except Exception as e:
        logger.error("prediction_failed", train_no=request.train_no, error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed")

# 3. Simple drift detection
class DriftMonitor:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_mean = reference_data.mean()
        self.reference_std = reference_data.std()
    
    def check_drift(self, current_data: pd.DataFrame) -> bool:
        """Simple statistical drift detection"""
        current_mean = current_data.mean()
        z_score = (current_mean - self.reference_mean) / self.reference_std
        
        # Alert if feature distribution shifts >2 std devs
        if any(abs(z_score) > 2):
            logger.warning("feature_drift_detected", z_scores=z_score.to_dict())
            return True
        return False
```

**Resources:**
- Read: "Building Machine Learning Powered Applications" (Chapter 8)
- Watch: "MLOps: Continuous Delivery for ML"
- Practice: Set up GitHub Actions for automated model testing

---

#### 8. **End-to-End ML System Integration**

**Theory:**
- Frontend ↔ Backend ↔ ML integration patterns
- WebSocket vs polling for real-time updates
- State management in ML applications
- Error propagation across layers
- User experience for ML uncertainty
- System resilience (graceful degradation)

**Interview Questions You'll Be Ready For:**
- "Design the full stack for a real-time ML application"
- "How would you show prediction uncertainty in the UI?"
- "Your ML API is down. How does the frontend handle it?"

**Applied in RailSarthi (Days 22-28):**
- ✅ **Day 22-25**: Build simple frontend (Streamlit or React)
- ✅ **Day 26**: Integration testing (frontend → FastAPI → model)
- ✅ **Day 27**: Add error handling (show "Prediction unavailable" gracefully)
- ✅ **Day 28**: Add loading states and confidence indicators in UI

**Integration Patterns:**
```python
# 1. Backend graceful degradation
@app.get("/api/v1/trains/{train_no}/status")
async def get_train_status(train_no: str):
    try:
        # Try Railway API first
        status = await railway_api.get_status(train_no)
        return {"status": status, "source": "api", "freshness": "real-time"}
    except APIError:
        # Fallback to cached/historical data
        cached = db.get_latest_status(train_no)
        if cached and (datetime.now() - cached.timestamp).seconds < 3600:
            return {"status": cached.data, "source": "cache", "freshness": "1 hour old"}
        else:
            return {"error": "Status unavailable", "train_no": train_no}

# 2. Frontend error handling (React example)
function TrainStatus({ trainNo }) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    fetch(`/api/v1/trains/${trainNo}/status`)
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          setError("Live status temporarily unavailable");
        } else {
          setStatus(data);
        }
        setLoading(false);
      })
      .catch(err => {
        setError("Unable to fetch status. Please try again.");
        setLoading(false);
      });
  }, [trainNo]);
  
  if (loading) return <Spinner />;
  if (error) return <Alert type="warning">{error}</Alert>;
  
  return (
    <div>
      <StatusBadge status={status.status} />
      <Freshness source={status.source} timestamp={status.freshness} />
    </div>
  );
}

# 3. Confidence visualization
@app.post("/api/v1/predict/delay")
async def predict_delay(request: DelayPredictionRequest):
    # Get prediction
    pred = model.predict(features)[0]
    
    # Estimate uncertainty (e.g., from ensemble or historical variance)
    uncertainty = estimate_uncertainty(features, historical_predictions)
    
    return {
        "predicted_delay_minutes": round(pred),
        "confidence_interval": [pred - uncertainty, pred + uncertainty],
        "confidence_level": "medium" if uncertainty < 10 else "low",
        "explanation": "Based on 45 similar journeys in the past month"
    }
```

**Resources:**
- Read: "Designing ML-Powered Products" (Google)
- Watch: "Communicating Uncertainty in ML Systems" (Airbnb Tech Talk)
- Practice: Add retry logic and fallback messages to your UI

---

### **Week 4 Deliverables for Interview Prep**

By end of Week 4, you should be able to:

1. **Full System Walkthrough**: Explain RailSarthi end-to-end
   - User searches for train → Frontend → FastAPI → Model → Response → UI
   - Show: where caching happens, where errors are handled, where logging occurs
   - Discuss: bottlenecks, failure modes, scaling limits

2. **Trade-offs Discussion**: Defend your technical decisions
   - "Why SQLite instead of PostgreSQL?" → Speed for MVP, easy deployment
   - "Why Streamlit instead of React?" → 2 days vs 1 week development time
   - "Why synthetic data for training?" → No free historical API, but patterns are realistic

3. **Production Readiness**: "What would you change for 1M users?"
   - SQLite → PostgreSQL/MongoDB for horizontal scaling
   - Single FastAPI server → Load balancer + multiple replicas
   - In-memory cache → Redis cluster
   - Model in server → Dedicated model serving (TorchServe, TFServing)
   - Streamlit → React SPA + CDN

**Practice Interview Questions:**
- "Walk me through your ML system's architecture from data to deployment"
- "How would you monitor model performance in production?"
- "Your system handles 10 QPS now. Design for 1000 QPS with <100ms latency"

---

## Comprehensive AI System Design Interview Framework

After 4 weeks, you should follow this framework for any AI system design interview:

### **Step 1: Requirements Clarification (5 minutes)**
- Functional requirements: What does the system do?
- Non-functional requirements: Scale, latency, accuracy
- Constraints: Budget, timeline, existing infrastructure

Example:
```
"Design a train delay prediction system"

Questions to ask:
- How many trains/routes to cover?
- Prediction horizon? (1 hour, 1 day, 1 week ahead)
- Required latency? (<100ms for real-time, <1s acceptable for batch)
- Accuracy target? (MAE < 10 minutes acceptable?)
- Budget? (free tier, $100/month, unlimited)
- Existing data? (historical logs, real-time feeds)
```

### **Step 2: High-Level Architecture (10 minutes)**
Draw boxes and arrows showing:
1. **Data Sources** → APIs, databases, streams
2. **Data Pipeline** → Collection, validation, storage
3. **Feature Engineering** → Offline + online features
4. **Model Training** → Notebooks, experiments, versioning
5. **Model Serving** → API, batch, embedded
6. **Application Layer** → Backend API, agent, frontend
7. **Monitoring** → Logs, metrics, alerts

**RailSarthi Example:**
```
[Railway API] → [Collector + Validator] → [Parquet + SQLite]
                                              ↓
                                    [Feature Engineering]
                                              ↓
                                    [XGBoost Training] → [Model v2.0]
                                              ↓
[User] → [React UI] → [FastAPI] → [Model Serving] → [Response]
                           ↓
                    [LangGraph Agent] → [Tools: predict, status, schedule]
                           ↓
                      [Gemini LLM]
```

### **Step 3: Deep Dive into Key Components (15-20 minutes)**

Pick 2-3 components based on interviewer's interest:

**Data Pipeline:**
- How do you handle API rate limits?
- What's your retry strategy?
- How do you validate data quality?

**ML Model:**
- Why XGBoost over LSTM?
- How do you handle the cold start problem?
- What's your feature engineering strategy?

**Model Serving:**
- How do you optimize inference latency?
- What's your caching strategy?
- How do you version models?

**Agent System:**
- How does tool selection work?
- How do you prevent hallucinations?
- What's your error recovery mechanism?

### **Step 4: Discuss Trade-offs (5-10 minutes)**

Always show you understand trade-offs:

| Decision | Pro | Con | When to Choose |
|----------|-----|-----|----------------|
| **Batch vs Real-time Prediction** | Batch: efficient, cheaper | Batch: stale predictions | Batch if delay tolerance >1 hour |
| **Online vs Offline Features** | Online: fresh data | Online: higher latency | Online for live status, offline for historical |
| **SQL vs NoSQL** | SQL: transactions, joins | NoSQL: horizontal scaling | SQL for <10M records, NoSQL for >100M |
| **Monolith vs Microservices** | Monolith: simple deployment | Microservices: independent scaling | Monolith for MVP, microservices at scale |
| **Synthetic vs Real Data** | Synthetic: fast start, no privacy | Real: actual patterns | Synthetic for MVP, real for production |

### **Step 5: Scalability & Bottlenecks (5 minutes)**

"What breaks at 100x scale?"

**RailSarthi Example:**
- Current: 10 trains, 10 requests/min, 1 server
- At 100x: 1000 trains, 1000 requests/min, need horizontal scaling

**Bottlenecks:**
1. **Railway API rate limits** → Cache aggressively, add fallback data sources
2. **Model inference latency** → Batch predictions, use GPU, cache common queries
3. **SQLite concurrent writes** → Switch to PostgreSQL with connection pooling
4. **Single FastAPI server** → Add load balancer + multiple replicas
5. **In-memory cache** → Switch to Redis cluster

### **Step 6: Monitoring & Reliability (3-5 minutes)**

"How do you know if the system is working?"

**Metrics to Track:**
- **Data Quality**: API success rate, missing value %, outlier count
- **Model Performance**: MAE, RMSE, prediction latency, drift score
- **API Health**: p50/p95/p99 latency, error rate, throughput
- **Business Metrics**: predictions served, user satisfaction, accuracy vs baseline

**Alerting Rules:**
- API error rate > 5% for 5 minutes → Page on-call
- Prediction latency p95 > 500ms → Investigate
- Feature drift detected → Retrain model
- Model MAE increases by 50% → Rollback to previous version

---

## Key Interview Patterns to Master

### **1. The Layered Architecture Pattern**
```
Presentation Layer (UI)
    ↓
Application Layer (API)
    ↓
Service Layer (Business Logic)
    ↓
Data Layer (Database, Cache)
    ↓
External Services (APIs)
```

### **2. The Feature Store Pattern**
```
Offline Features (Historical)
    → Parquet files / Data Warehouse
    → Used in training

Online Features (Real-time)
    → Redis / DynamoDB
    → Used in serving

Shared Feature Logic
    → Same code for both
    → Prevents train-serve skew
```

### **3. The Model Serving Pattern**
```
Option A: Embedded (model in API server)
  ✅ Simple, low latency
  ❌ Tight coupling, harder to update

Option B: Model as Service (separate model server)
  ✅ Independent scaling, A/B testing
  ❌ Network latency, more complexity

Option C: Batch Prediction (precompute results)
  ✅ Very fast serving, high throughput
  ❌ Stale predictions, requires storage
```

### **4. The Agent Pattern**
```
User Input
    ↓
Intent Classification (Router)
    ↓
Tool Selection (Planner)
    ↓
Tool Execution (Executor)
    ↓
Response Generation (Explainer)
    ↓
Output to User
```

---

## Resources by Week

### **Week 1: Data & ML Systems**
**Books:**
- "Designing Data-Intensive Applications" (Kleppmann) - Chapters 1-3
- "Machine Learning Systems Design" (Chip Huyen) - Chapters 1-4

**Videos:**
- "System Design Interview: API Rate Limiter" (Exponent)
- "ML System Design: Feed Ranking" (Facebook Engineering)

**Practice:**
- LeetCode System Design: Design TinyURL, Design Rate Limiter
- Draw RailSarthi's data pipeline architecture

### **Week 2: APIs & Model Serving**
**Books:**
- "Building ML Powered Applications" (Ameisen) - Chapters 5-7

**Videos:**
- "FastAPI Beyond CRUD" (TestDriven.io)
- "Optimizing ML Inference" (Google Cloud)

**Practice:**
- Design API for multi-model serving
- Benchmark your model's inference time

### **Week 3: LLMs & Agents**
**Papers:**
- "ReAct: Synergizing Reasoning and Acting" (Yao et al.)
- "Retrieval-Augmented Generation" (Lewis et al.)

**Videos:**
- "LangGraph Tutorial" (LangChain)
- "Building Production LLM Apps" (Anthropic)

**Practice:**
- Build 3-tool agent from scratch
- Design RAG system for railway knowledge

### **Week 4: Production & Integration**
**Books:**
- "Reliable Machine Learning" (Cheng et al.)

**Videos:**
- "MLOps Maturity Model" (Microsoft)
- "Deploying ML Models" (AWS)

**Practice:**
- Deploy RailSarthi end-to-end
- Write production monitoring plan

---

## Interview Cheat Sheet

### **Top 20 AI System Design Questions**

1. Design a recommendation system (Netflix, Instagram feed)
2. Design a search engine with ML ranking
3. Design an ad click prediction system
4. Design a fraud detection system
5. Design a chatbot with external tools
6. Design a real-time translation system
7. Design an image classification API
8. Design a personalized email system
9. Design a demand forecasting system
10. Design a document similarity service
11. Design a spam detection system
12. Design a question-answering system with RAG
13. Design a voice assistant (Alexa, Siri)
14. Design a video recommendation system (TikTok)
15. Design a price prediction system
16. Design an autonomous agent (train booking)
17. Design a content moderation system
18. Design a collaborative filtering system
19. Design a time-series forecasting API
20. Design a multi-armed bandit system

**RailSarthi Maps To:**
- #11: Spam detection (classification problem)
- #15: Price prediction (time-series forecasting)
- #16: Autonomous agent (tool-calling agent)
- #12: QA system (RAG + LLM)

### **Key Terms to Use**

**Data:**
- ETL/ELT pipeline
- Schema evolution
- Data versioning
- Cold start problem
- Feature store
- Train-serve skew

**ML:**
- Online vs offline learning
- Batch vs real-time inference
- Model versioning
- A/B testing
- Shadow deployment
- Drift detection

**Systems:**
- Horizontal vs vertical scaling
- Caching (Redis, CDN)
- Load balancing
- Circuit breaker
- Rate limiting
- Graceful degradation

**LLMs:**
- Prompt engineering
- RAG (Retrieval Augmented Generation)
- Tool calling / Function calling
- ReAct / Chain-of-Thought
- Hallucination prevention
- Context window optimization

---

## Final Week-by-Week Checklist

### **Week 1 Checklist**
- [ ] Implement Railway API collector with retry logic
- [ ] Create Pydantic validation schemas
- [ ] Set up Parquet storage
- [ ] Understand data pipeline patterns
- [ ] Practice: "Design a data collection system"

### **Week 2 Checklist**
- [ ] Build FastAPI with 3 core endpoints
- [ ] Add caching (5-min TTL for live status)
- [ ] Profile model inference latency
- [ ] Set up SQLite database
- [ ] Practice: "Design a prediction API"

### **Week 3 Checklist**
- [ ] Build LangGraph agent with 3 tools
- [ ] Implement structured tool calling
- [ ] Add error recovery for tool failures
- [ ] Add citations to agent responses
- [ ] Practice: "Design an AI agent system"

### **Week 4 Checklist**
- [ ] Deploy backend to Render
- [ ] Deploy frontend to Vercel
- [ ] Add logging and basic monitoring
- [ ] Test end-to-end user flows
- [ ] Practice: "Design full-stack ML system"

---

## Success Metrics

By Nov 13, you should be able to:

✅ **Whiteboard any ML system architecture in 15 minutes**
✅ **Explain trade-offs for 5+ design decisions**
✅ **Debug latency issues in an ML API**
✅ **Design an agent system with tool calling**
✅ **Discuss scalability from 10 QPS → 1000 QPS**
✅ **Answer "What breaks at 100x scale?" for any component**
✅ **Explain RailSarthi's architecture to a senior engineer**

---

## Example Interview Transcript

**Interviewer:** "Design a system to predict train delays for Indian Railways."

**You:** "Great! Let me start by clarifying requirements. 

*Requirements:*
- How many trains? Let's say 1000 popular trains.
- Prediction horizon? 1 day ahead.
- Latency? <500ms for API response, can be higher for batch predictions.
- Accuracy? MAE < 15 minutes compared to no-delay baseline.
- Scale? 10K predictions/day initially.

*High-level architecture:*
[Draw diagram]
- Data: Railway API → Collector → Validation → Parquet
- Features: Historical avg delay, temporal features (day/month), route features
- Model: XGBoost regressor (handles mixed features well, fast inference)
- Serving: FastAPI with 5-min cache, SQLite for predictions
- Monitoring: Track API health, model drift, prediction accuracy

*Deep dive into data pipeline:*
The key challenge is Railway APIs have rate limits. My strategy:
1. Batch collection every 30 minutes (not per request)
2. Aggressive caching with 5-min TTL
3. Fallback to historical average if API fails
4. Pydantic validation to catch bad data early

*Why XGBoost over LSTM?*
- Faster training (<5 min vs hours)
- Easier to debug (feature importance)
- Good enough for MVP (LSTM if we had 1M+ samples)

*Scaling to 100x:*
Current bottleneck is SQLite concurrent writes. At 1M predictions/day:
- Switch to PostgreSQL with connection pooling
- Add Redis for caching
- Horizontal scale API servers behind load balancer
- Batch predictions overnight, serve from cache

*Monitoring:*
- Alert if API error rate > 5%
- Track MAE weekly, retrain if it degrades >20%
- Log prediction latency p95/p99

Does this match what you were looking for?"

**Interviewer:** "Impressive! Now tell me about feature engineering..."

[Continue discussion]

---

## Next Steps for You

1. **Save this document** to your RailSarthi-AI/docs/ folder
2. **Print Week 1 section** and keep beside you while coding
3. **Set weekly goals** based on the checklists
4. **Practice whiteboarding** - draw RailSarthi's architecture on paper
5. **Join mock interviews** on Pramp or Interviewing.io (after Week 2)

By building RailSarthi while following this learning plan, you'll have:
- ✅ A deployed AI system (portfolio)
- ✅ Deep understanding of AI system design (interviews)
- ✅ Concrete examples for every interview question
- ✅ GitHub repo showing end-to-end ML engineering

Let's build this! 🚀
