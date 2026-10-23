"""
RailRadar API Client for RailSarthi AI
=======================================
Production client for RailRadar API v1 providing:
- Train route and schedule details (/trains/{train_no})
- Real-time live train tracking and delay status (/trains/{train_no}/live)
- Built-in exponential backoff retry logic, rate limiting, and circuit breaker
- Automatic raw JSON persistence for audit trail and data pipeline ingestion
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


class RateLimiter:
    """Sliding-window token bucket rate limiter to respect API quotas."""

    def __init__(self, max_calls: int = 30, time_window_seconds: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window_seconds
        self.calls = []

    def acquire(self) -> None:
        """Block if rate limit is reached until a slot opens."""
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.time_window]

        if len(self.calls) >= self.max_calls:
            oldest = min(self.calls)
            sleep_needed = self.time_window - (now - oldest) + 0.1
            if sleep_needed > 0:
                logger.warning("Rate limit approached. Throttling for %.2fs", sleep_needed)
                time.sleep(sleep_needed)
                self.calls = [t for t in self.calls if time.time() - t < self.time_window]

        self.calls.append(time.time())


class CircuitBreaker:
    """Circuit breaker pattern to fail fast during API outages."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def check_state(self) -> None:
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioned to HALF_OPEN")
            else:
                remaining = int(self.recovery_timeout - (time.time() - self.last_failure_time))
                raise RuntimeError(f"Circuit breaker is OPEN. Fast-failing request. Retry in {remaining}s")

    def record_success(self) -> None:
        if self.state != "CLOSED":
            logger.info("Circuit breaker restored to CLOSED state")
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error("Circuit breaker tripped to OPEN after %d failures", self.failure_count)


class RailRadarClient:
    """
    Robust API client for RailRadar v1.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        raw_dir: Optional[Path] = None,
        save_raw: bool = True
    ):
        self.api_key = api_key or os.getenv("RAILWAY_API_KEY")
        if not self.api_key:
            raise ValueError("RAILWAY_API_KEY must be provided or set in environment.")

        self.base_url = (base_url or os.getenv("RAILWAY_API_BASE_URL", "https://api.railradar.in/v1")).rstrip("/")
        self.save_raw = save_raw
        self.raw_dir = raw_dir or Path("data/raw")
        if self.save_raw:
            self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.rate_limiter = RateLimiter(max_calls=30, time_window_seconds=60)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self.session = self._create_session()

        logger.info("RailRadarClient initialized. Base URL: %s", self.base_url)

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.circuit_breaker.check_state()
        self.rate_limiter.acquire()

        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

        try:
            response = self.session.get(url, headers=headers, params=params, timeout=12)
            response.raise_for_status()
            data = response.json()
            self.circuit_breaker.record_success()
            return data
        except requests.RequestException as e:
            self.circuit_breaker.record_failure()
            logger.error("API request failed for %s: %s", url, e)
            raise

    def get_train_details(self, train_no: str) -> Dict[str, Any]:
        """
        Fetch train schedule, route, stop sequences, and metadata.
        Endpoint: GET /trains/{train_no}
        """
        train_no = str(train_no).strip()
        endpoint = f"/trains/{train_no}"
        payload = self._request(endpoint)

        if self.save_raw:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.raw_dir / f"train_{train_no}_{ts}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.debug("Saved raw train schedule to %s", file_path)

        return payload

    def get_live_status(self, train_no: str) -> Dict[str, Any]:
        """
        Fetch current live running status, delays, and progress.
        Endpoint: GET /trains/{train_no}/live
        """
        train_no = str(train_no).strip()
        endpoint = f"/trains/{train_no}/live"
        payload = self._request(endpoint)

        if self.save_raw:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.raw_dir / f"live_{train_no}_{ts}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.debug("Saved raw live status to %s", file_path)

        return payload


if __name__ == "__main__":
    client = RailRadarClient()
    print("Testing get_train_details(12302)...")
    res = client.get_train_details("12302")
    train = res.get("data", {}).get("train", {})
    route = res.get("data", {}).get("route", [])
    print(f"Train: {train.get('name')} ({train.get('number')})")
    print(f"Total halts: {train.get('totalHalts')}, Total route entries: {len(route)}")

    print("\nTesting get_live_status(12302)...")
    live = client.get_live_status("12302")
    live_data = live.get("data", {})
    print(f"Status: {live_data.get('status')}, Delay: {live_data.get('delayMinutes')} min")
    print("API verification completed successfully!")
