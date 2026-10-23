"""
Simple Data Collector - Test Version
=====================================
Fetch data from RailRadar API and save to JSON files
"""

import os
import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List
import requests
from dotenv import load_dotenv

# Load environment
load_dotenv()


class SimpleRailwayCollector:
    """Simple collector without heavy dependencies - for testing"""

    def __init__(self):
        self.api_key = os.getenv("RAILWAY_API_KEY")
        self.base_url = os.getenv("RAILWAY_API_BASE_URL", "https://railradar.in/api")

        if not self.api_key:
            print("[WARN] RAILWAY_API_KEY not set in .env file")
            print("Please create .env file with your RailRadar API key")
            return

        self.output_dir = Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[OK] Collector initialized")
        print(f"[DATA] Output directory: {self.output_dir}")
        print(f"[KEY] API Key: {self.api_key[:10]}...")

    def fetch_train_schedule(self, train_no: str) -> Dict[str, Any]:
        """Fetch train schedule"""
        url = f"{self.base_url}/trains/{train_no}/schedule"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        print(f"\n[TRAIN] Fetching schedule for train {train_no}...")
        print(f"        URL: {url}")

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Save raw response
            filename = f"schedule_{train_no}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.output_dir / filename

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"   ✅ Success! Saved to {filename}")
            return data

        except requests.exceptions.RequestException as e:
            print(f"        [ERROR] {e}")
            return {"error": str(e)}

    def fetch_live_status(self, train_no: str, journey_date: str = None) -> Dict[str, Any]:
        """Fetch live train status"""
        if journey_date is None:
            journey_date = date.today().strftime("%Y-%m-%d")

        url = f"{self.base_url}/trains/{train_no}/live"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"date": journey_date}

        print(f"\n[LIVE] Fetching live status for train {train_no} on {journey_date}...")
        print(f"       URL: {url}")

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Save raw response
            filename = f"status_{train_no}_{journey_date.replace('-', '')}_{datetime.now().strftime('%H%M%S')}.json"
            filepath = self.output_dir / filename

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"   ✅ Success! Saved to {filename}")
            return data

        except requests.exceptions.RequestException as e:
            print(f"        [ERROR] {e}")
            return {"error": str(e)}

    def collect_multiple_trains(self, train_numbers: List[str]):
        """Collect data for multiple trains"""
        print(f"\n[START] Starting collection for {len(train_numbers)} trains...")
        print("=" * 60)

        results = {
            "schedules": [],
            "statuses": [],
            "errors": []
        }

        for i, train_no in enumerate(train_numbers, 1):
            print(f"\n[{i}/{len(train_numbers)}] Processing train {train_no}")

            # Fetch schedule
            schedule = self.fetch_train_schedule(train_no)
            if "error" not in schedule:
                results["schedules"].append(train_no)
            else:
                results["errors"].append({"train": train_no, "type": "schedule", "error": schedule["error"]})

            time.sleep(2)  # Rate limiting

            # Fetch live status
            status = self.fetch_live_status(train_no)
            if "error" not in status:
                results["statuses"].append(train_no)
            else:
                results["errors"].append({"train": train_no, "type": "status", "error": status["error"]})

            time.sleep(2)  # Rate limiting

        # Print summary
        print("\n" + "=" * 60)
        print("[SUMMARY] COLLECTION SUMMARY")
        print("=" * 60)
        print(f"[OK] Schedules fetched: {len(results['schedules'])}")
        print(f"[OK] Live statuses fetched: {len(results['statuses'])}")
        print(f"[FAIL] Errors: {len(results['errors'])}")

        if results['errors']:
            print("\n[ERRORS] Errors encountered:")
            for err in results['errors']:
                print(f"         - Train {err['train']} ({err['type']}): {err['error']}")

        # Save summary
        summary_file = self.output_dir / f"collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n[SAVE] Summary saved to {summary_file.name}")
        print(f"[SAVE] All raw responses saved in {self.output_dir}")

        return results


# Popular trains to test with
POPULAR_TRAINS = [
    "12302",  # Rajdhani Express (NDLS-HWH)
    "12301",  # Rajdhani Express (HWH-NDLS)
    "12951",  # Mumbai Rajdhani
    "12309",  # Patna Rajdhani
    "12423",  # Rajdhani Dibrugarh
    "12429",  # Lucknow Rajdhani
    "12001",  # Shatabdi Express
    "12951",  # Mumbai Central Rajdhani
    "12553",  # Vaishali Express
    "12801"   # Purushottam Express
]


def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("RailSarthi AI - Data Collection Test")
    print("=" * 60)

    collector = SimpleRailwayCollector()

    if not collector.api_key:
        print("\n[ERROR] Cannot proceed without API key")
        print("\nSteps to fix:")
        print("1. Copy .env.example to .env")
        print("2. Add your RailRadar API key to .env")
        print("3. Run this script again")
        return

    # Test with a few trains first
    test_trains = POPULAR_TRAINS[:3]  # Start with 3 trains

    print(f"\n[TEST] Testing with {len(test_trains)} trains: {', '.join(test_trains)}")
    print("[INFO] Starting automatic collection...")

    results = collector.collect_multiple_trains(test_trains)

    print("\n[DONE] Collection complete!")
    print("\nNext steps:")
    print("1. Check data/raw/ folder for collected JSON files")
    print("2. Inspect the responses to understand API structure")
    print("3. Run data validation and processing scripts")


if __name__ == "__main__":
    main()
