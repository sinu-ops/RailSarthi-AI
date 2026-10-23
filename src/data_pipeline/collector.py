"""
Working Data Collector for RailRadar API
=========================================
Fetches train data using correct API structure
"""

import os
import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List
import requests
from dotenv import load_dotenv

load_dotenv()


class RailRadarCollector:
    """Data collector for RailRadar API"""

    def __init__(self):
        self.api_key = os.getenv("RAILWAY_API_KEY")
        self.base_url = "https://api.railradar.in/v1"

        if not self.api_key:
            print("[ERROR] RAILWAY_API_KEY not set in .env file")
            return

        self.output_dir = Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[OK] Collector initialized")
        print(f"[DATA] Output directory: {self.output_dir}")
        print(f"[API] Base URL: {self.base_url}")

    def fetch_train_info(self, train_no: str) -> Dict[str, Any]:
        """Fetch complete train information"""
        url = f"{self.base_url}/trains/{train_no}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        print(f"\n[FETCH] Train {train_no}...")

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            result = response.json()

            # Save raw response
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"train_{train_no}_{timestamp}.json"
            filepath = self.output_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[OK] Saved to {filename}")

            # Extract and display key info
            if result.get('success') and 'data' in result:
                data = result['data']
                train = data.get('train', {})
                print(f"     Name: {train.get('name', 'N/A')}")
                print(f"     Type: {train.get('type', 'N/A')}")

                schedule = data.get('schedule', [])
                print(f"     Stops: {len(schedule)} stations")

            return result

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] {str(e)}")
            return {"error": str(e), "train_no": train_no}

    def collect_multiple_trains(self, train_numbers: List[str]):
        """Collect data for multiple trains"""
        print(f"\n{'='*60}")
        print(f"[START] Collecting data for {len(train_numbers)} trains")
        print(f"{'='*60}")

        results = {
            "success": [],
            "failed": [],
            "timestamp": datetime.now().isoformat()
        }

        for i, train_no in enumerate(train_numbers, 1):
            print(f"\n[{i}/{len(train_numbers)}] Processing train {train_no}")

            data = self.fetch_train_info(train_no)

            if "error" not in data:
                results["success"].append(train_no)
            else:
                results["failed"].append({
                    "train_no": train_no,
                    "error": data.get("error")
                })

            # Rate limiting - wait 3 seconds between requests
            if i < len(train_numbers):
                time.sleep(3)

        # Save summary
        print(f"\n{'='*60}")
        print(f"[SUMMARY] Collection Complete")
        print(f"{'='*60}")
        print(f"[OK] Successfully fetched: {len(results['success'])} trains")
        print(f"[FAIL] Failed: {len(results['failed'])} trains")

        if results['failed']:
            print("\n[ERRORS]")
            for fail in results['failed']:
                print(f"  - Train {fail['train_no']}: {fail['error']}")

        # Save summary
        summary_file = self.output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        print(f"\n[SAVE] Summary saved to {summary_file.name}")
        print(f"[SAVE] All data saved in {self.output_dir}")

        return results


# Popular trains for testing
POPULAR_TRAINS = [
    "12302",  # Rajdhani Express (NDLS-HWH)
    "12301",  # Rajdhani Express (HWH-NDLS)
    "12951",  # Mumbai Rajdhani
    "12309",  # Patna Rajdhani
    "12423",  # Rajdhani Dibrugarh
    "12429",  # Lucknow Rajdhani
    "12001",  # Shatabdi Express
    "12553",  # Vaishali Express
    "12801",  # Purushottam Express
    "12433"   # Guwahati Rajdhani
]


def main():
    print("\n" + "=" * 60)
    print("RailSarthi AI - Data Collection")
    print("=" * 60)

    collector = RailRadarCollector()

    if not collector.api_key:
        print("\n[HELP] Setup Instructions:")
        print("1. Ensure .env file exists with RAILWAY_API_KEY")
        print("2. Run this script again")
        return

    # Collect data for trains
    num_trains = 5  # Start with 5 trains
    trains_to_collect = POPULAR_TRAINS[:num_trains]

    print(f"\n[INFO] Will collect data for {num_trains} trains:")
    for train_no in trains_to_collect:
        print(f"  - {train_no}")

    print("\n[INFO] Starting in 2 seconds...")
    time.sleep(2)

    results = collector.collect_multiple_trains(trains_to_collect)

    print("\n[DONE] Collection completed!")
    print("\nNext steps:")
    print("1. Check data/raw/ for collected JSON files")
    print("2. Examine the data structure")
    print("3. Build data validation and processing pipeline")


if __name__ == "__main__":
    main()
