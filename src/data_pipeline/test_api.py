"""
Quick API Test - Check RailRadar API Structure
===============================================
Test different endpoint formats to find the correct API structure
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RAILWAY_API_KEY")
BASE_URL = "https://api.railradar.in/v1"

print("=" * 60)
print("Testing RailRadar API Endpoints")
print("=" * 60)
print(f"API Key: {API_KEY[:15]}...")
print(f"Base URL: {BASE_URL}\n")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Test different endpoint patterns
test_endpoints = [
    "/trains/12302/schedule",
    "/trains/12302",
    "/train/12302",
    "/schedule/12302",
    "/live/12302",
    "/api/trains/12302",
]

print("Testing endpoints:")
print("-" * 60)

for endpoint in test_endpoints:
    url = f"{BASE_URL}{endpoint}"
    print(f"\nTrying: {endpoint}")
    print(f"Full URL: {url}")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            print("[SUCCESS] Got response!")
            data = response.json()
            print(f"Response keys: {list(data.keys())}")
            print(f"Sample data: {str(data)[:200]}...")
        elif response.status_code == 404:
            print("[NOT FOUND] Endpoint doesn't exist")
        elif response.status_code == 401:
            print("[AUTH ERROR] API key might be invalid")
        elif response.status_code == 403:
            print("[FORBIDDEN] No access to this endpoint")
        else:
            print(f"[OTHER] Response: {response.text[:200]}")

    except Exception as e:
        print(f"[ERROR] {e}")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)

# Also test without version in URL
print("\n\nTrying without /v1 in base URL:")
BASE_URL_ALT = "https://api.railradar.in"
url = f"{BASE_URL_ALT}/trains/12302"
print(f"URL: {url}")

try:
    response = requests.get(url, headers=headers, timeout=10)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("[SUCCESS]")
        print(f"Response: {response.json()}")
except Exception as e:
    print(f"[ERROR] {e}")

# Try checking API documentation endpoint
print("\n\nTrying to find API documentation:")
doc_urls = [
    "https://api.railradar.in/",
    "https://api.railradar.in/v1",
    "https://api.railradar.in/docs",
    "https://api.railradar.in/v1/docs",
]

for doc_url in doc_urls:
    try:
        response = requests.get(doc_url, timeout=5)
        if response.status_code == 200:
            print(f"\n[FOUND] {doc_url}")
            print(f"Content preview: {response.text[:300]}")
    except:
        pass
