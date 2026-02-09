import requests
import json
import os

INDIA_LOCATION_API = "https://india-location-hub.in/api/locations"
CACHE_FILE = "location_cache.json"

def prefetch():
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            try:
                cache = json.load(f)
            except:
                cache = {}
    
    print("Pre-fetching States...")
    url = f"{INDIA_LOCATION_API}/states"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('success'):
                states = data.get('data', {}).get('states', [])
                # Store the full normalized response that the app expects
                cache[url] = {"success": True, "states": states}
                
                # Pre-fetch districts for top 5 states by name
                top_states = sorted(states, key=lambda x: x['name'])[:5]
                for state in top_states:
                    state_name = state['name']
                    print(f"Pre-fetching districts for {state_name}...")
                    dist_url = f"{INDIA_LOCATION_API}/districts?state={state_name}"
                    d_resp = requests.get(dist_url, timeout=15)
                    if d_resp.status_code == 200:
                        d_data = d_resp.json()
                        if d_data.get('success'):
                            districts = d_data.get('data', {}).get('districts', [])
                            cache[f"/api/districts/{state_name}"] = {"success": True, "districts": districts}
        
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
        print("Pre-fetch complete! Cache updated.")
    except Exception as e:
        print(f"Error during pre-fetch: {e}")

if __name__ == "__main__":
    prefetch()
