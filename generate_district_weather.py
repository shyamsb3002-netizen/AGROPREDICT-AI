import requests
import json
import os

INDIA_LOCATION_API = "https://india-location-hub.in/api/locations"
WEATHER_AVERAGES_FILE = "weather_averages.json"

def generate_comprehensive_dataset():
    # Load current state-level averages
    if not os.path.exists(WEATHER_AVERAGES_FILE):
        print("Bailing: weather_averages.json not found")
        return
        
    with open(WEATHER_AVERAGES_FILE, 'r') as f:
        weather_data = json.load(f)

    # We only want to keep the original state keys plus new district keys
    # But districts should be keyed by their name (UPPERCASE for consistency)
    
    # Get all states
    print("Fetching states from API...")
    try:
        resp = requests.get(f"{INDIA_LOCATION_API}/states", timeout=15)
        if resp.status_code == 200:
            states_api_data = resp.json().get('data', {}).get('states', [])
            
            for state in states_api_data:
                state_name = state['name']
                state_key = state_name.upper()
                
                # Check if we have averages for this state
                # If not, use a generic fallback (though most should exist from my previous work)
                state_avg = weather_data.get(state_key, {"temperature": 25.0, "humidity": 60.0, "rainfall": 1000.0})
                
                print(f"Fetching districts for {state_name}...")
                d_resp = requests.get(f"{INDIA_LOCATION_API}/districts?state={state_name}", timeout=15)
                if d_resp.status_code == 200:
                    districts = d_resp.json().get('data', {}).get('districts', [])
                    for district in districts:
                        district_name = district['name']
                        dist_key = district_name.upper()
                        
                        # Add district to the dataset using state average as baseline
                        if dist_key not in weather_data:
                            weather_data[dist_key] = state_avg
        
        # Save the expanded dataset
        with open(WEATHER_AVERAGES_FILE, 'w') as f:
            json.dump(weather_data, f, indent=2)
            
        print(f"Successfully expanded dataset! New total entries: {len(weather_data)}")
        
    except Exception as e:
        print(f"Error during dataset generation: {e}")

if __name__ == "__main__":
    generate_comprehensive_dataset()
