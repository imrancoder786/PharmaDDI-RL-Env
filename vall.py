import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_health(session):
    print("Testing /health endpoint...")
    try:
        resp = session.get(f"{BASE_URL}/health")
        print(f"Response: {resp.text}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_reset(session):
    print("\nTesting /reset endpoint...")
    try:
        resp = session.post(f"{BASE_URL}/reset", json={})
        
        if resp.status_code != 200:
            print(f"Reset failed with status {resp.status_code}: {resp.text}")
            return False, None
            
        data = resp.json()
        obs = data.get("observation", {})
        print(f"Patient ID: {obs.get('patient_id')}")
        print(f"Task Name: {obs.get('task_name')}")
        print(f"Task Difficulty: {obs.get('task_difficulty')}")
        print(f"Medications: {[m['name'] for m in obs.get('medications', [])]}")
        
        return True, data
    except Exception as e:
        print(f"Reset check failed: {e}")
        return False, None

def test_step(session):
    print(f"\nTesting /step endpoint...")
    try:
        payload = {
            "action": {
                "interactions_found": []
            }
        }
        resp = session.post(f"{BASE_URL}/step", json=payload)
        
        if resp.status_code != 200:
            print(f"Step failed with status {resp.status_code}: {resp.text}")
            return False
            
        data = resp.json()
        obs = data.get("observation", {})
        print(f"Done: {data.get('done')}")
        print(f"Score: {obs.get('score')}")
        print(f"Feedback: {obs.get('feedback')}")
        return True
    except Exception as e:
        print(f"Step check failed: {e}")
        return False

def test_state(session):
    print("\nTesting /state endpoint...")
    try:
        resp = session.get(f"{BASE_URL}/state")
        
        if resp.status_code != 200:
            print(f"State failed with status {resp.status_code}: {resp.text}")
            return False
            
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"State check failed: {e}")
        return False

def main():
    print("=== PHARMA DDI ENV VALIDATOR ===\n")
    
    session = requests.Session()
    
    if not test_health(session):
        print("FAIL: Server not reachable. Make sure it is running on port 8000.")
        sys.exit(1)
        
    ok, reset_data = test_reset(session)
    if not ok:
        print("FAIL: Reset endpoint error.")
        sys.exit(1)
        
    if not test_step(session):
        print("FAIL: Step endpoint error.")
        sys.exit(1)
        
    if not test_state(session):
        print("FAIL: State endpoint error.")
        sys.exit(1)
        
    print("\nSUCCESS: Environment passed basic HTTP validation!")

if __name__ == "__main__":
    main()
