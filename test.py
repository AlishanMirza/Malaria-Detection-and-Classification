import requests
import json
import os
import time

# --- SCRIPT SANITY CHECK ---
print("--- Running Bulk API Test Script (test.py) ---")

try:
    # Import the new function to get multiple samples
    from sample import get_n_random_malware_samples
except ImportError:
    print("\n❌ FATAL ERROR: Cannot find the helper script 'sample.py' or the required function.")
    exit()

# --- CONFIG ---
API_URL = "http://127.0.0.1:5000/predict"
NUM_SAMPLES_TO_TEST = 50

def test_api_with_multiple_samples():
    """
    Gets multiple samples, sends them to the API one by one, and prints a summary.
    """
    print(f"\n➡️  Step 1: Getting {NUM_SAMPLES_TO_TEST} random malware samples...")
    features_list, true_families = get_n_random_malware_samples(NUM_SAMPLES_TO_TEST)

    if not features_list:
        print("\n❌ ERROR: The 'sample.py' script could not find any samples.")
        return

    print(f"✅ Got {len(features_list)} samples. Starting prediction loop...")
    
    correct_predictions = 0
    
    for i, (features, true_family) in enumerate(zip(features_list, true_families)):
        print(f"\n--- Testing Sample {i + 1}/{NUM_SAMPLES_TO_TEST} (True Family: '{true_family}') ---")
        
        # Convert numpy floats to standard Python floats for JSON
        features_as_python_floats = [float(x) for x in features]
        api_data = {"features": features_as_python_floats}
        
        try:
            response = requests.post(API_URL, json=api_data)
            response.raise_for_status()
            
            prediction_data = response.json()
            predicted_family = prediction_data['prediction']['family_name']
            
            print(f"   ➡️ Prediction: '{predicted_family}'")
            
            if predicted_family == true_family:
                correct_predictions += 1
                print("   ✅ CORRECT")
            else:
                print("   ❌ INCORRECT")

            time.sleep(0.1) # Small delay to avoid overwhelming the server

        except requests.exceptions.RequestException as e:
            print(f"\n❌ ERROR: Could not connect to the API at {API_URL}")
            print("   Please make sure your 'app.py' server is running in another terminal.")
            return # Stop the test if the server isn't running
            
    # --- Final Summary ---
    print("\n======================")
    print("   BULK TEST COMPLETE")
    print("======================")
    accuracy = (correct_predictions / len(features_list)) * 100
    print(f"Correct Predictions: {correct_predictions} / {len(features_list)}")
    print(f"Accuracy on this run: {accuracy:.2f}%")

if __name__ == "__main__":
    test_api_with_multiple_samples()

