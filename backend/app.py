import os
import math
import traceback  # For detailed error logging
import requests   # For the network test

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import openai

# --- Configuration ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# --- OpenAI Client Setup ---
client = None
OPENAI_MODEL_ID = "gpt-3.5-turbo"
try:
    # The modern openai library automatically finds the OPENAI_API_KEY in the environment.
    # We just need to check if it exists first.
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("CRITICAL: OPENAI_API_KEY is not set in the environment variables!")
    client = openai.OpenAI()
    print("OpenAI API client configured successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not configure OpenAI client. {e}")

# --- YOUR CALCULATION & VALIDATION FUNCTIONS (Unchanged, here for completeness) ---
def validate_inputs(scenario, parameters):
    if not isinstance(parameters, dict): return False, "Parameters must be a dictionary."
    if scenario == 'wireless_comm':
        required_params = ['samplerRate', 'quantizerBits', 'sourceEncoderRate', 'channelEncoderRate', 'interleaverDepth', 'burstLength']
        for param in required_params:
            if param not in parameters or not isinstance(parameters[param], (int, float)): return False, f"Missing or invalid parameter: '{param}'. It must be a numeric value."
            if parameters[param] <= 0: return False, f"'{param}' must be a positive value."
            if param in ['sourceEncoderRate', 'channelEncoderRate'] and not (0 < parameters[param] <= 1): return False, f"'{param}' must be between 0 and 1."
        return True, ""
    return True, "" # Add other scenario validations if needed

def perform_calculations(scenario, parameters):
    results = {}
    if scenario == 'wireless_comm':
        results['sampler_output_rate_bps'] = parameters['samplerRate'] * parameters['quantizerBits']
        results['quantizer_output_rate_bps'] = results['sampler_output_rate_bps']
        results['source_encoder_output_rate_bps'] = results['quantizer_output_rate_bps'] * parameters['sourceEncoderRate']
        results['channel_encoder_output_rate_bps'] = results['source_encoder_output_rate_bps'] / parameters['channelEncoderRate'] if parameters['channelEncoderRate'] > 0 else float('inf')
        results['interleaver_output_rate_bps'] = results['channel_encoder_output_rate_bps']
        results['burst_formatting_output_rate_bps'] = results['interleaver_output_rate_bps']
        results['notes'] = "Calculations based on ENCS5323 course material."
    return results # Add other scenario calculations if needed
# --- END of your original functions ---

# --- API Endpoints ---
@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "API is running. Now with advanced diagnostics."}), 200

# --- NEW DIAGNOSTIC ENDPOINT ---
@app.route('/network-test', methods=['GET'])
def network_test():
    """Tests if the server can make any outbound HTTPS request."""
    print("--- Running Network Test ---")
    try:
        response = requests.get("https://www.google.com", timeout=10)
        print("Network test SUCCESS: Able to connect to google.com")
        return jsonify({
            "status": "SUCCESS",
            "message": "Outbound connection to google.com works.",
            "http_status_code": response.status_code
        }), 200
    except Exception as e:
        print("---!!! Network test FAILED !!!---")
        traceback.print_exc()
        return jsonify({
            "status": "FAILURE",
            "message": "Server could not make a basic outbound HTTPS connection.",
            "error_type": str(type(e).__name__),
            "error_details": str(e)
        }), 500

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    if not data or 'scenario' not in data or 'parameters' not in data:
        return jsonify({"error": "Invalid JSON body. 'scenario' and 'parameters' are required."}), 400

    scenario = data.get('scenario')
    parameters = data.get('parameters')
    valid, error_msg = validate_inputs(scenario, parameters)
    if not valid: return jsonify({"error": error_msg}), 400

    calc_results = perform_calculations(scenario, parameters)

    if client is None:
        return jsonify({"results": calc_results, "explanation": "Explanation failed: OpenAI client is not configured on the server."})

    prompt = f"""You are an expert ENCS5323 teaching assistant. Explain the following calculations clearly for a university student.
    Scenario: {scenario}, Inputs: {parameters}, Calculations: {calc_results}"""

    # --- CRUCIAL CHANGE: ADVANCED ERROR LOGGING ---
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a helpful engineering teaching assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        explanation = response.choices[0].message.content
    except Exception as e:
        # This will print the FULL technical error to your Render logs.
        print("---! DETAILED OPENAI API ERROR TRACEBACK !---")
        traceback.print_exc()
        print("---------------------------------------------")
        # Provide a more specific error message if possible
        if isinstance(e, openai.AuthenticationError):
            explanation = "Explanation failed: The OpenAI API key is invalid or has been revoked. Please check it in Render."
        elif isinstance(e, openai.RateLimitError):
            explanation = "Explanation failed: The OpenAI account has hit its rate limit. Please check your usage limits."
        elif isinstance(e, openai.APIConnectionError):
            explanation = "Explanation failed: Could not connect to OpenAI's servers. This may be a temporary network issue on Render's side or a firewall problem."
        else:
            explanation = f"Explanation failed: An unexpected API error occurred. See server logs for details. (Error Type: {type(e).__name__})"

    return jsonify({"results": calc_results, "explanation": explanation})

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)