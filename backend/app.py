import os
import math
import traceback
import httpx
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import openai

# --- Configuration ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# --- OpenRouter Client Setup (The Main Change) ---
client = None
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

try:
    # We now need TWO environment variables from Render
    api_key = os.environ.get('OPENROUTER_API_KEY')
    model_id = os.environ.get('OPENROUTER_MODEL_ID')

    if not api_key or not model_id:
        raise ValueError("CRITICAL: OPENROUTER_API_KEY or OPENROUTER_MODEL_ID is not set!")

    # This is the key: we create a clean client that ignores Render's proxy
    custom_http_client = httpx.Client(trust_env=False)

    # We configure the OpenAI library to point to OpenRouter's servers
    client = openai.OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        http_client=custom_http_client
    )
    OPENAI_MODEL_ID = model_id # Use the model ID from the environment

    print(f"OpenRouter client configured successfully for model: {model_id}")

except Exception as e:
    print(f"CRITICAL ERROR: Could not configure OpenRouter client. {e}")
    traceback.print_exc()

# --- The rest of your code is perfect and needs no changes ---
def validate_inputs(scenario, parameters):
    if not isinstance(parameters, dict): return False, "Parameters must be a dictionary."
    if scenario == 'wireless_comm':
        required_params = ['samplerRate', 'quantizerBits', 'sourceEncoderRate', 'channelEncoderRate', 'interleaverDepth', 'burstLength']
        for param in required_params:
            if param not in parameters or not isinstance(parameters[param], (int, float)): return False, f"Missing or invalid parameter: '{param}'."
            if parameters[param] <= 0: return False, f"'{param}' must be a positive value."
            if param in ['sourceEncoderRate', 'channelEncoderRate'] and not (0 < parameters[param] <= 1): return False, f"'{param}' must be between 0 and 1."
        return True, ""
    return True, ""

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
    return results

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "API is live and configured for OpenRouter."}), 200

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    if not data: return jsonify({"error": "Invalid JSON body"}), 400
    scenario, parameters = data.get('scenario'), data.get('parameters')
    valid, error_msg = validate_inputs(scenario, parameters)
    if not valid: return jsonify({"error": error_msg}), 400
    calc_results = perform_calculations(scenario, parameters)

    if client is None:
        return jsonify({"results": calc_results, "explanation": "Explanation failed: AI client is not configured on the server."})

    prompt = f"Expertly explain these calculations for a university student. Scenario: {scenario}, Inputs: {parameters}, Calculations: {calc_results}"
    try:
        response = client.chat.completions.create(model=OPENAI_MODEL_ID, messages=[{"role": "user", "content": prompt}])
        explanation = response.choices[0].message.content
    except Exception as e:
        print("---! OPENROUTER API CALL FAILED !---")
        traceback.print_exc()
        explanation = "API Error: A problem occurred while contacting the AI model. Check server logs."

    return jsonify({"results": calc_results, "explanation": explanation})

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)