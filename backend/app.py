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

# --- OpenAI Client Setup ---
client = None
OPENAI_MODEL_ID = "gpt-3.5-turbo"

try:
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("CRITICAL: OPENAI_API_KEY is not set in the environment variables!")

    # Create a network client that explicitly DOES NOT TRUST the environment.
    custom_http_client = httpx.Client(trust_env=False)

    # Pass this clean, non-proxied client to OpenAI.
    client = openai.OpenAI(http_client=custom_http_client)

    print("OpenAI API client configured successfully by disabling environment trust.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not configure OpenAI client. {e}")
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
    return jsonify({"status": "API is live and configured correctly."}), 200

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    if not data: return jsonify({"error": "Invalid JSON body"}), 400
    scenario, parameters = data.get('scenario'), data.get('parameters')
    valid, error_msg = validate_inputs(scenario, parameters)
    if not valid: return jsonify({"error": error_msg}), 400
    calc_results = perform_calculations(scenario, parameters)

    if client is None:
        return jsonify({
            "results": calc_results,
            "explanation": "Explanation unavailable: OpenAI client is not configured on the server."
        }), 200

    prompt = f"Expertly explain these calculations for a university student. Scenario: {scenario}, Inputs: {parameters}, Calculations: {calc_results}"

    try:
        response = client.chat.completions.create(model=OPENAI_MODEL_ID, messages=[{"role": "user", "content": prompt}])
        explanation = response.choices[0].message.content
    except openai.RateLimitError as e:
        print("---! OPENAI API RATE LIMIT EXCEEDED !---")
        traceback.print_exc()
        return jsonify({
            "results": calc_results,
            "explanation": "Explanation unavailable: OpenAI API rate limit exceeded. Please try again later or contact support."
        }), 200
    except openai.AuthenticationError as e:
        print("---! OPENAI API AUTHENTICATION ERROR !---")
        traceback.print_exc()
        return jsonify({
            "results": calc_results,
            "explanation": "Explanation unavailable: Invalid OpenAI API key."
        }), 200
    except Exception as e:
        print("---! OPENAI API CALL FAILED !---")
        traceback.print_exc()
        return jsonify({
            "results": calc_results,
            "explanation": "Explanation unavailable: An error occurred while connecting to the OpenAI API. Check server logs."
        }), 200

    return jsonify({"results": calc_results, "explanation": explanation}), 200

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)