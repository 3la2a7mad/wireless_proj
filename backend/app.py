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

# --- OpenRouter Client Setup (Final Correct Version) ---
client = None
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

try:
    api_key = os.environ.get('OPENROUTER_API_KEY')
    model_id = os.environ.get('OPENROUTER_MODEL_ID')

    if not api_key or not model_id:
        raise ValueError("CRITICAL: OPENROUTER_API_KEY or OPENROUTER_MODEL_ID is not set!")

    custom_http_client = httpx.Client(trust_env=False)
    client = openai.OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        http_client=custom_http_client
    )
    OPENAI_MODEL_ID = model_id
    print(f"OpenRouter client configured successfully for model: {model_id}")
except Exception as e:
    print(f"CRITICAL ERROR: Could not configure OpenRouter client. {e}")
    traceback.print_exc()

# --- VALIDATION FUNCTION (FULL VERSION RESTORED) ---
def validate_inputs(scenario, parameters):
    if not isinstance(parameters, dict):
        return False, "Parameters must be a dictionary."

    if scenario == 'wireless_comm':
        required_params = ['samplerRate', 'quantizerBits', 'sourceEncoderRate', 'channelEncoderRate', 'interleaverDepth', 'burstLength']
    elif scenario == 'ofdm':
        required_params = ['numSubcarriers', 'symbolDuration', 'bitsPerSymbol', 'numResourceBlocks', 'bandwidth']
    elif scenario == 'link_budget':
        required_params = ['transmitPower_dBm', 'transmitAntennaGain_dBi', 'receiveAntennaGain_dBi', 'frequency_GHz', 'distance_km', 'noiseFigure_dB', 'bandwidth_Hz']
    elif scenario == 'cellular_design':
        required_params = ['numUsers', 'avgUserDataRate_Mbps', 'cellRadius_km', 'frequencyBand_GHz', 'maxTxPower_dBm', 'minRxSensitivity_dBm']
    else:
        return False, "Invalid scenario selected."

    for param in required_params:
        if param not in parameters or not isinstance(parameters[param], (int, float)):
            return False, f"Missing or invalid parameter: '{param}'. It must be a numeric value."

    return True, ""


# --- CALCULATION FUNCTION (FULL VERSION RESTORED) ---
def perform_calculations(scenario, parameters):
    results = {}
    if scenario == 'wireless_comm':
        results['sampler_output_rate_bps'] = parameters['samplerRate'] * parameters['quantizerBits']
        results['quantizer_output_rate_bps'] = results['sampler_output_rate_bps']
        results['source_encoder_output_rate_bps'] = results['quantizer_output_rate_bps'] * parameters['sourceEncoderRate']
        results['channel_encoder_output_rate_bps'] = results['source_encoder_output_rate_bps'] / parameters['channelEncoderRate'] if parameters['channelEncoderRate'] > 0 else float('inf')
        results['interleaver_output_rate_bps'] = results['channel_encoder_output_rate_bps']
        results['burst_formatting_output_rate_bps'] = results['interleaver_output_rate_bps']
        results['notes'] = "Calculations for Wireless Communication System are based on ENCS5323 course material."

    elif scenario == 'ofdm':
        bits_per_ofdm_symbol = parameters['numSubcarriers'] * parameters['bitsPerSymbol']
        ofdm_symbol_rate_Hz = 1 / parameters['symbolDuration'] if parameters['symbolDuration'] > 0 else float('inf')
        results['bits_per_ofdm_symbol'] = bits_per_ofdm_symbol
        results['ofdm_symbol_rate_Hz'] = ofdm_symbol_rate_Hz
        results['data_rate_bps'] = bits_per_ofdm_symbol * ofdm_symbol_rate_Hz
        results['spectral_efficiency_bps_per_Hz'] = results['data_rate_bps'] / parameters['bandwidth'] if parameters['bandwidth'] > 0 else 0
        results['notes'] = "Calculations for OFDM Systems are based on ENCS5323 course material."

    elif scenario == 'link_budget':
        f_Hz = parameters['frequency_GHz'] * 1e9
        d_m = parameters['distance_km'] * 1000
        FSPL_dB = 20 * math.log10(d_m) + 20 * math.log10(f_Hz) - 147.55 if d_m > 0 and f_Hz > 0 else float('inf')
        Pr_dBm = parameters['transmitPower_dBm'] + parameters['transmitAntennaGain_dBi'] + parameters['receiveAntennaGain_dBi'] - FSPL_dB
        N_watts = 1.38e-23 * 290 * parameters['bandwidth_Hz']
        N_dBm = 10 * math.log10(N_watts * 1000)
        SNR_dB = Pr_dBm - (N_dBm + parameters['noiseFigure_dB'])
        results['free_space_path_loss_dB'] = FSPL_dB
        results['received_signal_strength_dBm'] = Pr_dBm
        results['thermal_noise_power_dBm'] = N_dBm
        results['signal_to_noise_ratio_dB'] = SNR_dB
        results['notes'] = "Calculations for Link Budget are based on ENCS5323 course material."

    elif scenario == 'cellular_design':
        results['cell_area_km2'] = math.pi * parameters['cellRadius_km'] ** 2
        results['total_cell_capacity_Mbps'] = parameters['numUsers'] * parameters['avgUserDataRate_Mbps']
        results['notes'] = "Basic cellular design estimations based on user inputs."
    
    return results

# --- Main API Endpoints (No Changes Needed Here) ---
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