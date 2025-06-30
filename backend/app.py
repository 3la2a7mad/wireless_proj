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

# --- OpenRouter Client Setup (Correct and Final) ---
client = None
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
try:
    api_key = os.environ.get('OPENROUTER_API_KEY')
    model_id = os.environ.get('OPENROUTER_MODEL_ID')
    if not api_key or not model_id:
        raise ValueError("CRITICAL: OPENROUTER_API_KEY or OPENROUTER_MODEL_ID is not set!")
    
    # This creates a client that ignores Render's injected proxy settings, which is essential.
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

# --- VALIDATION FUNCTION (Complete and Correct) ---
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


# --- CALCULATION FUNCTION (ALL SCENARIOS VERIFIED AND CORRECTED) ---
def perform_calculations(scenario, parameters):
    results = {}
    
    if scenario == 'wireless_comm':
        sampler_rate = parameters.get('samplerRate', 0)
        quantizer_bits = parameters.get('quantizerBits', 0)
        source_encoder_rate = parameters.get('sourceEncoderRate', 1)
        channel_encoder_rate = parameters.get('channelEncoderRate', 1)
        
        results['sampler_output_rate_bps'] = sampler_rate * quantizer_bits
        results['quantizer_output_rate_bps'] = results['sampler_output_rate_bps'] # No rate change
        results['source_encoder_output_rate_bps'] = results['quantizer_output_rate_bps'] * source_encoder_rate
        results['channel_encoder_output_rate_bps'] = results['source_encoder_output_rate_bps'] / channel_encoder_rate if channel_encoder_rate > 0 else float('inf')
        
        # CORRECTED LOGIC: Interleaving and Burst Formatting are structural and do not change the data rate.
        # They re-order bits for error protection but the number of bits per second remains the same.
        results['interleaver_output_rate_bps'] = results['channel_encoder_output_rate_bps']
        results['burst_formatting_output_rate_bps'] = results['interleaver_output_rate_bps']
        
        results['notes'] = "Calculations based on standard digital communication formulas."

    elif scenario == 'ofdm':
        bits_per_ofdm_symbol = parameters.get('numSubcarriers', 0) * parameters.get('bitsPerSymbol', 0)
        ofdm_symbol_rate_Hz = 1 / parameters.get('symbolDuration', float('inf')) if parameters.get('symbolDuration', 0) > 0 else float('inf')
        results['bits_per_ofdm_symbol'] = bits_per_ofdm_symbol
        results['ofdm_symbol_rate_Hz'] = ofdm_symbol_rate_Hz
        results['data_rate_bps'] = bits_per_ofdm_symbol * ofdm_symbol_rate_Hz
        results['spectral_efficiency_bps_per_Hz'] = results['data_rate_bps'] / parameters.get('bandwidth', float('inf')) if parameters.get('bandwidth', 0) > 0 else 0
        results['notes'] = "Calculations for OFDM Systems are based on standard formulas."

    elif scenario == 'link_budget':
        f_Hz = parameters.get('frequency_GHz', 0) * 1e9
        d_m = parameters.get('distance_km', 0) * 1000
        FSPL_dB = 20 * math.log10(d_m) + 20 * math.log10(f_Hz) - 147.55 if d_m > 0 and f_Hz > 0 else float('inf')
        Pr_dBm = parameters.get('transmitPower_dBm', 0) + parameters.get('transmitAntennaGain_dBi', 0) + parameters.get('receiveAntennaGain_dBi', 0) - FSPL_dB
        N_watts = 1.38e-23 * 290 * parameters.get('bandwidth_Hz', 1)
        N_dBm = 10 * math.log10(N_watts * 1000)
        SNR_dB = Pr_dBm - (N_dBm + parameters.get('noiseFigure_dB', 0))
        results['free_space_path_loss_dB'] = round(FSPL_dB, 2)
        results['received_signal_strength_dBm'] = round(Pr_dBm, 2)
        results['thermal_noise_power_dBm'] = round(N_dBm, 2)
        results['signal_to_noise_ratio_dB'] = round(SNR_dB, 2)
        results['notes'] = "Calculations for Link Budget based on standard formulas (Friis transmission equation)."

    elif scenario == 'cellular_design':
        results['cell_area_km2'] = round(math.pi * (parameters.get('cellRadius_km', 0) ** 2), 2)
        results['total_cell_capacity_Mbps'] = parameters.get('numUsers', 0) * parameters.get('avgUserDataRate_Mbps', 0)
        results['notes'] = "Basic cellular design estimations based on user inputs."
    
    return results

# --- Main API Endpoints (Final and Correct) ---
@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "API is live and fully configured for OpenRouter."}), 200

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    if not data: return jsonify({"error": "Invalid JSON body"}), 400
    
    scenario, parameters = data.get('scenario'), data.get('parameters')
    if not scenario or not parameters:
        return jsonify({"error": "Request body must contain 'scenario' and 'parameters'."}), 400
        
    valid, error_msg = validate_inputs(scenario, parameters)
    if not valid: return jsonify({"error": error_msg}), 400
    
    calc_results = perform_calculations(scenario, parameters)

    if client is None:
        return jsonify({"results": calc_results, "explanation": "Explanation failed: AI client not configured."})

    prompt = f"Expertly explain these calculations step-by-step for a university student, using only the formulas implemented in the system. Be very precise. Scenario: {scenario}, Inputs: {parameters}, Calculations: {calc_results}"
    try:
        response = client.chat.completions.create(model=OPENAI_MODEL_ID, messages=[{"role": "user", "content": prompt}])
        explanation = response.choices[0].message.content
    except Exception as e:
        print("---! OPENROUTER API CALL FAILED !---"); traceback.print_exc()
        explanation = "API Error: A problem occurred while contacting the AI model. Please check the server logs for details."
    
    return jsonify({"results": calc_results, "explanation": explanation})

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)