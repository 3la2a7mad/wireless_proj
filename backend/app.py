# Import the new library at the top
import traceback

import os
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import openai

# Load environment variables from .env file for local testing
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for communication

# --- Configure the OFFICIAL OpenAI API (Slightly cleaner setup) ---
client = None
OPENAI_MODEL_ID = "gpt-3.5-turbo"

try:
    # This is a cleaner way to initialize the client for versions > 1.0
    # It reads the OPENAI_API_KEY from the environment automatically.
    api_key_from_env = os.environ.get('OPENAI_API_KEY')
    if not api_key_from_env:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    client = openai.OpenAI(api_key=api_key_from_env)
    print("OpenAI API client configured successfully.")
except Exception as e:
    print(f"Warning: Could not configure OpenAI client. {e}")


# --- YOUR CALCULATION AND VALIDATION FUNCTIONS ARE UNCHANGED ---
# (Leaving them out here for brevity, they remain the same)
def validate_inputs(scenario, parameters):
    # (Your existing validation code is perfect)
    if not isinstance(parameters, dict):
        return False, "Parameters must be a dictionary."
    if scenario == 'wireless_comm':
        required_params = ['samplerRate', 'quantizerBits', 'sourceEncoderRate', 'channelEncoderRate', 'interleaverDepth', 'burstLength']
        for param in required_params:
            if param not in parameters or not isinstance(parameters[param], (int, float)): return False, f"Missing or invalid parameter: '{param}'. It must be a numeric value."
            if parameters[param] <= 0: return False, f"'{param}' must be a positive value."
            if param in ['sourceEncoderRate', 'channelEncoderRate'] and not (0 < parameters[param] <= 1): return False, f"'{param}' must be between 0 and 1 (exclusive of 0)."
        return True, ""
    elif scenario == 'ofdm':
        required_params = ['numSubcarriers', 'symbolDuration', 'bitsPerSymbol', 'numResourceBlocks', 'bandwidth']
        for param in required_params:
            if param not in parameters or not isinstance(parameters[param], (int, float)): return False, f"Missing or invalid parameter: '{param}'. It must be a numeric value."
            if parameters[param] <= 0: return False, f"'{param}' must be a positive value."
        if parameters['bitsPerSymbol'] not in [1, 2, 4, 6, 8]: return False, "Bits Per Symbol must be 1, 2, 4, 6, or 8 (for common modulations)."
        return True, ""
    elif scenario == 'link_budget':
        required_params = ['transmitPower_dBm', 'transmitAntennaGain_dBi', 'receiveAntennaGain_dBi', 'frequency_GHz', 'distance_km', 'noiseFigure_dB', 'bandwidth_Hz']
        for param in required_params:
            if param not in parameters or not isinstance(parameters[param], (int, float)): return False, f"Missing or invalid parameter: '{param}'. It must be a numeric value."
            if param in ['frequency_GHz', 'distance_km', 'bandwidth_Hz'] and parameters[param] <= 0: return False, f"'{param}' must be a positive value."
        return True, ""
    elif scenario == 'cellular_design':
        required_params = ['numUsers', 'avgUserDataRate_Mbps', 'cellRadius_km', 'frequencyBand_GHz', 'maxTxPower_dBm', 'minRxSensitivity_dBm']
        for param in required_params:
            if param not in parameters or not isinstance(parameters[param], (int, float)): return False, f"Missing or invalid parameter: '{param}'. It must be a numeric value."
            if parameters[param] <= 0 and param not in ['minRxSensitivity_dBm']: return False, f"'{param}' must be a positive value."
        return True, ""
    else: return False, "Invalid scenario selected."

def perform_calculations(scenario, parameters):
    # (Your existing calculation code is perfect)
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
        data_rate_per_ofdm_symbol_period_bps = bits_per_ofdm_symbol * ofdm_symbol_rate_Hz
        bits_per_resource_block = 12 * 7 * parameters['bitsPerSymbol']
        rb_duration = 7 * parameters['symbolDuration']
        resource_block_data_rate_bps = bits_per_resource_block / rb_duration if rb_duration > 0 else float('inf')
        max_transmission_capacity_bps = data_rate_per_ofdm_symbol_period_bps * parameters['numResourceBlocks']
        spectral_efficiency_bps_per_Hz = max_transmission_capacity_bps / parameters['bandwidth'] if parameters['bandwidth'] > 0 else 0
        results.update({
            'bits_per_resource_element': parameters['bitsPerSymbol'],
            'bits_per_ofdm_symbol': bits_per_ofdm_symbol,
            'ofdm_symbol_rate_Hz': ofdm_symbol_rate_Hz,
            'data_rate_per_ofdm_symbol_period_bps': data_rate_per_ofdm_symbol_period_bps,
            'bits_per_resource_block': bits_per_resource_block,
            'resource_block_data_rate_bps': resource_block_data_rate_bps,
            'max_transmission_capacity_bps': max_transmission_capacity_bps,
            'spectral_efficiency_bps_per_Hz': spectral_efficiency_bps_per_Hz,
            'notes': "Calculations for OFDM Systems are based on ENCS5323 course material."
        })
    elif scenario == 'link_budget':
        f_Hz = parameters['frequency_GHz'] * 1e9
        d_m = parameters['distance_km'] * 1000
        FSPL_dB = 20 * math.log10(d_m) + 20 * math.log10(f_Hz) - 147.55 if d_m > 0 and f_Hz > 0 else float('inf')
        Pr_dBm = parameters['transmitPower_dBm'] + parameters['transmitAntennaGain_dBi'] + parameters['receiveAntennaGain_dBi'] - FSPL_dB
        N_watts = 1.38e-23 * 290 * parameters['bandwidth_Hz']
        N_dBm = 10 * math.log10(N_watts * 1000)
        SNR_dB = Pr_dBm - (N_dBm + parameters['noiseFigure_dB'])
        results.update({
            'free_space_path_loss_dB': FSPL_dB,
            'received_signal_strength_dBm': Pr_dBm,
            'thermal_noise_power_dBm': N_dBm,
            'signal_to_noise_ratio_dB': SNR_dB,
            'notes': "Calculations for Link Budget are based on ENCS5323 course material."
        })
    elif scenario == 'cellular_design':
        results['cell_area_km2'] = math.pi * parameters['cellRadius_km'] ** 2
        results['total_cell_capacity_Mbps'] = parameters['numUsers'] * parameters['avgUserDataRate_Mbps']
        results['notes'] = "Basic cellular design estimations based on user inputs."
    else: results['error'] = "Invalid scenario."
    return results
# --- END OF UNCHANGED CODE ---

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "API is running"}), 200

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    scenario = data.get('scenario')
    parameters = data.get('parameters')

    valid, error_msg = validate_inputs(scenario, parameters)
    if not valid:
        return jsonify({"error": error_msg}), 400

    calc_results = perform_calculations(scenario, parameters)

    if client is None:
        return jsonify({
            "results": calc_results,
            "explanation": "Could not generate explanation: OpenAI client is not configured on the server."
        }), 200

    prompt = f"""
    You are an expert ENCS5323 teaching assistant. Explain the following calculations clearly and concisely for a university student.
    Scenario: {scenario}, Inputs: {parameters}, Calculations: {calc_results}
    """

    # --- THIS IS THE CRUCIAL CHANGE ---
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a helpful engineering teaching assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.5,
        )
        explanation = response.choices[0].message.content
    except Exception as e:
        # This will print the FULL, detailed technical error to your Render logs.
        print("---! DETAILED OPENAI API ERROR TRACEBACK !---")
        traceback.print_exc()
        print("---------------------------------------------")
        explanation = f"Failed to generate explanation due to an API error. Check server logs for details."
    # --- END OF THE CHANGE ---

    return jsonify({
        "results": calc_results,
        "explanation": explanation
    }), 200


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)