import os
import math  # Make sure math module is imported for calculations
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import openai  # Using openai library for OpenRouter compatibility

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend to communicate with backend

# --- Configure OpenRouter API ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_ID = os.getenv("OPENROUTER_MODEL_ID")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"  # Standard OpenRouter base URL

print(f"OPENROUTER_API_KEY: {OPENROUTER_API_KEY is not None}, OPENROUTER_MODEL_ID: {OPENROUTER_MODEL_ID}")

if not OPENROUTER_API_KEY or not OPENROUTER_MODEL_ID:
    print("Warning: OPENROUTER_API_KEY or OPENROUTER_MODEL_ID not found in .env. LLM functionality will be unavailable.")
    openrouter_client = None
else:
    try:
        # OpenRouter API is compatible with OpenAI Python client
        openrouter_client = openai.OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
    except Exception as e:
        print(f"Error configuring OpenRouter API: {e}. LLM functionality will be limited.")
        openrouter_client = None


def validate_inputs(scenario, parameters):
    """
    Validates user inputs for each scenario based on the project requirements.
    Returns (True, "") if valid, or (False, "error message") otherwise.
    """
    if not isinstance(parameters, dict):
        return False, "Parameters must be a dictionary."

    if scenario == 'wireless_comm':
        required_params = ['samplerRate', 'quantizerBits', 'sourceEncoderRate', 'channelEncoderRate', 'interleaverDepth', 'burstLength']
        for param in required_params:
            if param not in parameters or not isinstance(parameters[param], (int, float)):
                return False, f"Missing or invalid parameter: '{param}'. It must be a numeric value."
            if parameters[param] <= 0:
                return False, f"'{param}' must be a positive value."
            if param in ['sourceEncoderRate', 'channelEncoderRate'] and not (0 < parameters[param] <= 1):
                return False, f"'{param}' must be between 0 and 1 (exclusive of 0)."
        return True, ""
    elif scenario == 'ofdm':
        required_params = ['numSubcarriers', 'symbolDuration', 'bitsPerSymbol', 'numResourceBlocks', 'bandwidth']
        for param in required_params:
            if param not in parameters or not isinstance(parameters[param], (int, float)):
                return False, f"Missing or invalid parameter: '{param}'. It must be a numeric value."
            if parameters[param] <= 0:
                return False, f"'{param}' must be a positive value."
        if parameters['bitsPerSymbol'] not in [1, 2, 4, 6, 8]:
            return False, "Bits Per Symbol must be 1, 2, 4, 6, or 8 (for common modulations)."
        return True, ""
    elif scenario == 'link_budget':
        required_params = ['transmitPower_dBm', 'transmitAntennaGain_dBi', 'receiveAntennaGain_dBi', 'frequency_GHz', 'distance_km', 'noiseFigure_dB', 'bandwidth_Hz']
        for param in required_params:
            if param not in parameters or not isinstance(parameters[param], (int, float)):
                return False, f"Missing or invalid parameter: '{param}'. It must be a numeric value."
            if param in ['frequency_GHz', 'distance_km', 'bandwidth_Hz'] and parameters[param] <= 0:
                return False, f"'{param}' must be a positive value."
        return True, ""
    elif scenario == 'cellular_design':
        required_params = ['numUsers', 'avgUserDataRate_Mbps', 'cellRadius_km', 'frequencyBand_GHz', 'maxTxPower_dBm', 'minRxSensitivity_dBm']
        for param in required_params:
            if param not in parameters or not isinstance(parameters[param], (int, float)):
                return False, f"Missing or invalid parameter: '{param}'. It must be a numeric value."
            if parameters[param] <= 0 and param not in ['minRxSensitivity_dBm']:
                return False, f"'{param}' must be a positive value."
        return True, ""
    else:
        return False, "Invalid scenario selected."


def perform_calculations(scenario, parameters):
    results = {}

    if scenario == 'wireless_comm':
        sampler_rate = parameters['samplerRate']
        quantizer_bits = parameters['quantizerBits']
        source_encoder_rate = parameters['sourceEncoderRate']
        channel_encoder_rate = parameters['channelEncoderRate']
        interleaver_depth = parameters['interleaverDepth']
        burst_length = parameters['burstLength']

        results['sampler_output_rate_bps'] = sampler_rate * quantizer_bits
        results['quantizer_output_rate_bps'] = results['sampler_output_rate_bps']
        results['source_encoder_output_rate_bps'] = results['quantizer_output_rate_bps'] * source_encoder_rate
        results['channel_encoder_output_rate_bps'] = results['source_encoder_output_rate_bps'] / channel_encoder_rate if channel_encoder_rate > 0 else float('inf')
        results['interleaver_output_rate_bps'] = results['channel_encoder_output_rate_bps']
        results['burst_formatting_output_rate_bps'] = results['interleaver_output_rate_bps']

        results['notes'] = "Calculations for Wireless Communication System are based on ENCS5323 course material."

    elif scenario == 'ofdm':
        num_subcarriers = parameters['numSubcarriers']
        symbol_duration = parameters['symbolDuration']
        bits_per_symbol = parameters['bitsPerSymbol']
        num_resource_blocks = parameters['numResourceBlocks']
        bandwidth = parameters['bandwidth']

        results['bits_per_resource_element'] = bits_per_symbol
        bits_per_ofdm_symbol = num_subcarriers * bits_per_symbol
        results['bits_per_ofdm_symbol'] = bits_per_ofdm_symbol
        ofdm_symbol_rate_Hz = 1 / symbol_duration if symbol_duration > 0 else float('inf')
        results['ofdm_symbol_rate_Hz'] = ofdm_symbol_rate_Hz
        data_rate_per_ofdm_symbol_period_bps = bits_per_ofdm_symbol * ofdm_symbol_rate_Hz
        results['data_rate_per_ofdm_symbol_period_bps'] = data_rate_per_ofdm_symbol_period_bps

        subcarriers_per_RB = 12
        symbols_per_RB = 7
        bits_per_resource_block = subcarriers_per_RB * symbols_per_RB * bits_per_symbol
        results['bits_per_resource_block'] = bits_per_resource_block
        rb_duration = symbols_per_RB * symbol_duration
        resource_block_data_rate_bps = bits_per_resource_block / rb_duration if rb_duration > 0 else float('inf')
        results['resource_block_data_rate_bps'] = resource_block_data_rate_bps

        max_transmission_capacity_bps = data_rate_per_ofdm_symbol_period_bps * num_resource_blocks
        results['max_transmission_capacity_bps'] = max_transmission_capacity_bps
        spectral_efficiency_bps_per_Hz = max_transmission_capacity_bps / bandwidth if bandwidth > 0 else 0
        results['spectral_efficiency_bps_per_Hz'] = spectral_efficiency_bps_per_Hz

        results['notes'] = "Calculations for OFDM Systems are based on ENCS5323 course material."

    elif scenario == 'link_budget':
        Pt_dBm = parameters['transmitPower_dBm']
        Gt_dBi = parameters['transmitAntennaGain_dBi']
        Gr_dBi = parameters['receiveAntennaGain_dBi']
        f_GHz = parameters['frequency_GHz']
        d_km = parameters['distance_km']
        NF_dB = parameters['noiseFigure_dB']
        B_Hz = parameters['bandwidth_Hz']

        c = 3e8
        f_Hz = f_GHz * 1e9
        d_m = d_km * 1000

        if d_m > 0 and f_Hz > 0:
            FSPL_dB = 20 * math.log10(d_m) + 20 * math.log10(f_Hz) - 147.55  # Corrected FSPL formula
        else:
            FSPL_dB = float('inf')
        results['free_space_path_loss_dB'] = FSPL_dB

        Pr_dBm = Pt_dBm + Gt_dBi + Gr_dBi - FSPL_dB
        results['received_signal_strength_dBm'] = Pr_dBm

        k = 1.38e-23
        T0 = 290
        N_watts = k * T0 * B_Hz
        N_dBm = 10 * math.log10(N_watts * 1000)
        results['thermal_noise_power_dBm'] = N_dBm

        SNR_dB = Pr_dBm - (N_dBm + NF_dB)
        results['signal_to_noise_ratio_dB'] = SNR_dB

        results['notes'] = "Calculations for Link Budget are based on ENCS5323 course material."

    elif scenario == 'cellular_design':
        num_users = parameters['numUsers']
        avg_user_data_rate_Mbps = parameters['avgUserDataRate_Mbps']
        cell_radius_km = parameters['cellRadius_km']
        freq_GHz = parameters['frequencyBand_GHz']
        max_tx_power_dBm = parameters['maxTxPower_dBm']
        min_rx_sensitivity_dBm = parameters['minRxSensitivity_dBm']

        cell_area_km2 = math.pi * cell_radius_km ** 2
        results['cell_area_km2'] = cell_area_km2

        total_capacity_Mbps = num_users * avg_user_data_rate_Mbps
        results['total_cell_capacity_Mbps'] = total_capacity_Mbps

        results['notes'] = "Basic cellular design estimations based on user inputs."

    else:
        results['error'] = "Invalid scenario."

    return results


@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "API is running"}), 200


@app.route('/test', methods=['GET'])
def test():
    if not OPENROUTER_API_KEY or not OPENROUTER_MODEL_ID:
        return jsonify({"error": "API key or model ID missing"}), 500
    if not openrouter_client:
        return jsonify({"error": "OpenRouter client not configured properly"}), 503

    try:
        response = openrouter_client.chat.completions.create(
            model=OPENROUTER_MODEL_ID,
            messages=[{"role": "system", "content": "Say hello"}],
            max_tokens=5
        )
        content = response.choices[0].message.content if response.choices else "No response"
        return jsonify({"success": True, "response": content}), 200
    except Exception as e:
        return jsonify({"error": f"OpenRouter connection failed: {str(e)}"}), 500


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

    if openrouter_client is None:
        return jsonify({
            "calculations": calc_results,
            "warning": "LLM (OpenRouter) not configured. No explanations generated."
        }), 200

    prompt = f"""
    You are an expert ENCS5323 teaching assistant.

    Scenario: {scenario}

    Inputs:
    {parameters}

    Calculations:
    {calc_results}

    Explain the above calculations clearly and concisely.
    """

    try:
        response = openrouter_client.chat.completions.create(
            model=OPENROUTER_MODEL_ID,
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        explanation = response.choices[0].message.content
    except Exception as e:
        explanation = f"Failed to generate explanation due to API error: {str(e)}"

    return jsonify({
        "calculations": calc_results,
        "explanation": explanation
    }), 200


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
