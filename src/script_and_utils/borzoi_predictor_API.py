'''Borzoi Predictor using Flask'''
import sys
import math
import json
import argparse

from flask import Flask

from config import BORZOI_DIR, HELP_FILE, PREDICTOR_NAME, SUPPORTED_REQUEST_FORMATS, SUPPORTED_RESPONSE_FORMATS
from error_checking_functions import APIError, ServerError, PredictionFailedError
from schema_validation import validate_request_payload, preprocess_data
from predictor_content_handler import decode_request, encode_response

# Add BORZOI_DIR to the Python path
if BORZOI_DIR not in sys.path:
    sys.path.insert(0, BORZOI_DIR)

from borzoi_predict_codebase import predict_borzoi
from model_validation import model_specific_payload_validation, apply_scaling

# --- Have arguments be defined globally ---
parser = argparse.ArgumentParser(description=f'{PREDICTOR_NAME} Predictor API')
parser.add_argument('ip', type=str, help='IP address to bind')
parser.add_argument('port', type=int, help='Port to bind')
parser.add_argument('matcher_ip', type=str, nargs='?', default=None, help='Matcher Service IP (Optional)')
parser.add_argument('matcher_port', type=int, nargs='?', default=None, help='Matcher Service Port (Optional)')

# # Optional Flag for Threads
# parser.add_argument("--threads", "-t", type=int, default=4, help="Waitress Threads (Default: 4)")

args = parser.parse_args()

predictor_ip = args.ip
predictor_port = args.port
matcher_ip = args.matcher_ip
matcher_port = args.matcher_port

if matcher_ip and matcher_port:
    print(f"Matcher service configured at: {matcher_ip}:{matcher_port}")
else:
    print("Matcher service is NOT configured. Running in 'exact-match-only' mode.")
    
# if args.threads != 4:
#     print(f"Manual Configuration: Using {args.threads} threads.")

# ------------ Sanitization Functions -----------
def _print_non_numeric(obj, path="root"):
    """
    Recursively print all non-numeric entries in a nested dictionary or list.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            _print_non_numeric(v, path + f".{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _print_non_numeric(v, path + f"[{i}]")
    else:
        try:
            # Try to convert to float
            val = float(obj)
            if not math.isfinite(val):
                print(f"Non-finite numeric at {path}: {obj}")
        except (ValueError, TypeError):
            # Not numeric at all
            print(f"Non-numeric at {path}: {obj}")

MAX_VALUE = 1e5 
MIN_VALUE = -1e5

def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            # Replace with a large finite number
            return MAX_VALUE if obj > 0 else MIN_VALUE
        else:
            return obj
    else:
        return obj

# -------- Flask App and Central Error Handler --------
app = Flask(__name__)
app.json.sort_keys = False

def create_error_response(error_key, messages, status_code):
    """ 
    Formats error response into a standardized JSON structure.
    
    Args:
        error_key (str): The category of the error (e.g. 'bad_prediction_request', 'prediction_request_failed').
        messages (list or str): A list of error message strings or a single message.
        status_code (int): Standard HTTP error status code based on the error.
    
    Returns:
        dict: A dictionary formatted for the standardized JSON error response.
    """
    if not isinstance(messages, list):
        messages = [str(messages)]
    error_payload = {"error": [{error_key: msg} for msg in messages]}
    print(error_payload)
    return error_payload, status_code

@app.errorhandler(APIError)
def handle_api_error(error):
    """This single handler catches all of our custom API errors."""
    # Get raw payload and status code
    payload, status_code = create_error_response(error.error_key, error.message, error.status_code)
    
    return encode_response(
        payload, 
        status_code=status_code,
        isError=True,
        predictor_name=PREDICTOR_NAME)
    
@app.after_request
def after_request_callback(response):
    """This function runs after each request is processed."""
    print(f"\n--- Sending predictions back to Evaluator. ---")
    print(f"--- Request Complete. {PREDICTOR_NAME} Predictor is listening on http://{predictor_ip}:{predictor_port} ---\n")
    return response

# --- API Endpoints ---
@app.route('/formats', methods=['GET'])
def formats_endpoint():
    """Provides the Predictor's supported formats"""
    supported_fmts = {
        "predictor_supported_request_formats": SUPPORTED_REQUEST_FORMATS,
        "predictor_supported_response_formats": SUPPORTED_RESPONSE_FORMATS
    }
    try:
        return encode_response(
            supported_fmts,
            status_code=200,
            predictor_name=PREDICTOR_NAME,
            supported_response_formats=SUPPORTED_RESPONSE_FORMATS)
    except Exception as e:
        raise ServerError(f"Error serializing supported format for /format endpoint: {e}")


@app.route('/help', methods=['GET'])
def help_endpoint():
    """Provides the Predictor's help/metadata information."""
    try:
        with open(HELP_FILE, 'r') as f:
            help_data = json.load(f)
        return encode_response(
            help_data,
            status_code=200,
            predictor_name=PREDICTOR_NAME,
            supported_response_formats=SUPPORTED_RESPONSE_FORMATS)
    except Exception as e:
        raise ServerError(f"Error reading help file: {e}")
    
@app.route('/predict', methods=['POST'])
def predict():
    """The main endpoint for receiving sequences and returning predictions."""
    try:
        # Decode incoming request
        evaluator_request = decode_request(SUPPORTED_REQUEST_FORMATS)
        
        # Validate the payload using the imported function
        # These functions will raise an APIError on failure,
        # which will be caught automatically by @app.errorhandler
        validate_request_payload(evaluator_request)
        
        # Model validation for all Borzoi-specific logic checks
        model_specific_payload_validation(evaluator_request)
        
        readout_type = evaluator_request['readout']
        is_point_readout = readout_type == "point" # point by default. If "track", then is_point_readout = False -- used for conditional logic
        
        # Preprocess request
        # Applies flanks, checks specs, checks prediction ranges, and slices
        sequences = preprocess_data(evaluator_request)
        
        # ---------------------- Extract Prediction Tasks and Run the Model ----------------------
        # Start big loop here for all the prediction_tasks
        # First step is to collect all unique tasks
        request_tasks = set()  # Store unique (request_type, cell_type) pairs
        
        for prediction_task in evaluator_request['prediction_tasks']:
            request_type = prediction_task['type']
            cell_type = prediction_task['cell_type']
            request_tasks.add((request_type, cell_type))
        
        print(f"Unique tasks extracted: {request_tasks}")
        
        # NOTE: ADDITION: Now extract prediction_ranges for all sequences, if provided
        prediction_ranges = evaluator_request.get('prediction_ranges', {})
        
        # Then run Borzoi Model ONCE for all required tracks
        print("Running Borzoi model on collected tasks...")
        task_predictions, matcher_version = predict_borzoi(
            sequences, request_tasks,
            matcher_ip, matcher_port,
            prediction_ranges, is_point_readout # NOTE ADDITION: Pass prediction ranges for proper handling downstream
            )
        
        # # Borzoi adds model_errors dictionary here, so if one task fails, 
        # # the other tasks will still return predictions. We need to check if 
        # # the overall task_predictions is an error message (str) or a dict of predictions.
        
        # if isinstance(task_predictions, str):
        #     # Wrap the error string into error payload 
        #     raise PredictionFailedError(task_predictions)
        
        # Pre-check: If the Matcher service suffered a fatal network failure, 
        # fail the entire request immediately before formatting the JSON.
        for task_data in task_predictions.values():
            if "error" in task_data and "dependent Matcher service" in task_data["error"]:
                from error_checking_functions import UpstreamDependencyError
                raise UpstreamDependencyError(task_data["error"])
        
        # Now format predictions to API JSON structure
        # Create JSON to return
        json_return = {
            'matcher_version': matcher_version,
            'bin_size': 32,
            # Prediction task is an array of objects for all requested tasks
            'prediction_tasks': []
        }
        
        # Loop through all the prediction tasks
        for prediction_task in evaluator_request['prediction_tasks']:
            task_name = prediction_task['name']
            request_type = prediction_task['type']
            cell_type = prediction_task['cell_type']
            
            # Determine Scale for predictions
            # Get requested scale
            requested_scale = prediction_task.get('scale')            

            # Retrieve the predictions for this task
            task_key = (request_type, cell_type)
            # Get raw result first
            task_result = task_predictions[task_key]
            
            # Build the base dictionary with all common keys
            predictions = {
                seq_id: result
                for seq_id, result in task_result.items()
                if seq_id not in ['track_indices', 'cell_type_actual', 'type_actual', 'trim_upstream']
            }
            
            if "error" in predictions:
                # Create structured response for the evaluator
                current_prediction_task = {
                    'name': task_name,
                    'type_requested': request_type,
                    'type_actual': "N/A",
                    'cell_type_requested': cell_type,
                    'cell_type_actual': "N/A",
                    'species_requested': prediction_task['species'],
                    'species_actual': prediction_task['species'],
                    'scale_prediction_requested': prediction_task.get('scale', None),
                    'scale_prediction_actual': "N/A",
                    'predictions': predictions  
                }
            else:
                
                # Sanitize any non-numeric values in predictions before applying scaling, to avoid errors during transformation
                predictions = _sanitize_for_json(predictions)
                _print_non_numeric(predictions)
                
                # Apply scale
                predictions_scaled, effective_scale = apply_scaling(predictions, requested_scale) 
                
                # Updating the logic here
                num_tracks_used = len(task_result['track_indices'])
                num_assay_types = len(task_result['type_actual'])
                aggregation = {}
                # Bin Aggregation (for point readouts)
                if is_point_readout:
                    aggregation["bins"] = "mean"
                # Cross-Assay Aggregation (e.g. DNASE + ATAC)
                if num_assay_types > 1:
                    aggregation["tracks"] = "mean"
                # Replicate Aggregation
                # If we have more physical tracks than assay types, we MUST have averaged replicates
                if num_tracks_used > num_assay_types:
                    aggregation["replicates"] = "mean"
                
                current_prediction_task = {
                    'name': task_name,
                    'type_requested': prediction_task['type'],
                    'type_actual': task_result['type_actual'],
                    'cell_type_requested': prediction_task['cell_type'],
                    'cell_type_actual': task_result['cell_type_actual'],
                    'species_requested': prediction_task['species'],
                    'species_actual': prediction_task['species'],
                    'scale_prediction_requested': requested_scale,
                    'scale_prediction_actual': effective_scale,
                    'predictions': predictions_scaled
                }
                
                # Only add aggregation if not empty
                if aggregation:
                    current_prediction_task['aggregation'] = aggregation
                
                # Conditionally add the 'trim_upstream' key
                if not is_point_readout: # This means 'track' readout
                    current_prediction_task['trim_upstream'] = task_result['trim_upstream'] 
            
            json_return['prediction_tasks'].append(current_prediction_task)
        
        final_payload = {"predictor_name": PREDICTOR_NAME,
                         **json_return}
        return encode_response(
            final_payload,
            status_code=200,
            predictor_name=PREDICTOR_NAME,
            supported_response_formats=SUPPORTED_RESPONSE_FORMATS)

        
    except Exception as e:
        # If it's already an APIError, re-raise it for the handler
        if isinstance(e, APIError):
            raise e
        # Otherwise, wrap the unknown error in a ServerError
        raise ServerError(f"An unexpected internal error occurred: {e}.")
    
# --- Run Flask ---
if __name__ == '__main__':
    # from waitress import serve
    print(f"{PREDICTOR_NAME} Predictor is running on http://{predictor_ip}:{predictor_port}")
    # serve(app, host=predictor_ip, port=predictor_port, threads=args.threads)
    app.run(host=predictor_ip, port=predictor_port, threaded=True, debug=False)