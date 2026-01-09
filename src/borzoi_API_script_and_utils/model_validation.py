import os
import sys
import numpy as np

MODEL_VALIDATION_DIR = os.path.dirname(__file__)
ERROR_FUNCTIONS_DIR = os.path.abspath(os.path.join(MODEL_VALIDATION_DIR, "..", "script_and_utils"))

if ERROR_FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, ERROR_FUNCTIONS_DIR)
    
from error_checking_functions import PredictionFailedError

# Define what this specific model supports globally
SUPPORTED_SCALES = ["linear", "log"]
DEFAULT_SCALE = "linear"

# A set of all exact types this Borzoi model supports
SUPPORTED_EXACT_TYPES = {
    "expression",
    "expression_pol1",
    "expression_pol2",
    "expression_pol3",
    "expression_mrna",
    "accessibility",
    "all_tracks" # HIDDEN KEY
}

# A string for the error message
SUPPORTED_TYPES_LIST_MSG = "['expression', 'expression_pol[1-3]', 'expression_mrna', 'accessibility', 'binding_[molecule]']"

def model_specific_payload_validation(payload):
    
    errors = {'prediction_request_failed': []}

    readout_type = payload['readout']

    # Handle unsupported `interaction_matrix` readout
    if readout_type == "interaction_matrix":
        print("Borzoi cannot handle 'interaction_matrix' readout type. Exiting gracefully!")
        errors['prediction_request_failed'].append("Borzoi cannot process 'interaction_matrix' readout type.")
        

    # --- MODEL SPECIFIC: Ensure this Borzoi Predictor only supports homo_sapiens ---
    for task in payload['prediction_tasks']:
        if task.get('species', '').lower() != "homo_sapiens":
            errors['prediction_request_failed'].append(
                f"This predictor only supports species: homo_sapiens. Received '{task.get('species')}' for task '{task.get('name')}'."
            )
        task_type = task.get('type', '').lower()
        if task_type.startswith("binding_"):
            pass
        elif task_type in SUPPORTED_EXACT_TYPES:
            pass
        else:
            # This type is not supported by this specific model
            errors['prediction_request_failed'].append(
                f"This predictor does not support type: '{task.get('type')}'. "
                f"Supported types are {SUPPORTED_TYPES_LIST_MSG} for task '{task.get('name')}'."
            )
            
        # --- MODEL-SPECIFIC: Determine the scale of the prediction requested ---
        # This can change for other Predictors, in which case they should remap scale_prediction_actual
        # and specify explicitly what base they are using, if scale is logarithmic.
        # Must be 'linear' or 'log'.
        req_scale = task.get('scale')
        if req_scale and req_scale.lower() not in SUPPORTED_SCALES:
            errors['prediction_request_failed'].append(
                f"Unsupported scale: '{req_scale}'. Supported scales are: {SUPPORTED_SCALES}."
            )
    
    #If you want to add error checking that restricts sequences with N bases, add that here
    if any(errors.values()):
        flagged_errors = [msg for sublist in errors.values() for msg in sublist]
        raise PredictionFailedError(flagged_errors)

def apply_scaling(predictions_dict, requested_scale):
    """
    Applies scaling transformation specific to THIS model's output and returns the applied scale name.
    
    Borzoi Default Output: Linear
    Logic: 
      - If 'linear' requested: Do nothing.
      - If 'log' requested: log2(x + 1)
      
    Args:
        predictions_dict (dict): The raw linear predictions
        requested_scale (str or None): The scale requested by the user
        
    Returns:
        tuple: (transformed_dict, actual_scale_str)
    """
    
    # Determine Effective Scale
    if not requested_scale:
        # Default if None provided
        effective_scale = DEFAULT_SCALE
    else:
        effective_scale = requested_scale.lower()
    
    if effective_scale == "linear":
        return predictions_dict, "linear"
    
    transformed_preds = {}
    for seq_id, values in predictions_dict.items():
        # Convert to numpy for fast vectorized math
        arr = np.array(values)
        
        if effective_scale == "log":
            arr = np.log2(arr + 1)
        
        # Convert back to list for JSON serialization
        transformed_preds[seq_id] = arr.tolist()
        
    return transformed_preds, effective_scale
    