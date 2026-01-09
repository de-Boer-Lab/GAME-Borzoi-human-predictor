# borzoi_predict_codebase.py
import os
import sys
import json
import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict

BORZOI_SCRIPT_DIR = os.path.dirname(__file__)

sys.path.append(BORZOI_SCRIPT_DIR)
from borzoi_utils import *

sys.path.append(f"{BORZOI_SCRIPT_DIR}/baskerville/src")
from baskerville import seqnn
from baskerville import dna

sys.path.append(f"{BORZOI_SCRIPT_DIR}/borzoi/examples")
from borzoi_helpers import *

params_file = f"{BORZOI_SCRIPT_DIR}/borzoi/examples/params_pred.json"
targets_file = f"{BORZOI_SCRIPT_DIR}/borzoi/examples/targets_human.txt"

utils_path = f"{BORZOI_SCRIPT_DIR}/simplify_targets"
simplified_targets_file = f"{utils_path}/borzoi_human_targets_simplified.txt"
# Simplified targets file was created to easily map the requested type and cell type
# to the right tracks. The python script for that is in `simplify_targets/` directory.

saved_models_path = f"{BORZOI_SCRIPT_DIR}/borzoi/examples/saved_models"

# Sequence parameters
SEQ_LEN = 524288
MODEL_INPUT_LEN = SEQ_LEN
prediction_window = SEQ_LEN - 1024
BIN_SIZE = 32
HALF_BUFFER = int(1024/2)
n_folds = 4  # Use all 4 model folds. Can vary between 1 and 4 (inclusive).
rc = True    # Reverse-complement predictions

# 1. Load model parameters
def load_model_parameters():
    with open(params_file) as params_open:
        params = json.load(params_open)
    return params['model'], params['train']

# 2. Load target files
def load_targets():
    targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')
    simplified_targets_df = pd.read_csv(simplified_targets_file, sep='\t')
    
    simplified_targets_df.columns = simplified_targets_df.columns.str.strip()
    
    return targets_df, simplified_targets_df

# 3. Not filtering target index and slice_pair (Same as OG Borzoi codebase)
def load_target_index():
    targets_df, _ = load_targets()
    target_index = targets_df.index
    
    # Load strand pairing for reverse complement predictions
    if rc:
        strand_pair = targets_df.strand_pair
        target_slice_dict = {ix: i for i, ix in enumerate(target_index.values.tolist())}
        slice_pair = np.array([
            target_slice_dict[ix] if ix in target_slice_dict else ix for ix in strand_pair.values.tolist()
        ], dtype='int32')
        
    return target_index, slice_pair

# 4. Initialize model ensemble
def initilize_model_ensemble(target_index, slice_pair, params_model):
    models = []
    for fold_ix in range(n_folds) :

        model_file = f"{saved_models_path}/f3c{str(fold_ix)}/train/model0_best.h5"

        seqnn_model = seqnn.SeqNN(params_model)
        seqnn_model.restore(model_file, 0)
        seqnn_model.build_slice(target_index)
        if rc:
            seqnn_model.strand_pair.append(slice_pair)
        #seqnn_model.build_ensemble(rc, '0')
        seqnn_model.build_ensemble(rc, [0])
        models.append(seqnn_model)
        
    return models

# Load parameters, target indices, and models
params_model, _ = load_model_parameters()
target_index, slice_pair = load_target_index()
models = initilize_model_ensemble(target_index, slice_pair, params_model)

# 5. Prediction Function -- Runs Once and Filters Predictions Based on Request Type
def predict_borzoi(sequences, request_tasks, matcher_ip=None, matcher_port=None, is_point_readout=False):
    """
    Runs the Borzoi model on provided sequences and filters track predictions.

    This function processes a batch of sequences, determines the necessary model
    output tracks based on the requested tasks, calls a matcher service if needed,
    runs the model prediction once, and formats the output.
    
    Args:
        sequences (dict): A dictionary of key-value pairs {sequence_id: sequence}.
        request_tasks (set): A set of strings (request_type, cell_type) pairs 
                             to determine required tracks.
        matcher_ip (str): The IP address of the Matcher server.
        matcher_port (int): The port number of the Matcher server.
        is_point_readout (bool): If True, aggregates track predictions to a single
                                 value.
    
    Returns:
        tuple: A tuple containing:
            - task_predictions (dict or str): On success, a dictionary of
              prediction results. On failure, an error message string.
            - overall_matcher_version (str): The version of the matcher
              service used, or "N/A" if it was not called.
            
            For "all_tracks" tasks, predictions are not averaged over tracks
            and the full prediction matrix is returned (shape [1, num_sliced_bins, 7611 tracks]).
    
    """
    print("Running Borzoi Model Predictions on ALL tracks before filtering...")
    
    # 5.1. Collect all required track indices
    print("Collecting track indices for required tasks...")
    task_to_indices = {} # Dictionary to store required track indices from each task
    # Example: {('expression', 'H1'): [1, 3], 
    #           ('accessibility', 'K562'): [2],
    #           ('expression_pol2', 'H1'): [1, 3]} (fallback to RNA:H1, 
    #                                               since there are no CAGE:H1 tracks)
    
    track_to_tasks = defaultdict(set) # Maps each track index to a set of tasks that require it.
                                      # This prevents predicting on the same track twice.
    # Example: {1: {('expression', 'H1'), ('expression_pol2', 'H1')},  # Track 1 needed by both expression tasks
    #           3: {('expression', 'H1'), ('expression_pol2', 'H1')},  # Track 3 needed by both expression tasks
    #           2: {('accessibility', 'K562')}}  # Track 2 needed only by accessibility task
    
    unique_track_indices = set() # Stores all the unique tracks needed for prediction
                                 # ensuring we only process relevant tracks once!
    # Example: [1, 2, 3]
    
    # Initialize variable to hold Matcher version
    overall_matcher_version = "N/A"
    
    _, simplified_targets_df = load_targets()

    for request_type, cell_type in request_tasks:
        print(f"Performing track selection for {request_type} and {cell_type}...")
        # Get track indices of desired tracks for filtering predictions
        filtered_tracks, cell_type_actual, type_actual, task_matcher_version = filter_evaluator_request(simplified_targets_df,
                                                request_type, cell_type, matcher_ip, matcher_port) # NOTE: Filter function will call the Matcher
        
        if task_matcher_version not in ["N/A", "error"]:
            overall_matcher_version = task_matcher_version
        
        task_key = (request_type, cell_type)
        
        if isinstance(filtered_tracks, str):
            # The function failed, and 'filtered_tracks' now holds the error message.
            # Return this error message to the API to be sent to the client.
            print(f"No matching tracks found for {request_type} and {cell_type}. Skipping...")
            task_to_indices[task_key] = {"error": filtered_tracks}
            continue
        else:    
            # Otherwise, proceed as before -- knowing filtered_tracks is a DataFrame
            track_indices = filtered_tracks.index.tolist()
            if not track_indices:
                print(f"No matching tracks found for {request_type} and {cell_type}. Skipping...")
                continue
            # Avoid printing a huge list for "all_tracks" requests:
            if request_type.lower() == "all_tracks":
                print(f"Using all {len(track_indices)} track indices for ({request_type}, {cell_type}).")
            else:
                print(f"Using Track Indices for ({request_type}, {cell_type}): {track_indices}")
            
            task_to_indices[task_key] = {
                'track_indices': track_indices,
                'cell_type_actual': cell_type_actual,
                'type_actual': type_actual
            }
            
            for index in track_indices:
                track_to_tasks[index].add(task_key) # Mapping track to tasks
                
            unique_track_indices.update(track_indices)
        
    # Convert to sorted list to maintain order -- easy to test
    unique_track_indices = sorted(list(unique_track_indices))
    
    # Check if any request is an all_tracks request
    if any(rt.lower() == "all_tracks" for rt, _ in request_tasks):
        print(f"Unique required track indices for this task: ALL {len(unique_track_indices)} tracks.")
    else:
        print(f"Unique required track indices for all tasks: {unique_track_indices}")
    
    task_predictions = task_to_indices
    
    # If no tracks were found for any of the requested tasks return the metadata
    # with the errors stores in the predictions. Don't bother making any predictions.
    if not unique_track_indices:
        error_msg = "No valid track indices found for any tasks."
        print(error_msg)
        for task_key, values in task_to_indices.items():
                task_predictions[task_key] = values
        return task_predictions, overall_matcher_version
    
    # 5.2. Process each sequence and run prediction
    #    - Iterate over sequences and run model prediction only for the required tracks
    print("Processing sequences and storing predictions only for required tracks...")
    # Process each sequence
    for seq_id, sequence in tqdm.tqdm(sequences.items(),
                                      desc="Predictions in progress", 
                                      unit="sequence",
                                      total=len(sequences),
                                      dynamic_ncols=True):
        print(f"Predicting on sequence ID: {seq_id} 🧬")
        
        # ----- TRIM UPSTREAM SLICING CODE -----
        trim_upstream = 0 # Will be 0 for long path, or calculated for short path
        
        # If sequence is shorter than prediction window then centre it on the
        # receptive field and predict. Crop N bins from both sides
        if len(sequence) <= prediction_window:
            # Pad and encode sequence
            encoded_seq = dna.dna_1hot(seq=sequence, seq_len=SEQ_LEN)
            # print(f"Shape of encoded sequence before predict_tracks: {encoded_seq.shape}")
            
            # Run model prediction once for all required tracks
            raw_predictions = predict_tracks(models, encoded_seq)[:, :, :, unique_track_indices]
            # print(f"Shape of raw predictions: {raw_predictions.shape}")
            
            # Average across model folds to reduce (1, n_folds, 16352, num_tracks) -> (1, 16352, num_tracks)
            fold_averaged_predictions = np.mean(raw_predictions, axis=1)
            # print(f"Shape of predictions after averaging model folds: {fold_averaged_predictions.shape}")
            
            # This is where the predictions should be sliced and the only bins with N-padding removed 
            # (1, 16352, num_tracks) -> (1, num_sliced_bins, num_tracks)
            fold_averaged_predictions_sliced, trim_upstream = slice_prediction_tracks(
                full_track=fold_averaged_predictions,
                original_seq_len=len(sequence),
                model_input_len=MODEL_INPUT_LEN,
                bin_size=BIN_SIZE
            )
            
        else:
            predictions = []
            # If sequence is longer than prediction window then make multiple predictions.
            # Pad upstream of the sequence so that the first prediction corresponds 
            # to the first base on the sequence.
            sequence_with_upstream_pad = ('N' * (HALF_BUFFER)) + sequence
            
            # Mark how much of the actual sequence has been predicted on
            seq_predicted_end = HALF_BUFFER
            start_pos = 0
            
            while seq_predicted_end < len(sequence_with_upstream_pad):
                #The position is either the start position of the current sequence or the end of the sequence
                end_pos = min(len(sequence_with_upstream_pad), start_pos+MODEL_INPUT_LEN)
                
                # Current sequence
                seq_chunk = sequence_with_upstream_pad[start_pos:end_pos]
                
                # CASE 1 (Chunk if full) If sequence fits prediction with, no padding needed
                if len(seq_chunk) == MODEL_INPUT_LEN:
                    # Pad and encode sequence
                    encoded_seq = dna.dna_1hot(seq=seq_chunk, seq_len=SEQ_LEN)
                    # print(f"Shape of encoded sequence before predict_tracks: {encoded_seq.shape}")
                    
                    # Run model prediction once for all required tracks
                    raw_pred_chunk = predict_tracks(models, encoded_seq)[:, :, :, unique_track_indices]
                    pred_chunk = np.mean(raw_pred_chunk, axis=1).squeeze(0) # Shape (16352, n_tracks)
                    
                    print(f"Pred chunk shape: {pred_chunk.shape}")
                    predictions.append(pred_chunk)
                    
                    # Slide the window
                    seq_predicted_end = seq_predicted_end + prediction_window 
                    start_pos = start_pos + prediction_window
                
                # CASE 2 and 3 (Last chunk): If the amount of sequence you can pull is less than model input length
                else:
                    print("For sequence chunks shorter than the model's input length")
                    
                    # --- Case 2: Chunk is still larger than the output window ---
                    if len(seq_chunk) > prediction_window:
                        print("Last chunk > prediction_window. Padding downstream.")
                        downstream_pad = MODEL_INPUT_LEN - len(seq_chunk)
                        seq_chunk_downstreamN = seq_chunk + ('N' * downstream_pad)

                        encoded_seq = dna.dna_1hot(seq=seq_chunk_downstreamN, seq_len=SEQ_LEN)
                        
                        raw_pred_chunk = predict_tracks(models, encoded_seq)[:, :, :, unique_track_indices]
                        pred_chunk = np.mean(raw_pred_chunk, axis=1).squeeze(0) # Shape (16352, n_tracks)

                        predictions.append(pred_chunk)
                        
                        # Slide window (this will end the loop)
                        start_pos = start_pos + prediction_window
                        seq_predicted_end = seq_predicted_end + prediction_window
                        
                    # --- Case 3: Chunk is smaller than the output window ---
                    elif len(seq_chunk) <= prediction_window:
                        print("Last chunk <= prediction_window. Padding and slicing.")
                        downstream_pad = MODEL_INPUT_LEN - len(seq_chunk)
                        seq_chunk_downstreamN = seq_chunk + ('N' * downstream_pad)

                        encoded_seq = dna.dna_1hot(seq=seq_chunk_downstreamN, seq_len=SEQ_LEN)
                        
                        raw_pred_chunk = predict_tracks(models, encoded_seq)[:, :, :, unique_track_indices]
                        pred_chunk = np.mean(raw_pred_chunk, axis=1).squeeze(0) # Shape (16352, n_tracks)
                        
                        # Manually slice the N-padded bins from the end
                        bases_to_crop_from_end = downstream_pad - HALF_BUFFER
                        bins_to_crop_from_end = bases_to_crop_from_end // BIN_SIZE 
                        
                        if bins_to_crop_from_end == 0:
                            predictions.append(pred_chunk)
                        else:
                            # Crop the N-padded bins from the end
                            predictions.append(pred_chunk[:-bins_to_crop_from_end, :])

                        # This is the last chunk, so we break
                        break
                    
            # Now, concatenate all chunks into one big track
            fold_averaged_predictions_sliced = np.concatenate(predictions, axis=0)
            
            # Manually trim the final track to the exact number of bins
            # for the original sequence length (just in case of rounding)
            total_bins = int(np.ceil(len(sequence) / BIN_SIZE))
            fold_averaged_predictions_sliced = fold_averaged_predictions_sliced[:total_bins, :]
            
            # print(f"Final concatenated predictions shape: {fold_averaged_predictions_sliced.shape}")
            
            # Add the batch dimension back for the rest of the script
            fold_averaged_predictions_sliced = fold_averaged_predictions_sliced[np.newaxis, :, :] # Shape (1, N_bins, n_tracks)
            
        # --- TRIM UPSTREAM SLICE CODE ENDS ---
        
        # Now assign filtered predictions to each task to be averaged
        for task_key, values in task_to_indices.items():
            
            # Check if this task had an error during track selection
            if "error" in values:
                task_predictions[task_key] = values # Keep the error message
                continue # Skip to the next task
            
            indices = values['track_indices']
            # Extract relevant track predictions per task
            # Special case: for "all_tracks" request, return full predictions without averaging over tracks
            if task_key[0].lower() == "all_tracks":
                # print(f"Assigning prediction for tasks: {task_key} (All tracks: [1, {len(indices)}])")
                task_predictions[task_key][seq_id] = np.vectorize(lambda x: float(f"{x:.5f}"))(fold_averaged_predictions_sliced).squeeze().tolist()
            else:
                # print(f"Assigning prediction for tasks: {task_key} (Tracks: {indices})")
                selected_tracks = fold_averaged_predictions_sliced[:, :, 
                                [unique_track_indices.index(idx) for idx in indices]]
                # Average duplicate tracks per task
                # print(f"Averaging duplicate track predictions for task {task_key} (Tracks: {indices})")
                avg_prediction = np.mean(selected_tracks, axis=-1, keepdims=True)
                
                if is_point_readout:
                    # "point" readout: Average across (16352 bins - N-padding bins) to a single value per sequence
                    print(f"Generating point readout for task: {task_key}")
                    point_prediction = np.mean(avg_prediction, axis=1, keepdims=True)
                    # task_predictions[task_key][seq_id] = np.vectorize(lambda x: float(f"{x:.5f}"))(point_prediction).squeeze().tolist()
                    task_predictions[task_key][seq_id] = np.round(point_prediction, 5).squeeze().tolist()
                else:
                    # "track" readout: Return full (16352 bins - N-padding bins) predictions
                    # Store predictions in task-specific dictionary
                    # task_predictions[task_key][seq_id] = np.vectorize(lambda x: float(f"{x:.5f}"))(avg_prediction).squeeze().tolist()
                    # Just round and convert. It is 100x faster.
                    task_predictions[task_key][seq_id] = np.round(avg_prediction, 5).squeeze().tolist()
                    # Need to add trim upstream if it exists
                    task_predictions[task_key].setdefault("trim_upstream", {})[seq_id] = trim_upstream
    
    return task_predictions, overall_matcher_version
