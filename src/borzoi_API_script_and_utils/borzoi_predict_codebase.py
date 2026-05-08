# borzoi_predict_codebase.py
import os
import sys
import copy
import json
import tqdm
import numpy as np
import pandas as pd

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
BUFFER = 32*32
HALF_BUFFER = int(BUFFER/2)
n_folds = 4  # Use all 4 model folds. Can vary between 1 and 4 (inclusive).
rc = True    # Reverse-complement predictions

# Load model parameters
def _load_model_parameters():
    with open(params_file) as params_open:
        params = json.load(params_open)
    return params['model'], params['train']

# Load target files
def _load_targets():
    targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')
    simplified_targets_df = pd.read_csv(simplified_targets_file, sep='\t')
    
    simplified_targets_df.columns = simplified_targets_df.columns.str.strip()
    
    return targets_df, simplified_targets_df

# Not filtering target index and slice_pair (Same as OG Borzoi codebase)
def _load_target_index():
    targets_df, _ = _load_targets()
    target_index = targets_df.index
    
    # Load strand pairing for reverse complement predictions
    if rc:
        strand_pair = targets_df.strand_pair
        target_slice_dict = {ix: i for i, ix in enumerate(target_index.values.tolist())}
        slice_pair = np.array([
            target_slice_dict[ix] if ix in target_slice_dict else ix for ix in strand_pair.values.tolist()
        ], dtype='int32')
        
    return target_index, slice_pair

# Initialize model ensemble
def _initialize_model_ensemble(target_index, slice_pair, params_model):
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
params_model, _ = _load_model_parameters()
target_index, slice_pair = _load_target_index()
models = _initialize_model_ensemble(target_index, slice_pair, params_model)

# Break predict_borzoi into multiple functions for better readability and testing. 
# The main function will be predict_borzoi, which will call helper functions for each step of the process (track selection, prediction, and formatting output). 
# This way, we can test each component separately and ensure that the overall logic is clear.


def _collect_required_track_indices(request_tasks, matcher_ip, matcher_port):
    """
    Maps requested tasks to the required track indices for prediction.

    Args:
        request_tasks (set): A set of strings (request_type, cell_type) pairs to determine required tracks.
        matcher_ip (str): The IP address of the Matcher server.
        matcher_port (int): The port number of the Matcher server.

    Returns:
        task_to_indices (dict): Dictionary to store required track indices from each task.
        unique_track_indices (list): Sorted list of all unique track indices needed for prediction.
        overall_matcher_version (str): The version of Matcher service used, or "N/A" if it was not called.
    """
    
    print("Collecting track indices for requested tasks...")
    task_to_indices = {} # Dictionary to store required track indices from each task
    # Example: {('expression', 'H1'): [1, 3], 
    #           ('accessibility', 'K562'): [2],
    #           ('expression_pol2', 'H1'): [1, 3]} (fallback to RNA:H1, 
    #                                               since there are no CAGE:H1 tracks)
    
    # track_to_tasks = defaultdict(set) # Maps each track index to a set of tasks that require it.
    #                                   # This prevents predicting on the same track twice.
    # # Example: {1: {('expression', 'H1'), ('expression_pol2', 'H1')},  # Track 1 needed by both expression tasks
    #           # 3: {('expression', 'H1'), ('expression_pol2', 'H1')},  # Track 3 needed by both expression tasks
    #           # 2: {('accessibility', 'K562')}}  # Track 2 needed only by accessibility task
    
    unique_track_indices = set() # Stores all the unique tracks needed for prediction
                                 # ensuring we only process relevant tracks once!
    # Example: [1, 2, 3]   

    # Initialize variable to hold Matcher version
    overall_matcher_version = "N/A"
    
    _, simplified_targets_df = _load_targets()

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
            
        unique_track_indices.update(track_indices)
        
    # Convert to sorted list to maintain order -- easy to test
    unique_track_indices = sorted(list(unique_track_indices))
    return task_to_indices, unique_track_indices, overall_matcher_version


def _encode_predict_fold_average(seq_chunk, unique_track_indices):
    # Pad and encode sequence
    encoded_seq = dna.dna_1hot(seq=seq_chunk, seq_len=SEQ_LEN)
    
    # Run model prediction once for all required tracks
    raw_pred = predict_tracks(models, encoded_seq)[:, :, :, unique_track_indices]
    
    # Average across model folds to reduce (1, n_folds, 16352, num_tracks) -> (1, 16352, num_tracks)
    fold_averaged_predictions = np.mean(raw_pred, axis=1).squeeze(0) # Shape (16352, num_tracks)
    
    return fold_averaged_predictions


def _predict_on_short_sequence(sequence, unique_track_indices, new_range_start=None, new_range_end=None):
    """
    Predict on a short sequence by centering it on the receptive field, 
    padding with N's, and then slicing the predictions to remove the N-padding bins.
    
    If prediction ranges are provided, the function will also slice the predictions
    to the new range after removing N-padding bins.
    
    Args:
        sequence (str): The input (subsetted) sequence.
        unique_track_indices (list): The list of unique track indices to predict on.
        new_range_start (int or None): The new start index for prediction after subsetting. Defaults to None.
        new_range_end (int or None): The new end index for prediction after subsetting. Defaults to None.
        
    Returns:
        predictions (np.array): The predicted values for the short sequence, 
                                sliced to remove N-padding bins and adjusted for 
                                prediction ranges if provided. 3D array of shape 
                                (1, num_sliced_bins, num_tracks).
        trim_upstream (int): Bases in first bin before sequence/range start.
    """
    pred = _encode_predict_fold_average(sequence, unique_track_indices)
    
    # Add the batch dimension back for the rest of the script
    pred = pred[np.newaxis, :, :] # Shape (1, 16352, num_tracks)
    
    if new_range_start is not None:
        # Slice to range bins
        sliced, trim_upstream = slice_prediction_tracks_for_range(
            full_track=pred,
            original_seq_len=len(sequence),
            model_input_len=MODEL_INPUT_LEN,
            bin_size=BIN_SIZE,
            range_start=new_range_start,
            range_end=new_range_end
        )
    else:
        # Slice to remove N-padding bins only
        sliced, trim_upstream = slice_prediction_tracks(
            full_track=pred,
            original_seq_len=len(sequence),
            model_input_len=MODEL_INPUT_LEN,
            bin_size=BIN_SIZE
        )
    return sliced, trim_upstream


def _predict_last_chunk_of_long_sequence(seq_chunk, unique_track_indices):
    """
    Handle a tail end chunk of a long sequence when the remaining sequence is 
    shorter than the MODEL_INPUT_LEN but COULD still be longer than the prediction window
    (CASES 2 and 3 for the last chunk of a long sequence -- they only differ in whether
    the tiling loop continues after this chunk or not, but the prediction logic is the same for both cases):
    
        - Pad downstream with N's to MODEL_INPUT_LEN and predict.
        - Then slice off the N-padded bins after prediction from the end.
        
    Args:
        seq_chunk (str): The last chunk of the long sequence that is shorter than MODEL_INPUT_LEN.
        unique_track_indices (list): The list of unique track indices to predict on.
        
    Returns:
        pred_chunk (np.array): The predicted values for the last chunk of the long sequence, 
                               sliced to remove N-padding bins from the end. Shape (num_sliced_bins, num_tracks).
    """
    downstream_pad =  MODEL_INPUT_LEN - len(seq_chunk)
    seq_chunk_padded = seq_chunk + ('N' * downstream_pad)
    
    pred_chunk = _encode_predict_fold_average(seq_chunk_padded, unique_track_indices)
    
    # Manually slice the N-padded bins from the end
    bases_to_crop_from_end = downstream_pad - HALF_BUFFER
    bins_to_crop_from_end = bases_to_crop_from_end // BIN_SIZE
    
    if bins_to_crop_from_end > 0:
        pred_chunk = pred_chunk[:-bins_to_crop_from_end, :]
        
    return pred_chunk


def _predict_on_long_sequence(sequence, unique_track_indices, new_range_start=None, new_range_end=None):
    """
    Predict on a long sequence by making multiple predictions on sliding windows,
    by tiling with N-padding such that bin 0 always corresponds to the first base of the sequence, 
    and then concatenating the predictions together.
    
    If prediction ranges are provided, slices the final concatenated predictions
    to the range bins and computes trim_upstream based on the new range start.
    
    Args:
        sequence (str): The input (subsetted) sequence.
        unique_track_indices (list): The list of unique track indices to predict on.
        new_range_start (int or None): The new start index for prediction after subsetting. Defaults to None.
        new_range_end (int or None): The new end index for prediction after subsetting. Defaults to None.
        
    Returns:
        predictions (np.array): The predicted values for the long sequence, 
                                sliced to remove N-padding bins, adjusted for 
                                prediction ranges if provided, and concatenated. 
                                3D array of shape (1, total, num_tracks).
        trim_upstream (int): Bases in first bin before sequence/range start.
    """
    predictions = []
    
    # If sequence is longer than prediction window then make multiple predictions.
    # Pad upstream of the sequence so that the first prediction corresponds 
    # to the first base on the sequence.
    sequence_with_upstream_pad = ('N' * (HALF_BUFFER)) + sequence
    
    # Mark how much of the actual sequence has been predicted on
    seq_predicted_end = HALF_BUFFER
    start_pos = 0
    
    while seq_predicted_end < len(sequence_with_upstream_pad):
        end_pos = min(len(sequence_with_upstream_pad), start_pos+MODEL_INPUT_LEN)
        seq_chunk = sequence_with_upstream_pad[start_pos:end_pos] # Current sequence chunk of length <= MODEL_INPUT_LEN
        
        # CASE 1 (full chunk) -- no padding needed
        if len(seq_chunk) == MODEL_INPUT_LEN:
            pred_chunk = _encode_predict_fold_average(seq_chunk, unique_track_indices)
            predictions.append(pred_chunk)
            start_pos += prediction_window # Slide the window by the prediction window (not the full model input length)
            seq_predicted_end += prediction_window
        
        # CASE 2 (last chunk > prediction_window)
        # -> Pad downstream with N's to MODEL_INPUT_LEN and then slice 
        #    off the N-padded bins after prediction 
        elif len(seq_chunk) > prediction_window:
            pred_chunk = _predict_last_chunk_of_long_sequence(seq_chunk, unique_track_indices)
            predictions.append(pred_chunk)
            start_pos += prediction_window # Slide the window by the prediction window (not the full model input length)
            seq_predicted_end += prediction_window
        
        # CASE 3 (last chunk <= prediction_window)
        # Both CASE 2 and CASE 3 are handled in the same way
        else:
            pred_chunk = _predict_last_chunk_of_long_sequence(seq_chunk, unique_track_indices)
            predictions.append(pred_chunk)
            break
    
    # Concatenate all chunks into one big track
    concatenated_predictions = np.concatenate(predictions, axis=0) # shape (total_bins, n_tracks)
    
    # Apply range slicing if prediction ranges are provided
    trim_upstream = 0
    if new_range_start is not None:
        start_bin = math.floor(new_range_start / BIN_SIZE)
        end_bin = math.ceil(new_range_end / BIN_SIZE)
        concatenated_predictions = concatenated_predictions[start_bin:end_bin, :]
        trim_upstream = new_range_start - (start_bin * BIN_SIZE)
        
    # Add the batch dimension back for the rest of the script
    concatenated_predictions = concatenated_predictions[np.newaxis, :, :] # Shape (1, total_bins, n_tracks)
    return concatenated_predictions, trim_upstream


def _assign_predictions_to_tasks(task_predictions, task_to_indices, predictions,
                                 seq_id, trim_upstream, unique_track_indices,
                                 is_point_readout):
    """
    For each task, extract the relevant track predictions from fold-averaged predictions,
    handle "point" vs "track" readout, and store them in task_predictions dictionary.
    
    Args:
        task_predictions (dict): The dictionary to store predictions for each task (deep copy of task_to_indices).
        task_to_indices (dict): The dictionary with task metadata with track_indices per task.
        predictions (np.array): The fold-averaged predictions for all required tracks. 3D array (1, num_bins, num_tracks).
        seq_id (str): The sequence ID for which predictions were made.
        trim_upstream (int): Bases in first bin before sequence/range start, to be stored for Evaluator alignment.
        unique_track_indices (list): The list of all unique track indices corresponding to the predictions.
        is_point_readout (bool): If True, aggregates track predictions to a single value per sequence.
    """
    
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
            task_predictions[task_key][seq_id] = np.round(predictions, 5).squeeze().tolist()
        else:
            # print(f"Assigning prediction for tasks: {task_key} (Tracks: {indices})")
            selected_tracks = predictions[:, :, 
                            [unique_track_indices.index(idx) for idx in indices]]
            # Average duplicate tracks per task
            # print(f"Averaging duplicate track predictions for task {task_key} (Tracks: {indices})")
            avg_prediction = np.mean(selected_tracks, axis=-1, keepdims=True)
            
            if is_point_readout:
                # "point" readout: Average across bins of interest to a single value per sequence
                print(f"Generating point readout for task: {task_key}")
                point_prediction = np.mean(avg_prediction, axis=1, keepdims=True)
                task_predictions[task_key][seq_id] = np.round(point_prediction, 5).squeeze().tolist()
            else:
                # "track" readout: Return predictions across bins of interest without averaging across them
                # Store predictions in task-specific dictionary
                task_predictions[task_key][seq_id] = np.round(avg_prediction, 5).squeeze().tolist()
                # Need to add trim upstream if it exists
                task_predictions[task_key].setdefault("trim_upstream", {})[seq_id] = trim_upstream


# Prediction Function -- Runs Once and Filters Predictions Based on Request Type
def predict_borzoi(sequences, request_tasks, matcher_ip=None, matcher_port=None,
                   prediction_ranges=None, is_point_readout=False):
    """
    Runs the Borzoi model on provided sequences and filters track predictions.
    
    Args:
        sequences (dict): A dictionary of key-value pairs {sequence_id: sequence}.
        request_tasks (set): A set of strings (request_type, cell_type) pairs 
                             to determine required tracks.
                             {(request_type1, cell_type1), (request_type2, cell_type2), ...}
        matcher_ip (str): The IP address of the Matcher server.
        matcher_port (int): The port number of the Matcher server.
        prediction_ranges (dict): A dictionary of key-value pairs {sequence_id: [start, end], ...} or None
                                  When provided, predictions will be sliced to the specified range
                                  for each sequence after prediction and trim_upstream will be calculated
                                  for Evaluator alignment.
        is_point_readout (bool): If True, aggregates track predictions to a single value.
    
    Returns:
        tuple: A tuple containing:
            - task_predictions (dict or str): On success, a dictionary of
              prediction results. On failure, an error message string.
            - overall_matcher_version (str): The version of the matcher
              service used, or "N/A" if it was not called.
            
            For optional "all_tracks" tasks, predictions are not averaged over tracks
            and the full prediction matrix is returned (shape [1, num_sliced_bins, 7611 tracks]).
    
    """
    if prediction_ranges is None:
        prediction_ranges = {}
    
    print("Running Borzoi Model Predictions on ALL tracks before filtering...")
    
    # 1. Collect all required track indices
    print("Collecting track indices for required tasks...")
    task_to_indices, unique_track_indices, overall_matcher_version = (
        _collect_required_track_indices(request_tasks, matcher_ip, matcher_port)
    )
    
    task_predictions = copy.deepcopy(task_to_indices)
    
    # If no tracks were found for any of the requested tasks return the metadata
    # with the errors stored in the predictions. Don't bother making any predictions.
    if not unique_track_indices:
        error_msg = "No valid track indices found for any tasks."
        print(error_msg)
        # for task_key, values in task_to_indices.items():
        #         task_predictions[task_key] = values
        return task_predictions, overall_matcher_version
    
    print(f"Unique track indices: {len(unique_track_indices)} tracks.")
    
    # 2. Process each sequence and run prediction
    #    - Iterate over sequences and run model prediction only for the required tracks
    # Process each sequence
    for seq_id, sequence in tqdm.tqdm(sequences.items(),
                                      desc="Predictions in progress", 
                                      unit="sequence",
                                      total=len(sequences),
                                      dynamic_ncols=True):
        print(f"\nPredicting on sequence ID: {seq_id} ({len(sequence)} bp)")
        
        # Check for prediction ranges for this sequence
        pred_range = prediction_ranges.get(seq_id, [])
        new_range_start = None
        new_range_end = None    
        
        # Subset sequence if prediction range is provided
        if len(pred_range) > 0:
            print(f"Requested prediction range for {seq_id}: {pred_range}")
            sequence, new_range_start, new_range_end = subset_sequence_for_ranges(
                sequence, pred_range, prediction_window, context_flank=HALF_BUFFER
                )
            # print(f"After subsetting: {len(sequence)} bp; new_range: [{new_range_start}, {new_range_end}]")
        
        # Now we have the subsetted sequence and new prediction range. We can run prediction on the subsetted sequence and then slice the predictions to the new range.
        
        # If sequence is shorter than prediction window then centre it on the
        # receptive field and predict. Crop N bins from both sides
        if len(sequence) <= prediction_window:
            predictions, trim_upstream = _predict_on_short_sequence(
                sequence, unique_track_indices, new_range_start, new_range_end
                )
        else:
            predictions, trim_upstream = _predict_on_long_sequence(
                sequence, unique_track_indices, new_range_start, new_range_end
            )
        # print(f"DEBUG: Predictions shape {predictions.shape}, trim_upstream: {trim_upstream}")
        
        # Assign predictions to tasks
        _assign_predictions_to_tasks(
            task_predictions, task_to_indices, predictions,
            seq_id, trim_upstream, unique_track_indices,
            is_point_readout
        )
        
    return task_predictions, overall_matcher_version

# ============================================================================
# Test Suite: predict_borzoi math, range slicing, and invariants
# ============================================================================
if __name__ == "__main__":
    """
    Tests for prediction ranges, trim_upstream math, and sequence tiling.
    Uses the REAL borzoi_utils functions. No model calls needed.

    Borzoi architecture reference:
        Input (SEQ_LEN):             524,288 bp
        Effective input window:      524,288 bp
        Prediction window:           523,264 bp  (16,352 bins * 32 bp)
        Buffer per side:             512 bp      (16 bins * 32 bp)
        BIN_SIZE:                    32 bp
    """
    import math
    import numpy as np

    # -- Constants ----------------------------------------------------------
    _MIL = SEQ_LEN             # 524,288
    _PW  = prediction_window   # 523,264
    _BS  = BIN_SIZE            # 32
    _SC  = HALF_BUFFER         # 512
    _BUF = 1024                # 1024 (total buffer)
    _NBINS = 16352             # 16,352

    # -- Helpers ------------------------------------------------------------
    def _labeled(n_bins, n_tracks=2):
        """Mock prediction array where each bin's value = its index."""
        a = np.zeros((1, n_bins, n_tracks))
        for i in range(n_bins):
            a[0, i, :] = i
        return a

    def _simulate_tail_chunk_math(seq_len):
        """Simulates the math inside _predict_tail_chunk for long sequences."""
        seq_chunk_len = seq_len + _SC # Accounts for the prepended upstream buffer
        downstream_pad = _MIL - seq_chunk_len
        bases_to_crop = max(0, downstream_pad - _SC)
        bins_to_crop = bases_to_crop // _BS
        bins_kept = _NBINS - bins_to_crop
        return bins_kept

    # -- Test Runner -------------------------------------------------------
    _p, _f = 0, 0
    def _ok(cond, name, detail=""):
        global _p, _f
        if cond:
            _p += 1; print(f"  ✓ {name}")
        else:
            _f += 1; print(f"  ✗ {name}")
            if detail: print(f"    {detail}")

    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 1: subset_sequence_for_ranges")
    print("=" * 70)

    # 1a: Standard subset (2M bp sequence, requested [500k, 510k])
    sub, rs, re = subset_sequence_for_ranges("A"*2_000_000, [500_000, 510_000], _PW, _SC)
    _ok(len(sub) == _MIL,   f"2M bp → subsetted to exactly {_MIL}", f"Got {len(sub)}")
    _ok(rs == 257_144,      f"new_range_start = 257,144",     f"Got {rs}")
    _ok(re == 267_144,      f"new_range_end = 267,144",       f"Got {re}")
    _ok(re - rs == 10_000,  "Range size preserved (10,000)",  f"Got {re-rs}")

    # 1b: Short sequence — should bypass subsetting math
    sub2, rs2, re2 = subset_sequence_for_ranges("A"*50_000, [10_000, 30_000], _PW, _SC)
    _ok(len(sub2) == 50_000,                   "50k: no trimming needed", f"Got {len(sub2)}")
    _ok(rs2 == 10_000 and re2 == 30_000,       "Range coords perfectly preserved")

    # 1c: Massive range (larger than prediction window)
    sub3, rs3, _ = subset_sequence_for_ranges("A"*2_000_000, [200_000, 800_000], _PW, _SC)
    exp3 = (800_000 + _SC) - (200_000 - _SC)
    _ok(len(sub3) == exp3,  f"Massive range: len = range + 2*context ({exp3})", f"Got {len(sub3)}")
    _ok(rs3 == _SC,         f"Massive range: new_range_start = {_SC}", f"Got {rs3}")

    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 2: slice_prediction_tracks (Short Seq, No Ranges)")
    print("=" * 70)

    p16352 = _labeled(_NBINS)

    # 2a: 1000 bp
    sl, tu = slice_prediction_tracks(p16352, 1000, _MIL, _BS, _BUF)
    _ok(sl.shape[1] == 32,              f"1000 bp → 32 bins", f"Got {sl.shape[1]}")
    _ok(sl.shape[1]*_BS >= 1000,        f"Covers ≥ 1000 bp ({sl.shape[1]*_BS})")
    _ok(tu == 12,                       f"trim_upstream = 12", f"Got {tu}")
    _ok(int(sl[0,0,0]) == 8160,         f"First bin index = 8160", f"Got {int(sl[0,0,0])}")

    # 2b: 200 bp (e.g., MPRA element)
    sl3, tu3 = slice_prediction_tracks(p16352, 200, _MIL, _BS, _BUF)
    _ok(sl3.shape[1] == 8,              "200 bp → 8 bins", f"Got {sl3.shape[1]}")
    _ok(sl3.shape[1]*_BS >= 200,        f"Covers ≥ 200 bp ({sl3.shape[1]*_BS})")
    _ok(tu3 == 28,                      f"trim_upstream = 28", f"Got {tu3}")

    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 3: slice_prediction_tracks_for_range (Short Seq, Ranges)")
    print("=" * 70)

    # 3a: 50k bp seq, range [10k, 30k]
    sl_r, tu_r = slice_prediction_tracks_for_range(
        p16352, 50_000, 10_000, 30_000, _MIL, _BS, _BUF)
    _ok(sl_r.shape[1] == 626,          f"50k [10k,30k]: 626 bins", f"Got {sl_r.shape[1]}")
    _ok(sl_r.shape[1]*_BS >= 20_000,   f"Covers ≥ 20,000 bp ({sl_r.shape[1]*_BS})")
    _ok(tu_r == 8,                     f"trim_upstream = 8", f"Got {tu_r}")

    # 3b: Bin-aligned range (trim should be 0)
    # 50k seq output offset = 236632. 236632 % 32 = 24. 
    # Range start of 8 aligns perfectly to a 32bp boundary (24+8=32).
    sl_a, tu_a = slice_prediction_tracks_for_range(
        p16352, 50_000, 8, 30_000, _MIL, _BS, _BUF)
    _ok(tu_a == 0, "Bin-aligned range: trim_upstream = 0", f"Got {tu_a}")

    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Odd-Length Sequence Centering Math")
    print("=" * 70)

    # 4a: 1001 bp (Odd length)
    sl_odd, tu_odd = slice_prediction_tracks(p16352, 1001, _MIL, _BS, _BUF)
    _ok(sl_odd.shape[1] == 32,          f"1001 bp → 32 bins", f"Got {sl_odd.shape[1]}")
    _ok(tu_odd == 11,                   f"Odd trim_upstream = 11", f"Got {tu_odd}")
    _ok(tu_odd == tu - 1,               "Odd trim = Even trim - 1 (Shifts sequence left 1bp)")

    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 5: Long Sequence Tail-Chunk Math Invariant (CASE 2/3)")
    print("=" * 70)
    print("Proving that padding downstream 'N's and cropping bins exactly restores original length.")

    # 5a: Sequence exactly 100,000 bp long
    seq_len = 100_000
    kept_bins = _simulate_tail_chunk_math(seq_len)
    recovered_bp = kept_bins * _BS
    _ok(recovered_bp == seq_len, 
        f"Cropped tail bins perfectly recover sequence length: {kept_bins} bins * 32 = {recovered_bp} bp")

    # 5b: Edge case: Sequence exactly 1 bp long (Extremely short tail)
    kept_bins_1 = _simulate_tail_chunk_math(1)
    _ok(kept_bins_1 * _BS == 32, 
        f"1 bp tail recovers exactly 1 bin (32 bp) to cover the single base.")

    # ======================================================================
    print("\n" + "=" * 70)
    print("TEST 6: Long Sequence Tiling Range Slicing")
    print("=" * 70)

    # Mock a concatenated tiled track of 1_000_000 bp (~31250 bins)
    p_tiled = _labeled(31250)

    # 6a: Range [93304, 103304] 
    start_bin = math.floor(93304 / _BS)
    end_bin   = math.ceil(103304 / _BS)
    sl_tiled  = p_tiled[:, start_bin:end_bin, :]
    tu_tiled  = 93304 - (start_bin * _BS)

    _ok(sl_tiled.shape[1] == 314,      "314 bins cover 10k bp",   f"Got {sl_tiled.shape[1]}")
    _ok(int(sl_tiled[0,0,0]) == 2915,  "First bin index = 2915",  f"Got {int(sl_tiled[0,0,0])}")
    _ok(tu_tiled == 24,                "trim_upstream = 24",      f"Got {tu_tiled}")

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 70)
    print(f"RESULTS: {_p} passed, {_f} failed out of {_p + _f}")
    print("=" * 70)
    if _f:
        print("\n❌  FAILURES — review output above.")
    else:
        print("\n✅ All Borzoi math invariants successfully verified!")
