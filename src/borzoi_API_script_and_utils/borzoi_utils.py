# borzoi_utils.py
import math
import requests

# Function to handle Evaluator request
# Fed into the model by Predictor

MATCHER_NULL_RESPONSE = "NULL"

class MatcherNotConfiguredError(Exception):
    """
    Custom exception raised when an exact match failed and the Matcher is 
    called, but not configured.
    """
    pass

def slice_prediction_tracks(full_track, original_seq_len, model_input_len, bin_size, buffer_bp=1024):
    
    """
    Sliced a full prediction track to keep only bins corresponding to with just
    N-padding predictions removed.
    
    Args:
        full_track (np.array): The 3D full-length prediction array from the model (1, 16352, num_tracks)
        original_seq_len (int): The length of original, unpadded sequence.
        model_input_len (int): The length of the sequence required by the model
                               after padding/ trimming.
        bin_size (int): The size of each prediction bin in base pairs.
        buffer_bp (int): The total number of base pairs cropped by the model.
        
    Returns:
        sliced_track (np.array): The sliced 3D prediction track with bins with just
                                 N-padding removed.
        N_in_left_bin (int): Number of N-padding in the leftmost bin with sequence. This is needed for `track` readouts, which has to be incorporated in borzoi_predict_codebase
        
    """
    
    # Calculate the total padding
    total_padding = model_input_len - original_seq_len
    
    left_buffer = buffer_bp//2
    # Sequence is centred but padding is right-biased (Extra N on the right, if total padding is odd)
    left_padding = total_padding // 2
    left_padding_after_buffer = max(0, left_padding - left_buffer)
    
    # Calculate the indices of bins containing the sequence
    start_bin_index = left_padding_after_buffer // bin_size
    end_bin_index = math.ceil((left_padding_after_buffer + original_seq_len)/bin_size)
    
    number_bins_with_seq = end_bin_index - start_bin_index
    total_bases = number_bins_with_seq*bin_size
    N_in_bins = total_bases - original_seq_len
    N_in_left_bin = N_in_bins//2
    
    sliced_track = full_track[:, start_bin_index:end_bin_index, :]
    
    return sliced_track, N_in_left_bin

def matcher_communication(matcher_ip, matcher_port, message_for_Matcher):
    """
    Helper function to send a single request to the Matcher API's /match endpoint.
    Checks for Matcher configuration before attempting to connect.
    
    Args:
        matcher_ip (str): The IP address of the Matcher server.
        matcher_port (int): The port number of the Matcher server.
        message_for_Matcher (dict): The JSON payload to send.
        
    Returns:
        dict: The JSON response from the Matcher.
        
    Raises:
        MatcherNotConfiguredError: If matcher_ip or matcher_port are None.
        requests.exceptions.HTTPError: If the Matcher returns a 4xx or 5xx status.
        requests.exceptions.RequestException: For connection errors, timeouts, etc.
    """
    
    if not matcher_ip or not matcher_port:
        raise MatcherNotConfiguredError("Matcher service is not configured.")
    
    try:
        # Build Matcher URL
        matcher_url = f"http://{matcher_ip}:{matcher_port}"
        endpoint = f"{matcher_url}/match"
        
        response = requests.post(endpoint, json=message_for_Matcher)
        
        # Raise an exception for bad status codes (4xx, 5xx)
        response.raise_for_status()
        
        # Return the parsed JSON response
        return response.json()

    except requests.exceptions.HTTPError as e:
        print(f"Matcher API returned an error: {e.response.status_code} {e.response.text}")
        raise ConnectionError(f"Matcher API returned an error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to Matcher at {endpoint}")
        raise ConnectionError(f"Matcher service unavailable: {e}")


def filter_evaluator_request(simplified_targets_df, request_type, cell_type, matcher_ip=None, matcher_port=None, molecule=None):
    
    """
    Filters evaluator request based on assay type, cell type, and molecule.
    
    Args:
        simplified_targets_df (pd.DataFrame): Data Frame containing simplified target data.
        request_type (str): Requested type of prediction:
            - "accessibility": Uses ATAC and DNASE (concatenated)
            - "expression", "expression_mrna", "expression_pol1", "expression_pol3": Uses RNA
            - "expression_pol2": Uses CAGE (with RNA fallback)
            - "binding_{molecule}": Uses CHIP assay with specified molecule.
            - (HIDDEN VALUE FOR KEY type: "all_tracks": Return all available tracks. 
               Overrides the provided cell type.)
        cell_type (str): Requested cell type for prediction.
        matcher_ip (str): The IP address of the Matcher server.
        matcher_port (int): The port number of the Matcher server.
        molecule (str, optional): TF binding/ histone modification molecule for ChIP-Seq requests.
        
    Returns:
        tuple: A tuple containing four elements:
            - pd.DataFrame or str: A DataFrame of filtered tracks or an error string.
            - str or None: The actual cell type used (requested or matched).
            - list or None: The actual assay type used (requested or matched; e.g. ["RNA"] or ["ATAC", "DNASE"]).
            - str: The version of the matcher service used, or "N/A".
    """
    request_error_msg = f"Request Error: No tracks in the requested type: {request_type} and cell type: {cell_type} found."
    
    try: 
        print(f"Received evaluator request from Predictor to filter desired tracks\
            \n Type Requested: {request_type},\
            \n Cell Type: {cell_type}")
        
        # Normalize inputs to lowercase for case-insensitive handling
        request_type = request_type.lower() if request_type else None
        cell_type = cell_type.lower() if cell_type else None
        
        # Special case: if request_type is "all_tracks", return all available tracks, no matter the cell type
        if request_type == "all_tracks":
            print("All tracks request detected: returning all available tracks for prediction.")
            return simplified_targets_df, "all", ["all"], "N/A"
        
        # Define TF binding/ histone modification molecule for ChIP-Seq
        molecule = request_type.split("_")[1] if request_type.startswith("binding_") else None
        molecule = molecule.lower() if molecule else None
        print(f"TF Binding/ Histone Modification (if any, else None): {molecule}")
        
        # 1. Accessibility (Parse both, ATAC and DNASE, tracks and concatenate)
        #check for exact cell_type match in both ATAC-seq or DNase
        if request_type == "accessibility":
            print(f"Parsing both ATAC and DNASE tracks for cell type provided: {cell_type}")
            accessibility_tracks = simplified_targets_df[
                (simplified_targets_df['Assay'].isin(['ATAC', 'DNASE'])) &
                (simplified_targets_df['Cell Type'].str.lower() == cell_type)
            ]
            # if an exact match was found that the cell_type_actual = cell_type_requested 
            
            if not accessibility_tracks.empty:
                request_actual = accessibility_tracks['Assay'].unique().tolist()
                return accessibility_tracks, cell_type, request_actual, "N/A"

            # if no exact match was found use the Matcher module to find a closely related cell type
            if accessibility_tracks.empty:
                print(f"No exact matching cell types in ATAC-seq/DNAse assay for cell type: {cell_type}. Querying Matcher for similar cell types in ATAC and DNASE tracks.")
                
                # NOTE: Send request to Matcher here -- We only care about cell-type and binding_{molecule} matching at the moment for the heatmap
                # Species matching, while the Matcher supports it, this script does not call Matcher for species-matching.
                # Filter out all accessibility tracks -- ATAC and DNAse
                all_accessibility_tracks = simplified_targets_df[(simplified_targets_df['Assay'].isin(['ATAC', 'DNASE']))]
                # set up any dictionary to send to matcher
                message_for_Matcher = {
                    'cell_type_requested': cell_type,
                    'cell_type_list': all_accessibility_tracks['Cell Type'].unique().tolist()
                    }
                
                matcher_result = matcher_communication(matcher_ip, matcher_port, message_for_Matcher)
                matcher_version = matcher_result.get('matcher_version', 'UnknownMatcher')

                # matcher could not find any closely related cell_types
                # NOTE: adding more error checks and using .get(), which will return NoneType if missing, which is seemingly safer for type errors
                if not matcher_result or not matcher_result.get('cell_type_actual') or matcher_result.get('cell_type_actual') == MATCHER_NULL_RESPONSE:
                    print("No similar cell types were found using Matcher")
                    return request_error_msg, None, None, matcher_version
                else:
                    matched_cell_type = matcher_result['cell_type_actual']
                    print(f"Matcher cell type will now be used for accessibility: {matched_cell_type}")
                    
                    matched_tracks = simplified_targets_df[
                        (simplified_targets_df['Assay'].isin(['ATAC', 'DNASE'])) &
                        (simplified_targets_df['Cell Type'].str.lower() == matched_cell_type.lower())
                    ]

                    return (
                        matched_tracks, matched_cell_type, (matched_tracks['Assay'].unique().tolist()), matcher_version
                        ) if not matched_tracks.empty else (request_error_msg, None, None, matcher_version)
                        
        # 2. Expression (RNA for all request_type, except "expression_pol2")
        elif request_type in ["expression", "expression_mrna", "expression_pol1", "expression_pol3"]:
            rna_tracks = simplified_targets_df[
                (simplified_targets_df['Assay'] == 'RNA') &
                (simplified_targets_df['Cell Type'].str.lower() == cell_type)
            ]
            if not rna_tracks.empty:
                return rna_tracks, cell_type, ["RNA"], "N/A"
            
            # If no exact match was found, use the Matcher module
            if rna_tracks.empty:
                print(f"No exact matching cell types for RNA assay for {cell_type}. Querying Matcher.")
                all_rna_tracks = simplified_targets_df[simplified_targets_df['Assay'] == 'RNA']
                message_for_Matcher = {
                    'cell_type_requested': cell_type,
                    'cell_type_list': all_rna_tracks['Cell Type'].unique().tolist()
                    }
                matcher_result = matcher_communication(matcher_ip, matcher_port, message_for_Matcher)
                matcher_version = matcher_result.get('matcher_version', 'UnknownMatcher')

                if not matcher_result or not matcher_result.get('cell_type_actual') or matcher_result.get('cell_type_actual') == MATCHER_NULL_RESPONSE:
                    print("No similar cell types were found using Matcher")
                    return request_error_msg, None, None, matcher_version
                else:
                    matched_cell_type = matcher_result['cell_type_actual']
                    print(f"Matcher cell type will now be used for RNA: {matched_cell_type}")
                    
                    matched_tracks = simplified_targets_df[
                        (simplified_targets_df['Assay'] == 'RNA') &
                        (simplified_targets_df['Cell Type'].str.lower() == matched_cell_type.lower())
                    ]
                    return (matched_tracks, matched_cell_type, ["RNA"], matcher_version) if not matched_tracks.empty else (request_error_msg, None, None, matcher_version)
            
        # 3. Expression Pol2 (Parse CAGE with RNA as fallback)
        elif request_type == "expression_pol2":
            # Try to find CAGE tracks
            print("Attempting to find CAGE tracks")
            cage_tracks = simplified_targets_df[
                (simplified_targets_df['Assay'] == 'CAGE') &
                (simplified_targets_df['Cell Type'].str.lower() == cell_type)
            ]
            if not cage_tracks.empty:
                return cage_tracks, cell_type, ["CAGE"], "N/A"
            
            # If no exact CAGE match, query the Matcher for a CAGE substitute
            if cage_tracks.empty:
                print(f"No exact CAGE match for {cell_type}. Querying Matcher for CAGE.")
                all_cage_tracks = simplified_targets_df[simplified_targets_df['Assay'] == 'CAGE']
                message_for_Matcher = {
                    'cell_type_requested': cell_type,
                    'cell_type_list': all_cage_tracks['Cell Type'].unique().tolist()
                    }
                matcher_result_cage = matcher_communication(matcher_ip, matcher_port, message_for_Matcher)
                matcher_version = matcher_result_cage.get('matcher_version', 'UnknownMatcher')
                
                if matcher_result_cage and matcher_result_cage.get('cell_type_actual') and matcher_result_cage.get('cell_type_actual') != MATCHER_NULL_RESPONSE:
                    matched_cell_type = matcher_result_cage['cell_type_actual']
                    print(f"Matcher cell type will be used for CAGE: {matched_cell_type}")
                    matched_cage_tracks = simplified_targets_df[
                        (simplified_targets_df['Assay'] == 'CAGE') &
                        (simplified_targets_df['Cell Type'].str.lower() == matched_cell_type.lower())
                        ]
                    
                    if not matched_cage_tracks.empty:
                        return matched_cage_tracks, matched_cell_type, ["CAGE"], matcher_version
            
            # Fallback to RNA if CAGE matching fails completely
            print("No matching cell types in CAGE assay type. Falling back to RNA assay.")
            rna_tracks = simplified_targets_df[
                (simplified_targets_df['Assay'] == 'RNA') &
                (simplified_targets_df['Cell Type'].str.lower() == cell_type)
            ]
            if not rna_tracks.empty:
                    return rna_tracks, cell_type, ["RNA"], "N/A"
                
            if rna_tracks.empty:
                print(f"No exact RNA match for {cell_type}. Querying Matcher for RNA fallback.")
                all_rna_tracks = simplified_targets_df[simplified_targets_df['Assay'] == 'RNA']
                message_for_Matcher = {
                    'cell_type_requested': cell_type,
                    'cell_type_list': all_rna_tracks['Cell Type'].unique().tolist()
                    }
                matcher_result_rna = matcher_communication(matcher_ip, matcher_port, message_for_Matcher)
                matcher_version = matcher_result_rna.get('matcher_version', 'UnknownMatcher')

                if not matcher_result_rna or not matcher_result_rna.get('cell_type_actual') or matcher_result_rna.get('cell_type_actual') == MATCHER_NULL_RESPONSE:
                    print("No similar cell types were found using Matcher")
                    return request_error_msg, None, None, matcher_version
                else:
                    matched_cell_type = matcher_result_rna['cell_type_actual']
                    print(f"Matcher cell type will be used for RNA fallback: {matched_cell_type}")
                    
                    matched_rna_tracks = simplified_targets_df[
                        (simplified_targets_df['Assay'] == 'RNA') &
                        (simplified_targets_df['Cell Type'].str.lower() == matched_cell_type.lower())
                        ]
                    return (matched_rna_tracks, matched_cell_type, ["RNA"], matcher_version) if not matched_rna_tracks.empty else (request_error_msg, None, None, matcher_version)
                
        # 4. Binding -- binding_{molecule}
        # (Parse CHIP assays, filter out TF binding/ histone modification molecule and cell_type)
        elif request_type.startswith("binding_"):
            chip_tracks = simplified_targets_df[
                (simplified_targets_df['Assay'] == 'CHIP') &
                (simplified_targets_df['Molecule'].str.lower() == molecule) &
                (simplified_targets_df['Cell Type'].str.lower() == cell_type)
            ]

            if not chip_tracks.empty:
                # request_actual_assay = chip_tracks['Assay'].unique()
                # request_actual_molecule = chip_tracks['Molecule'].unique()
                # return chip_tracks, cell_type, f"{request_actual_assay}_{request_actual_molecule}" # CHIP_{molecule}
                return chip_tracks, cell_type, [f"CHIP_{molecule}"], "N/A"

            else:
                # If no exact match, try to match the molecule first
                print(f"No exact matching cell type and molecule pairs in CHIP assay: {cell_type}, {molecule}. Querying Matcher for similar molecules.")
                #Send request to Matcher here
                all_chip_tracks = simplified_targets_df[
                    (simplified_targets_df['Assay'] == 'CHIP')
                    ]
                #set up any dictionary to send to matcher
                message_for_Matcher = {
                    'binding_molecule_requested': molecule,
                    'binding_molecule_list': all_chip_tracks['Molecule'].unique().tolist()
                    }
                
                matcher_result_molecule = matcher_communication(matcher_ip, matcher_port, message_for_Matcher)
                matcher_version = matcher_result_molecule.get('matcher_version', 'UnknownMatcher')
                
                # matcher could not find any closely related cell_types
                if not matcher_result_molecule or not matcher_result_molecule.get('binding_molecule_actual') or matcher_result_molecule.get('binding_molecule_actual') == MATCHER_NULL_RESPONSE:
                    print("No similar molecule tracks were found using Matcher.")
                    return request_error_msg, None, None, matcher_version
                
                else:
                    # Got a matched molecule to proceed with
                    matched_molecule = matcher_result_molecule['binding_molecule_actual']
                    print(f"Matcher molecule will now be used for CHIP: {matched_molecule}")
                    
                    # With the matched molecule, try an exact match on the ORIGINAL cell type again.
                    chip_tracks = simplified_targets_df[
                    (simplified_targets_df['Assay'] == 'CHIP') &
                    (simplified_targets_df['Molecule'].str.lower() == matched_molecule.lower()) &
                    (simplified_targets_df['Cell Type'].str.lower() == cell_type)
                    ]   
                    
                    # If empty request Matcher to map cell_type from the newly filtered tracks
                    if not chip_tracks.empty:
                        return chip_tracks, cell_type, [f"CHIP_{matched_molecule}"], matcher_version
                    else:
                        # If that still fails, then call the matcher to map cell-type
                        # Send request to Matcher to map cell type next
                        print(f"No exact matching cell types in CHIP assay for Matcher-mapped molecule: {matched_molecule}. Querying Matcher for similar cell type in CHIP tracks.")            
                        #Send request to Matcher here
                        chip_tracks_molecule_mapped = simplified_targets_df[
                            (simplified_targets_df['Assay'] == 'CHIP') &
                            (simplified_targets_df['Molecule'].str.lower() == matched_molecule.lower())
                        ]
                        #set up any dictionary to send to matcher
                        message_for_Matcher = {
                            'cell_type_requested': cell_type,
                            'cell_type_list': chip_tracks_molecule_mapped['Cell Type'].unique().tolist()
                            }
                        
                        matcher_result_cell_type = matcher_communication(matcher_ip, matcher_port, message_for_Matcher)
                        matcher_version = matcher_result_cell_type.get('matcher_version', 'UnknownMatcher')
                        
                        # Use the robust check for the cell type result
                        if not matcher_result_cell_type or not matcher_result_cell_type.get('cell_type_actual') or matcher_result_cell_type.get('cell_type_actual') == MATCHER_NULL_RESPONSE:
                            print("No similar cell types were found using Matcher for the specified molecule.")
                            return request_error_msg, None, None, matcher_version
                        
                        else:
                            matched_cell_type = matcher_result_cell_type['cell_type_actual']
                            print(f"Matcher cell type will now be used for CHIP_{matched_molecule}: {matched_cell_type}")

                            chip_tracks_cell_type_mapped = simplified_targets_df[
                                (simplified_targets_df['Assay'] == 'CHIP') &
                                (simplified_targets_df['Molecule'].str.lower() == matched_molecule.lower()) &
                                (simplified_targets_df['Cell Type'].str.lower() == matched_cell_type.lower())

                            ]

                            return (chip_tracks_cell_type_mapped,matched_cell_type, [f"CHIP_{matched_molecule}"], matcher_version) if not chip_tracks_cell_type_mapped.empty else (request_error_msg, None, None, matcher_version)
                
        # Invalid request type
        else:
            raise ValueError(f"Invalid request type {request_type}")
        
    except MatcherNotConfiguredError as e:
        print(f"No exact match found and Matcher is not configured to perform this task: {e}")
        return request_error_msg, None, None, "N/A"

    except ConnectionError as e:
        print(f"A fatal error occurred while communicating with the Matcher: {e}")
        error_message = f"Internal Server Error: The dependent Matcher service at {matcher_ip}:{matcher_port} is unavailable."
        # Return a 4-element tuple to match the success signature and avoid crashing the caller
        return error_message, None, None, "error"
